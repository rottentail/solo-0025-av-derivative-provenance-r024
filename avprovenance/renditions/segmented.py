"""Checkpointed segmented execution of normalized media plans.

A segmented rendition splits the source timeline into fixed-size sample or
frame segments and renders one operation stage at a time. Every rendered
segment is stored as a content-addressed blob and recorded in a durable
render session, so an interrupted process can resume from the last
checkpoint instead of starting over.

The engine is deterministic by construction: per-sample and per-frame
operations are applied to exactly the same values as the whole-clip
renderer, and ``audio_normalize`` first derives the global peak across all
input segments so every segment shares one gain factor. Assembling the
final segments therefore yields bytes identical to a single uninterrupted
run of the same plan.
"""

from __future__ import annotations

from typing import Any, Callable

from ..errors import RenderFailed, UnsupportedMedia
from ..media.audio_ops import (
    PCM_MAX,
    audio_peak,
    gain_audio,
    mix_audio,
    scale_audio,
)
from ..media.errors import MediaOperationFailure
from ..media.video_ops import crop_video, mirror_video, offset_luma
from ..media.wav import AudioClip, decode_wav, encode_wav
from ..media.y4m import Y4mClip, Y4mFrame, decode_y4m, encode_y4m
from ..serialization import deep_copy, sha256_hex, utc_now


SegmentStore = Callable[[bytes], tuple[str, str]]
SegmentLoader = Callable[[str], bytes]
SessionPersist = Callable[[dict[str, Any]], None]


def new_render_session(
    rendition_id: str,
    media_kind: str,
    operations: list[dict[str, Any]],
    *,
    segment_frames: int,
    segments_per_run: int,
) -> dict[str, Any]:
    now = utc_now()
    return {
        "rendition_id": rendition_id,
        "media_kind": media_kind,
        "segment_frames": int(segment_frames),
        "segments_per_run": int(segments_per_run),
        "stream": None,
        "stage_index": 0,
        "stages": [
            {
                "operation": deep_copy(operation),
                "status": "pending",
                "input_segments": None,
                "input_frame_count": None,
                "output_frame_count": None,
                "trim_start": None,
                "analysis": None,
                "output_segments": [],
                "next_segment_index": 0,
            }
            for operation in operations
        ],
        "segments_completed": 0,
        "created_at": now,
        "updated_at": now,
    }


def session_content_keys(session: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for stage in session.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        for field in ("input_segments", "output_segments"):
            for ref in stage.get(field) or []:
                key = ref.get("key")
                if key:
                    keys.add(str(key))
    return keys


def render_progress(session: dict[str, Any]) -> dict[str, Any]:
    stages = list(session.get("stages") or [])
    stage_index = int(session.get("stage_index", 0))
    current = stages[stage_index] if stage_index < len(stages) else None
    total = None
    if current is not None and current.get("output_frame_count") is not None:
        segment_frames = int(session["segment_frames"])
        total = -(-int(current["output_frame_count"]) // segment_frames)
    return {
        "mode": "segmented",
        "segment_frames": int(session["segment_frames"]),
        "segments_per_run": int(session["segments_per_run"]),
        "stage_index": stage_index,
        "stage_count": len(stages),
        "stage_status": str(current["status"]) if current is not None else "done",
        "stage_segments_completed": (
            int(current["next_segment_index"]) if current is not None else 0
        ),
        "stage_segments_total": total,
        "segments_completed": int(session.get("segments_completed", 0)),
    }


class _RunContext:
    def __init__(
        self,
        primary_bytes: bytes,
        secondary_bytes: bytes | None,
    ) -> None:
        self.primary_bytes = primary_bytes
        self.secondary_bytes = secondary_bytes
        self._secondary_clip: AudioClip | None = None

    def secondary_clip(self) -> AudioClip:
        if self.secondary_bytes is None:
            raise MediaOperationFailure(
                "AUDIO_MIX_SOURCE_MISSING",
                "audio_mix requires a secondary media asset.",
            )
        if self._secondary_clip is None:
            self._secondary_clip = decode_wav(self.secondary_bytes)
        return self._secondary_clip


def _load_ref(ref: dict[str, Any], load: SegmentLoader) -> bytes:
    data = load(str(ref["key"]))
    if sha256_hex(data) != str(ref["digest"]):
        raise MediaOperationFailure(
            "RENDER_CHECKPOINT_CORRUPT",
            "A checkpointed segment does not match its recorded digest.",
        )
    return data


def _read_audio_ref(ref: dict[str, Any], load: SegmentLoader) -> AudioClip:
    return decode_wav(_load_ref(ref, load))


def _read_video_ref(ref: dict[str, Any], load: SegmentLoader) -> Y4mClip:
    return decode_y4m(_load_ref(ref, load))


def _chunk_audio(
    clip: AudioClip,
    segment_frames: int,
    store: SegmentStore,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for start in range(0, clip.frame_count, segment_frames):
        end = min(clip.frame_count, start + segment_frames)
        segment = AudioClip(
            sample_rate=clip.sample_rate,
            channels=tuple(values[start:end] for values in clip.channels),
        )
        key, digest = store(encode_wav(segment))
        refs.append({"key": key, "digest": digest, "frame_count": end - start})
    return refs


def _chunk_video(
    clip: Y4mClip,
    segment_frames: int,
    store: SegmentStore,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for start in range(0, clip.frame_count, segment_frames):
        end = min(clip.frame_count, start + segment_frames)
        segment = Y4mClip(
            width=clip.width,
            height=clip.height,
            rate_numerator=clip.rate_numerator,
            rate_denominator=clip.rate_denominator,
            interlace=clip.interlace,
            pixel_format=clip.pixel_format,
            frames=clip.frames[start:end],
        )
        key, digest = store(encode_y4m(segment))
        refs.append({"key": key, "digest": digest, "frame_count": end - start})
    return refs


def _audio_stream(clip: AudioClip) -> dict[str, Any]:
    return {"sample_rate": clip.sample_rate, "channels": clip.channel_count}


def _video_stream(clip: Y4mClip) -> dict[str, Any]:
    return {
        "width": clip.width,
        "height": clip.height,
        "rate_numerator": clip.rate_numerator,
        "rate_denominator": clip.rate_denominator,
        "interlace": clip.interlace,
        "pixel_format": clip.pixel_format,
    }


def _prepare_stage(
    session: dict[str, Any],
    stage_index: int,
    context: _RunContext,
    store: SegmentStore,
    load: SegmentLoader,
) -> None:
    stage = session["stages"][stage_index]
    operation = stage["operation"]
    operation_type = str(operation["type"])
    kind = session["media_kind"]
    segment_frames = int(session["segment_frames"])

    if stage_index == 0:
        if kind == "audio":
            clip = decode_wav(context.primary_bytes)
            session["stream"] = _audio_stream(clip)
            stage["input_segments"] = _chunk_audio(clip, segment_frames, store)
            stage["input_frame_count"] = clip.frame_count
        else:
            clip = decode_y4m(context.primary_bytes)
            session["stream"] = _video_stream(clip)
            stage["input_segments"] = _chunk_video(clip, segment_frames, store)
            stage["input_frame_count"] = clip.frame_count
    else:
        previous = session["stages"][stage_index - 1]
        stage["input_segments"] = previous["output_segments"]
        stage["input_frame_count"] = sum(
            int(ref["frame_count"]) for ref in previous["output_segments"]
        )
        previous["output_segments"] = []
        previous["input_segments"] = None

    input_count = int(stage["input_frame_count"])
    if operation_type == "media_trim":
        if kind == "audio":
            sample_rate = int(session["stream"]["sample_rate"])
            start = int(round(float(operation["start_seconds"]) * sample_rate))
            duration = int(round(float(operation["duration_seconds"]) * sample_rate))
            out_of_range = "AUDIO_TRIM_OUT_OF_RANGE"
            empty = "AUDIO_TRIM_EMPTY"
        else:
            start = int(operation["start_frame"])
            duration = int(operation["frame_count"])
            out_of_range = "VIDEO_TRIM_OUT_OF_RANGE"
            empty = "VIDEO_TRIM_EMPTY"
        if start < 0 or start >= input_count:
            raise MediaOperationFailure(
                out_of_range,
                "The requested trim starts outside the source.",
            )
        end = min(input_count, start + max(1, duration))
        if end <= start:
            raise MediaOperationFailure(
                empty,
                "The requested trim produces no frames.",
            )
        stage["trim_start"] = start
        stage["output_frame_count"] = end - start
    elif operation_type == "audio_mix":
        secondary = context.secondary_clip()
        stream = session["stream"]
        if int(stream["sample_rate"]) != secondary.sample_rate:
            raise MediaOperationFailure(
                "AUDIO_MIX_RATE_MISMATCH",
                "Mixed audio sources must share a sample rate.",
            )
        if int(stream["channels"]) != secondary.channel_count:
            raise MediaOperationFailure(
                "AUDIO_MIX_CHANNEL_MISMATCH",
                "Mixed audio sources must share a channel count.",
            )
        stage["output_frame_count"] = max(input_count, secondary.frame_count)
    elif operation_type == "audio_normalize":
        peak = 0
        for ref in stage["input_segments"]:
            peak = max(peak, audio_peak(_read_audio_ref(ref, load)))
        if peak == 0:
            stage["analysis"] = {"peak": 0, "factor": None}
        else:
            target = float(operation["target_peak"])
            desired = max(1.0, target * PCM_MAX)
            stage["analysis"] = {"peak": peak, "factor": desired / peak}
        stage["output_frame_count"] = input_count
    elif operation_type == "video_crop":
        x = int(operation["x"])
        y = int(operation["y"])
        width = int(operation["width"])
        height = int(operation["height"])
        if any(value % 2 for value in (x, y, width, height)):
            raise MediaOperationFailure(
                "VIDEO_CROP_ALIGNMENT",
                "Video crop geometry must align to even pixels.",
            )
        if width < 2 or height < 2:
            raise MediaOperationFailure(
                "VIDEO_CROP_TOO_SMALL",
                "Crop width and height must be at least two pixels.",
            )
        stream = session["stream"]
        if x + width > int(stream["width"]) or y + height > int(stream["height"]):
            raise MediaOperationFailure(
                "VIDEO_CROP_OUT_OF_RANGE",
                "Crop geometry exceeds the source frame.",
            )
        stream["width"] = width
        stream["height"] = height
        stage["output_frame_count"] = input_count
    elif operation_type in {
        "audio_gain",
        "video_luma_offset",
        "video_mirror",
    }:
        stage["output_frame_count"] = input_count
    else:
        raise MediaOperationFailure(
            "UNSUPPORTED_STAGE_OPERATION",
            f"Unsupported staged operation: {operation_type}",
        )
    stage["status"] = "running"


def _gather_audio(
    session: dict[str, Any],
    stage: dict[str, Any],
    load: SegmentLoader,
    start: int,
    end: int,
) -> AudioClip:
    segment_frames = int(session["segment_frames"])
    refs = stage["input_segments"]
    first = start // segment_frames
    last = (end - 1) // segment_frames
    stream = session["stream"]
    channels: list[list[int]] = [
        [] for _ in range(int(stream["channels"]))
    ]
    for index in range(first, last + 1):
        clip = _read_audio_ref(refs[index], load)
        for channel_index, values in enumerate(clip.channels):
            channels[channel_index].extend(values)
    base = first * segment_frames
    return AudioClip(
        sample_rate=int(stream["sample_rate"]),
        channels=tuple(
            values[start - base : end - base] for values in channels
        ),
    )


def _gather_video_frames(
    session: dict[str, Any],
    stage: dict[str, Any],
    load: SegmentLoader,
    start: int,
    end: int,
) -> tuple[Y4mFrame, ...]:
    segment_frames = int(session["segment_frames"])
    refs = stage["input_segments"]
    first = start // segment_frames
    last = (end - 1) // segment_frames
    frames: list[Y4mFrame] = []
    for index in range(first, last + 1):
        frames.extend(_read_video_ref(refs[index], load).frames)
    base = first * segment_frames
    return tuple(frames[start - base : end - base])


def _video_clip(session: dict[str, Any], frames: tuple[Y4mFrame, ...]) -> Y4mClip:
    stream = session["stream"]
    return Y4mClip(
        width=int(stream["width"]),
        height=int(stream["height"]),
        rate_numerator=int(stream["rate_numerator"]),
        rate_denominator=int(stream["rate_denominator"]),
        interlace=str(stream["interlace"]),
        pixel_format=str(stream["pixel_format"]),
        frames=frames,
    )


def _render_audio_segment(
    session: dict[str, Any],
    stage: dict[str, Any],
    segment_index: int,
    context: _RunContext,
    load: SegmentLoader,
) -> bytes | dict[str, Any]:
    operation = stage["operation"]
    operation_type = str(operation["type"])
    segment_frames = int(session["segment_frames"])
    begin = segment_index * segment_frames
    end = min(int(stage["output_frame_count"]), begin + segment_frames)

    if operation_type == "media_trim":
        trim_start = int(stage["trim_start"])
        clip = _gather_audio(session, stage, load, trim_start + begin, trim_start + end)
        return encode_wav(clip)
    if operation_type == "audio_gain":
        segment = _read_audio_ref(stage["input_segments"][segment_index], load)
        return encode_wav(gain_audio(segment, operation))
    if operation_type == "audio_normalize":
        factor = stage["analysis"]["factor"]
        if factor is None:
            return dict(stage["input_segments"][segment_index])
        segment = _read_audio_ref(stage["input_segments"][segment_index], load)
        return encode_wav(scale_audio(segment, float(factor)))
    if operation_type == "audio_mix":
        stream = session["stream"]
        input_count = int(stage["input_frame_count"])
        if begin < input_count:
            primary = _read_audio_ref(stage["input_segments"][segment_index], load)
        else:
            primary = AudioClip(
                sample_rate=int(stream["sample_rate"]),
                channels=tuple(
                    [] for _ in range(int(stream["channels"]))
                ),
            )
        secondary = context.secondary_clip()
        secondary_segment = AudioClip(
            sample_rate=secondary.sample_rate,
            channels=tuple(
                values[begin:end] for values in secondary.channels
            ),
        )
        return encode_wav(mix_audio(primary, secondary_segment, operation))
    raise MediaOperationFailure(
        "UNSUPPORTED_AUDIO_OPERATION",
        f"Unsupported audio operation: {operation_type}",
    )


def _render_video_segment(
    session: dict[str, Any],
    stage: dict[str, Any],
    segment_index: int,
    load: SegmentLoader,
) -> bytes:
    operation = stage["operation"]
    operation_type = str(operation["type"])
    segment_frames = int(session["segment_frames"])
    begin = segment_index * segment_frames
    end = min(int(stage["output_frame_count"]), begin + segment_frames)

    if operation_type == "media_trim":
        trim_start = int(stage["trim_start"])
        frames = _gather_video_frames(
            session,
            stage,
            load,
            trim_start + begin,
            trim_start + end,
        )
        return encode_y4m(_video_clip(session, frames))
    segment = _read_video_ref(stage["input_segments"][segment_index], load)
    if operation_type == "video_crop":
        return encode_y4m(crop_video(segment, operation))
    if operation_type == "video_luma_offset":
        return encode_y4m(offset_luma(segment, operation))
    if operation_type == "video_mirror":
        return encode_y4m(mirror_video(segment, operation))
    raise MediaOperationFailure(
        "UNSUPPORTED_VIDEO_OPERATION",
        f"Unsupported video operation: {operation_type}",
    )


def _render_segment(
    session: dict[str, Any],
    stage: dict[str, Any],
    context: _RunContext,
    store: SegmentStore,
    load: SegmentLoader,
) -> None:
    segment_index = int(stage["next_segment_index"])
    if session["media_kind"] == "audio":
        produced = _render_audio_segment(
            session,
            stage,
            segment_index,
            context,
            load,
        )
    else:
        produced = _render_video_segment(session, stage, segment_index, load)
    if isinstance(produced, dict):
        stage["output_segments"].append(produced)
    else:
        key, digest = store(produced)
        segment_frames = int(session["segment_frames"])
        begin = segment_index * segment_frames
        end = min(int(stage["output_frame_count"]), begin + segment_frames)
        stage["output_segments"].append(
            {"key": key, "digest": digest, "frame_count": end - begin}
        )
    stage["next_segment_index"] = segment_index + 1


def _expected_segments(session: dict[str, Any], stage: dict[str, Any]) -> int:
    segment_frames = int(session["segment_frames"])
    return -(-int(stage["output_frame_count"]) // segment_frames)


def _assemble(session: dict[str, Any], load: SegmentLoader) -> bytes:
    stage = session["stages"][-1]
    refs = stage["output_segments"]
    stream = session["stream"]
    if session["media_kind"] == "audio":
        channels: list[list[int]] = [
            [] for _ in range(int(stream["channels"]))
        ]
        for ref in refs:
            clip = _read_audio_ref(ref, load)
            for channel_index in range(len(channels)):
                channels[channel_index].extend(clip.channels[channel_index])
        return encode_wav(
            AudioClip(
                sample_rate=int(stream["sample_rate"]),
                channels=tuple(channels),
            )
        )
    frames: list[Y4mFrame] = []
    for ref in refs:
        frames.extend(_read_video_ref(ref, load).frames)
    return encode_y4m(_video_clip(session, tuple(frames)))


def run_segmented_session(
    session: dict[str, Any],
    *,
    primary_bytes: bytes,
    secondary_bytes: bytes | None,
    store: SegmentStore,
    load: SegmentLoader,
    persist: SessionPersist,
) -> bytes | None:
    """Advance a render session by at most its recorded segment budget.

    Returns the assembled output bytes when the final stage completes, or
    ``None`` when the budget is exhausted and the session was persisted as
    resumable progress.
    """

    context = _RunContext(primary_bytes, secondary_bytes)
    budget = int(session["segments_per_run"])
    stages = session["stages"]
    while True:
        stage_index = int(session["stage_index"])
        if stage_index >= len(stages):
            return _assemble(session, load)
        stage = stages[stage_index]
        if stage["status"] == "pending":
            _prepare_stage(session, stage_index, context, store, load)
            session["updated_at"] = utc_now()
            persist(session)
            continue
        if budget <= 0:
            return None
        _render_segment(session, stage, context, store, load)
        session["segments_completed"] = int(session["segments_completed"]) + 1
        budget -= 1
        if int(stage["next_segment_index"]) >= _expected_segments(session, stage):
            stage["status"] = "done"
            session["stage_index"] = stage_index + 1
        session["updated_at"] = utc_now()
        persist(session)


def execute_segmented(
    session: dict[str, Any],
    *,
    primary_bytes: bytes,
    secondary_bytes: bytes | None,
    store: SegmentStore,
    load: SegmentLoader,
    persist: SessionPersist,
) -> bytes | None:
    try:
        return run_segmented_session(
            session,
            primary_bytes=primary_bytes,
            secondary_bytes=secondary_bytes,
            store=store,
            load=load,
            persist=persist,
        )
    except MediaOperationFailure as exc:
        raise RenderFailed(
            "The rendition could not be produced.",
            details={"reason": exc.code, "detail": exc.message},
        ) from exc
    except UnsupportedMedia as exc:
        raise RenderFailed(
            "The rendition produced unsupported media bytes.",
            details={"reason": exc.code},
        ) from exc
