#!/usr/bin/env python3
"""Run bounded production workflow checks against an ephemeral HTTP server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import io
import json
import math
import struct
import sys
import tempfile
import threading
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avprovenance.http_api import ProvenanceHTTPServer, build_http_server


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_wav(
    *,
    frequency: float,
    duration_seconds: float = 0.6,
    sample_rate: int = 8000,
    amplitude: int = 9000,
) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    values = [
        int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
        for index in range(frame_count)
    ]
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(values)}h", *values))
    return payload.getvalue()


def make_y4m(
    *,
    width: int = 64,
    height: int = 36,
    frame_count: int = 5,
    rate_numerator: int = 12,
    rate_denominator: int = 1,
) -> bytes:
    header = (
        "YUV4MPEG2 "
        f"W{width} H{height} "
        f"F{rate_numerator}:{rate_denominator} "
        "Ip C420jpeg\n"
    ).encode("ascii")
    chroma_width = width // 2
    chroma_height = height // 2
    parts = [header]
    for frame_index in range(frame_count):
        parts.append(b"FRAME\n")
        y_plane = bytes(
            (x * 3 + y * 5 + frame_index * 11) % 256
            for y in range(height)
            for x in range(width)
        )
        u_plane = bytes(
            (96 + x * 2 + frame_index * 3) % 256
            for _y in range(chroma_height)
            for x in range(chroma_width)
        )
        v_plane = bytes(
            (144 + y * 2 + frame_index * 5) % 256
            for y in range(chroma_height)
            for _x in range(chroma_width)
        )
        parts.extend((y_plane, u_plane, v_plane))
    return b"".join(parts)


class HTTPClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        connection = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=15,
        )
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            value = json.loads(raw.decode("utf-8"))
            return response.status, value
        finally:
            connection.close()

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        status, body = self.request(
            "POST",
            "/v1/assets",
            {
                "filename": filename,
                "data_base64": base64.b64encode(data).decode("ascii"),
            },
        )
        require_equal(status, 201, "asset upload status")
        return body

    def create_rendition(self, payload: dict[str, Any]) -> dict[str, Any]:
        status, body = self.request("POST", "/v1/renditions", payload)
        require_equal(status, 201, "rendition creation status")
        return body


@contextmanager
def server_on(data_dir: Path) -> Iterator[tuple[HTTPClient, ProvenanceHTTPServer]]:
    server = build_http_server("127.0.0.1", 0, data_dir)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.02},
        daemon=True,
    )
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield HTTPClient(str(host), int(port)), server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@contextmanager
def running_server() -> Iterator[tuple[HTTPClient, ProvenanceHTTPServer]]:
    with tempfile.TemporaryDirectory(prefix="avprovenance-check-") as temporary:
        with server_on(Path(temporary)) as pair:
            yield pair


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_ingest_media_asset(client: HTTPClient) -> dict[str, Any]:
    health_status, health = client.request("GET", "/health")
    require_equal(health_status, 200, "health status")
    require_equal(health["status"], "ok", "health payload")

    audio = make_wav(frequency=330)
    asset = client.upload("tone-source.wav", audio)
    require_equal(asset["kind"], "audio", "asset kind")
    require_equal(asset["sha256"], _sha256(audio), "asset digest")
    require_equal(asset["descriptor"]["container"], "wav", "asset container")
    require("content_key" not in asset, "public asset exposed an internal content key")

    get_status, fetched = client.request("GET", f"/v1/assets/{asset['id']}")
    require_equal(get_status, 200, "asset get status")
    require_equal(fetched["sha256"], asset["sha256"], "asset read digest")

    bad_status, bad_body = client.request(
        "POST",
        "/v1/assets",
        {
            "filename": "unknown.bin",
            "data_base64": base64.b64encode(b"not-media").decode("ascii"),
        },
    )
    require_equal(bad_status, 415, "unsupported media status")
    require_equal(
        bad_body["error"]["code"],
        "UNSUPPORTED_MEDIA",
        "unsupported media code",
    )
    return {
        "workflow": "ingest-media-asset",
        "asset_id": asset["id"],
        "kind": asset["kind"],
        "digest": asset["sha256"],
    }


def run_execute_rendition(client: HTTPClient) -> dict[str, Any]:
    primary_bytes = make_wav(frequency=440, amplitude=10000)
    secondary_bytes = make_wav(frequency=220, amplitude=7000)
    primary = client.upload("primary-tone.wav", primary_bytes)
    secondary = client.upload("secondary-tone.wav", secondary_bytes)

    audio_rendition = client.create_rendition(
        {
            "primary_asset_id": primary["id"],
            "secondary_asset_id": secondary["id"],
            "output_name": "mixed-tone.wav",
            "operations": [
                {
                    "type": "media_trim",
                    "start_seconds": 0.05,
                    "duration_seconds": 0.3,
                },
                {"type": "audio_gain", "db": -3},
                {
                    "type": "audio_mix",
                    "secondary_gain": -6,
                    "primary_weight": 0.75,
                    "secondary_weight": 0.4,
                },
                {"type": "audio_normalize", "target_peak": 0.8},
            ],
        }
    )
    require_equal(audio_rendition["status"], "completed", "audio rendition status")
    history = [item["to"] for item in audio_rendition["transitions"]]
    require_equal(
        history,
        ["queued", "running", "completed"],
        "audio state history",
    )
    require(
        audio_rendition["output_asset"]["descriptor"]["container"] == "wav",
        "audio output descriptor was not WAV",
    )

    video_bytes = make_y4m()
    video = client.upload("source-frames.y4m", video_bytes)
    video_rendition = client.create_rendition(
        {
            "primary_asset_id": video["id"],
            "output_name": "cropped-frames.y4m",
            "operations": [
                {
                    "type": "video_crop",
                    "x": 8,
                    "y": 6,
                    "width": 32,
                    "height": 18,
                },
                {"type": "video_luma_offset", "amount": 12},
                {"type": "video_mirror"},
            ],
        }
    )
    require_equal(video_rendition["status"], "completed", "video rendition status")
    properties = video_rendition["output_asset"]["descriptor"]["properties"]
    require_equal(properties["width"], 32, "video output width")
    require_equal(properties["height"], 18, "video output height")

    lineage_status, lineage = client.request(
        "GET",
        f"/v1/renditions/{audio_rendition['id']}/lineage",
    )
    require_equal(lineage_status, 200, "lineage status")
    require_equal(len(lineage["parents"]), 2, "lineage parent count")
    require_equal(
        lineage["edge"]["plan_digest"],
        audio_rendition["plan_digest"],
        "lineage plan digest",
    )
    invalid_status, invalid_body = client.request(
        "POST",
        "/v1/renditions",
        {
            "primary_asset_id": primary["id"],
            "output_name": "invalid.wav",
            "operations": [
                {
                    "type": "video_crop",
                    "x": 0,
                    "y": 0,
                    "width": 2,
                    "height": 2,
                }
            ],
        },
    )
    require_equal(invalid_status, 422, "invalid plan status")
    require_equal(invalid_body["error"]["code"], "PLAN_INVALID", "invalid plan code")
    return {
        "workflow": "execute-rendition",
        "audio_rendition_id": audio_rendition["id"],
        "video_rendition_id": video_rendition["id"],
        "audio_history": history,
        "invalid_plan_code": invalid_body["error"]["code"],
        "video_dimensions": [
            properties["width"],
            properties["height"],
        ],
    }


def run_verify_reproduction(
    client: HTTPClient,
    server: ProvenanceHTTPServer,
) -> dict[str, Any]:
    video_bytes = make_y4m(frame_count=7)
    video = client.upload("verification-source.y4m", video_bytes)
    rendition = client.create_rendition(
        {
            "primary_asset_id": video["id"],
            "output_name": "verification-output.y4m",
            "operations": [
                {"type": "media_trim", "start_frame": 1, "frame_count": 4},
                {
                    "type": "video_crop",
                    "x": 4,
                    "y": 4,
                    "width": 40,
                    "height": 20,
                },
                {"type": "video_luma_offset", "amount": -9},
                {"type": "video_mirror"},
            ],
        }
    )
    check_status, check = client.request(
        "POST",
        f"/v1/renditions/{rendition['id']}/integrity-checks",
        {},
    )
    require_equal(check_status, 201, "integrity check status")
    require_equal(check["status"], "verified", "integrity result")
    require_equal(
        check["expected_digest"],
        check["reproduced_digest"],
        "integrity digest equality",
    )
    require_equal(
        check["expected_descriptor"],
        check["reproduced_descriptor"],
        "integrity descriptor equality",
    )

    lineage_status, lineage = client.request(
        "GET",
        f"/v1/renditions/{rendition['id']}/lineage",
    )
    require_equal(lineage_status, 200, "verification lineage status")
    require_equal(len(lineage["parents"]), 1, "verification parent count")

    output_asset_id = rendition["output_asset"]["id"]

    def drift_metadata(state: dict[str, Any]) -> None:
        state["assets"][output_asset_id]["sha256"] = "0" * 64

    server.application.state_file.mutate(drift_metadata)
    mismatch_status, mismatch = client.request(
        "POST",
        f"/v1/renditions/{rendition['id']}/integrity-checks",
        {},
    )
    require_equal(mismatch_status, 409, "mismatch status")
    require_equal(
        mismatch["error"]["code"],
        "INTEGRITY_MISMATCH",
        "mismatch error code",
    )
    require_equal(
        mismatch["error"]["details"]["audit"]["status"],
        "mismatch",
        "persisted mismatch status",
    )
    return {
        "workflow": "verify-reproduction",
        "rendition_id": rendition["id"],
        "integrity_check_id": check["id"],
        "digest": check["reproduced_digest"],
        "mismatch_status": mismatch["error"]["details"]["audit"]["status"],
    }


def resume_until_done(
    client: HTTPClient,
    rendition_id: str,
    *,
    limit: int = 500,
) -> tuple[dict[str, Any], int]:
    resumes = 0
    while True:
        status, body = client.request(
            "POST",
            f"/v1/renditions/{rendition_id}/resume",
            {},
        )
        require_equal(status, 200, "resume status")
        resumes += 1
        require(resumes <= limit, "segmented rendition did not finish")
        if body["status"] != "paused":
            return body, resumes


def run_segmented_rendition_resume() -> dict[str, Any]:
    audio_plan = [
        {"type": "media_trim", "start_seconds": 0.05, "duration_seconds": 1.0},
        {"type": "audio_gain", "db": -3},
        {
            "type": "audio_mix",
            "secondary_gain": -6,
            "primary_weight": 0.75,
            "secondary_weight": 0.4,
        },
        {"type": "audio_normalize", "target_peak": 0.8},
    ]
    video_plan = [
        {"type": "media_trim", "start_frame": 1, "frame_count": 7},
        {"type": "video_crop", "x": 8, "y": 6, "width": 32, "height": 18},
        {"type": "video_luma_offset", "amount": 12},
        {"type": "video_mirror"},
    ]
    with tempfile.TemporaryDirectory(prefix="avprovenance-check-") as temporary:
        data_dir = Path(temporary)
        with server_on(data_dir) as (client, server):
            primary = client.upload(
                "long-primary.wav",
                make_wav(frequency=440, duration_seconds=1.1, amplitude=10000),
            )
            secondary = client.upload(
                "long-secondary.wav",
                make_wav(frequency=220, duration_seconds=1.4, amplitude=7000),
            )

            create_status, segmented = client.request(
                "POST",
                "/v1/renditions",
                {
                    "primary_asset_id": primary["id"],
                    "secondary_asset_id": secondary["id"],
                    "output_name": "segmented-mix.wav",
                    "operations": audio_plan,
                    "execution": {
                        "mode": "segmented",
                        "segment_frames": 2000,
                        "segments_per_run": 2,
                    },
                },
            )
            require_equal(create_status, 201, "segmented create status")
            require_equal(segmented["status"], "paused", "segmented create pauses")
            require(segmented["output_asset"] is None, "paused rendition exposed an output")
            progress = segmented["render_progress"]
            require_equal(progress["mode"], "segmented", "progress mode")
            require(progress["segments_completed"] > 0, "no segments were checkpointed")
            segmented_id = segmented["id"]

            _, paused_lineage = client.request(
                "GET",
                f"/v1/renditions/{segmented_id}/lineage",
            )
            require_equal(paused_lineage["status"], "paused", "paused lineage status")
            require(paused_lineage["output"] is None, "paused lineage exposed an output")
            require(paused_lineage["edge"] is None, "paused lineage exposed an edge")
            require_equal(paused_lineage["parents"], [], "paused lineage parents")

            early_status, early_check = client.request(
                "POST",
                f"/v1/renditions/{segmented_id}/integrity-checks",
                {},
            )
            require_equal(early_status, 409, "paused integrity status")
            require_equal(
                early_check["error"]["code"],
                "RENDITION_NOT_COMPLETED",
                "paused integrity code",
            )

            inline = client.create_rendition(
                {
                    "primary_asset_id": primary["id"],
                    "secondary_asset_id": secondary["id"],
                    "output_name": "inline-mix.wav",
                    "operations": audio_plan,
                }
            )
            require_equal(inline["status"], "completed", "inline rendition status")
            require(inline["render_progress"] is None, "inline rendition had a session")
            require_equal(
                inline["plan_digest"],
                segmented["plan_digest"],
                "execution mode changed the plan digest",
            )
            inline_digest = inline["output_asset"]["sha256"]

            segmented, audio_resumes = resume_until_done(client, segmented_id)
            require_equal(segmented["status"], "completed", "resumed audio status")
            require(
                segmented["output_asset"]["sha256"] == inline_digest,
                "resumed audio bytes differ from a single-run render",
            )
            history = [item["to"] for item in segmented["transitions"]]
            require("paused" in history, "segmented history missed the pause")
            require_equal(history[-1], "completed", "segmented history terminal state")

            _, lineage = client.request(
                "GET",
                f"/v1/renditions/{segmented_id}/lineage",
            )
            require_equal(lineage["status"], "completed", "completed lineage status")
            require_equal(len(lineage["parents"]), 2, "completed lineage parents")
            require_equal(
                lineage["output"]["sha256"],
                inline_digest,
                "completed lineage output digest",
            )

            check_status, check = client.request(
                "POST",
                f"/v1/renditions/{segmented_id}/integrity-checks",
                {},
            )
            require_equal(check_status, 201, "segmented integrity status")
            require_equal(check["status"], "verified", "segmented integrity result")
            require_equal(
                check["reproduced_digest"],
                inline_digest,
                "integrity replay digest",
            )

            again_status, again = client.request(
                "POST",
                f"/v1/renditions/{segmented_id}/resume",
                {},
            )
            require_equal(again_status, 409, "completed resume status")
            require_equal(
                again["error"]["code"],
                "RENDITION_NOT_PAUSED",
                "completed resume code",
            )

            video = client.upload("long-frames.y4m", make_y4m(frame_count=9))
            _, video_segmented = client.request(
                "POST",
                "/v1/renditions",
                {
                    "primary_asset_id": video["id"],
                    "output_name": "segmented-video.y4m",
                    "operations": video_plan,
                    "execution": {
                        "mode": "segmented",
                        "segment_frames": 2,
                        "segments_per_run": 1,
                    },
                },
            )
            require_equal(
                video_segmented["status"],
                "paused",
                "segmented video create pauses",
            )
            video_inline = client.create_rendition(
                {
                    "primary_asset_id": video["id"],
                    "output_name": "inline-video.y4m",
                    "operations": video_plan,
                }
            )
            video_segmented, video_resumes = resume_until_done(
                client,
                video_segmented["id"],
            )
            require_equal(
                video_segmented["status"],
                "completed",
                "resumed video status",
            )
            require_equal(
                video_segmented["output_asset"]["sha256"],
                video_inline["output_asset"]["sha256"],
                "resumed video bytes differ from a single-run render",
            )
            vcheck_status, vcheck = client.request(
                "POST",
                f"/v1/renditions/{video_segmented['id']}/integrity-checks",
                {},
            )
            require_equal(vcheck_status, 201, "video integrity status")
            require_equal(vcheck["status"], "verified", "video integrity result")

            crash_status, crash = client.request(
                "POST",
                "/v1/renditions",
                {
                    "primary_asset_id": primary["id"],
                    "secondary_asset_id": secondary["id"],
                    "output_name": "interrupted-mix.wav",
                    "operations": audio_plan,
                    "execution": {
                        "mode": "segmented",
                        "segment_frames": 2000,
                        "segments_per_run": 1,
                    },
                },
            )
            require_equal(crash_status, 201, "interrupted create status")
            require_equal(crash["status"], "paused", "interrupted create pauses")
            crash_id = crash["id"]

            def simulate_process_kill(state: dict[str, Any]) -> None:
                state["renditions"][crash_id]["status"] = "running"

            server.application.state_file.mutate(simulate_process_kill)

        with server_on(data_dir) as (client, server):
            get_status, fetched = client.request(
                "GET",
                f"/v1/renditions/{crash_id}",
            )
            require_equal(get_status, 200, "restarted rendition status")
            require_equal(
                fetched["status"],
                "paused",
                "interrupted rendition was not reconciled to paused",
            )
            require(
                fetched["render_progress"] is not None,
                "restarted rendition lost its render session",
            )

            crashed, crash_resumes = resume_until_done(client, crash_id)
            require_equal(crashed["status"], "completed", "crash-resumed status")
            require_equal(
                crashed["output_asset"]["sha256"],
                inline_digest,
                "crash-resumed bytes differ from a single-run render",
            )
            crash_check_status, crash_check = client.request(
                "POST",
                f"/v1/renditions/{crash_id}/integrity-checks",
                {},
            )
            require_equal(crash_check_status, 201, "crash integrity status")
            require_equal(
                crash_check["status"],
                "verified",
                "crash integrity result",
            )

            state = server.application.state_file.read()
            require_equal(
                state["render_sessions"],
                {},
                "render sessions were not cleaned up",
            )
            asset_digests = {
                str(asset["content_key"]).split("/")[-1]
                for asset in state["assets"].values()
            }
            blob_digests = {
                path.name
                for path in (data_dir / "content").rglob("*")
                if path.is_file()
            }
            require_equal(
                blob_digests,
                asset_digests,
                "segment checkpoints leaked into content storage",
            )
            return {
                "workflow": "segmented-rendition-resume",
                "audio_rendition_id": segmented_id,
                "video_rendition_id": video_segmented["id"],
                "crash_rendition_id": crash_id,
                "audio_resumes": audio_resumes,
                "video_resumes": video_resumes,
                "crash_resumes": crash_resumes,
                "digest": inline_digest,
                "integrity": crash_check["status"],
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "workflow",
        choices=(
            "ingest-media-asset",
            "execute-rendition",
            "verify-reproduction",
            "segmented-rendition-resume",
        ),
    )
    args = parser.parse_args()

    if args.workflow == "segmented-rendition-resume":
        result = run_segmented_rendition_resume()
    else:
        with running_server() as (client, server):
            if args.workflow == "ingest-media-asset":
                result = run_ingest_media_asset(client)
            elif args.workflow == "execute-rendition":
                result = run_execute_rendition(client)
            else:
                result = run_verify_reproduction(client, server)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
