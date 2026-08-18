#!/usr/bin/env python3
"""
Test the /v1/chat/completions SSE endpoint with an RTSP source.

The DeepStream-backed vLLM server detects rtsp:// URLs in the
``messages`` content and emits one chat.completion.chunk per decoded
segment caption. The wire format mirrors RTVI's bend on chat/completions:

  - One id, one created across all chunks in a single stream.
  - Each data chunk: ``delta.content = "<full caption for one segment>"``,
    ``finish_reason: null``.
  - Terminal chunk: ``delta: {}``, ``finish_reason: "stop"``.
  - Then literal ``[DONE]``.

Usage:
    python3 test_rtsp_stream.py [rtsp_url] [options]

Examples:
    python3 test_rtsp_stream.py rtsp://10.24.217.130:8554/
    python3 test_rtsp_stream.py rtsp://10.24.217.130:8554/ --segments 3
    python3 test_rtsp_stream.py file:///data/video/drivesim.mp4
"""

import argparse
import json
import os
import sys
import time

import requests


def stream_captions(
    rtsp_url: str,
    server: str = "http://localhost:8000",
    model: str = "bench-model",
    prompt: str = "Describe what is happening in this video segment.",
    chunk_duration: float = 10.0,
    num_frames: int = 8,
    max_tokens: int = 256,
    temperature: float = 0.0,
    max_segments: int | None = None,
) -> None:
    url = f"{server}/v1/chat/completions"
    payload = {
        "model": model,
        "stream": True,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": rtsp_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Custom extras tolerated by OpenAIBaseModel(extra="allow"):
        "chunk_duration": chunk_duration,
        "num_frames_per_chunk": num_frames,
    }

    print(f"Connecting to: {url}")
    print(f"RTSP source  : {rtsp_url}")
    print(f"Chunk        : {chunk_duration}s per segment, {num_frames} frames")
    print("-" * 60)

    t_connect = time.monotonic()
    segments_received = 0

    try:
        with requests.post(url, json=payload, stream=True, timeout=None) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue

                line = (
                    raw_line.decode("utf-8")
                    if isinstance(raw_line, bytes)
                    else raw_line
                )
                if not line.startswith("data: "):
                    continue

                data = line[len("data: "):]
                if data == "[DONE]":
                    print("\n[DONE] — stream ended")
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    print(f"[WARN] Could not parse: {data!r}")
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice0 = choices[0]
                delta = choice0.get("delta") or {}
                finish_reason = choice0.get("finish_reason")

                content = delta.get("content")
                if content:
                    segments_received += 1
                    elapsed = time.monotonic() - t_connect
                    print(
                        f"\n[Segment {segments_received - 1}]  "
                        f"id={chunk.get('id', '?')}  "
                        f"(wall +{elapsed:.1f}s)"
                    )
                    print(f"  {content}")

                    if max_segments is not None and segments_received >= max_segments:
                        print(
                            f"\nReached --segments {max_segments} limit, "
                            "disconnecting."
                        )
                        break

                if finish_reason == "stop":
                    print("\n[finish_reason=stop] — terminal chunk")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except requests.exceptions.ConnectionError as e:
        print(f"\n[ERROR] Could not connect to {server}: {e}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n[ERROR] HTTP {e.response.status_code}: {e.response.text}")
        sys.exit(1)

    total = time.monotonic() - t_connect
    print(f"\n{segments_received} segment(s) received in {total:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Test /v1/chat/completions SSE endpoint with an RTSP source"
    )
    parser.add_argument(
        "rtsp_url",
        nargs="?",
        default="rtsp://10.24.217.130:8554/",
        help="RTSP or file:// URI (default: rtsp://10.24.217.130:8554/)",
    )
    # Default server URL honors $PORT (and $HOST) env vars so the script
    # works with `PORT=8001 python3 test_rtsp_stream.py ...` without
    # needing --server. Explicit --server still overrides.
    default_server = (
        f"http://{os.environ.get('HOST', 'localhost')}"
        f":{os.environ.get('PORT', '8000')}"
    )
    parser.add_argument("--server", default=default_server,
                        help=f"Server base URL (default: {default_server}, "
                             f"override via $HOST/$PORT or --server)")
    parser.add_argument("--model",    default="bench-model")
    parser.add_argument(
        "--prompt",
        default="Describe what is happening in this video segment.",
    )
    parser.add_argument(
        "--chunk-duration", type=float, default=10.0,
        help="Seconds per chunk/segment",
    )
    parser.add_argument(
        "--frames", type=int, default=8, help="Frames per segment",
    )
    parser.add_argument(
        "--tokens", type=int, default=256, help="Max output tokens",
    )
    parser.add_argument(
        "--segments", type=int, default=None, help="Stop after N segments",
    )
    args = parser.parse_args()

    stream_captions(
        rtsp_url=args.rtsp_url,
        server=args.server,
        model=args.model,
        prompt=args.prompt,
        chunk_duration=args.chunk_duration,
        num_frames=args.frames,
        max_tokens=args.tokens,
        max_segments=args.segments,
    )


if __name__ == "__main__":
    main()
