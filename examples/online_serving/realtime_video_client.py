# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example client for streaming video frames to vLLM's realtime API.

Usage:
    # Stream from webcam (requires opencv-python):
    python realtime_video_client.py --source webcam --model Qwen/Qwen3-Omni-30B-A3B-Instruct

    # Stream from a video file:
    python realtime_video_client.py --source video.mp4 --model Qwen/Qwen3-Omni-30B-A3B-Instruct

    # Stream from a directory of images:
    python realtime_video_client.py --source ./frames/ --model Qwen/Qwen3-Omni-30B-A3B-Instruct

Requires:
    pip install websockets pillow opencv-python
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path

import websockets


def encode_frame_jpeg(frame_rgb, quality: int = 85) -> str:
    """Encode a numpy RGB frame to base64 JPEG."""
    from PIL import Image

    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def stream_webcam(ws, fps: float, query: str, duration: float):
    """Stream frames from webcam."""
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    interval = 1.0 / fps
    start = time.monotonic()
    frame_count = 0

    try:
        while time.monotonic() - start < duration:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            b64 = encode_frame_jpeg(frame_rgb)

            await ws.send(json.dumps({
                "type": "input_video_frame.append",
                "image": b64,
                "timestamp": time.monotonic() - start,
            }))
            frame_count += 1

            if frame_count % int(fps) == 0:
                print(f"  Sent {frame_count} frames "
                      f"({time.monotonic() - start:.1f}s)")

            await asyncio.sleep(interval)
    finally:
        cap.release()

    # Commit with query
    await ws.send(json.dumps({
        "type": "input_video_frame.commit",
        "query": query,
        "final": True,
    }))
    print(f"Committed {frame_count} frames with query: {query!r}")


async def stream_video_file(ws, path: str, fps: float, query: str):
    """Stream frames from a video file."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = max(1, int(src_fps / fps))
    frame_idx = 0
    sent = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            b64 = encode_frame_jpeg(frame_rgb)

            await ws.send(json.dumps({
                "type": "input_video_frame.append",
                "image": b64,
                "timestamp": frame_idx / src_fps,
            }))
            sent += 1

            if sent % 10 == 0:
                print(f"  Sent {sent} frames")

        frame_idx += 1

    cap.release()

    await ws.send(json.dumps({
        "type": "input_video_frame.commit",
        "query": query,
        "final": True,
    }))
    print(f"Committed {sent} frames from {path}")


async def stream_image_dir(ws, dir_path: str, fps: float, query: str):
    """Stream frames from a directory of images."""
    from PIL import Image

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(
        p for p in Path(dir_path).iterdir()
        if p.suffix.lower() in exts
    )

    if not files:
        print(f"ERROR: No image files found in {dir_path}")
        return

    interval = 1.0 / fps
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        await ws.send(json.dumps({
            "type": "input_video_frame.append",
            "image": b64,
            "timestamp": i / fps,
        }))

        if (i + 1) % 10 == 0:
            print(f"  Sent {i + 1}/{len(files)} frames")

        await asyncio.sleep(interval)

    await ws.send(json.dumps({
        "type": "input_video_frame.commit",
        "query": query,
        "final": True,
    }))
    print(f"Committed {len(files)} frames from {dir_path}")


async def receive_responses(ws):
    """Listen for server responses and print them."""
    try:
        async for message in ws:
            event = json.loads(message)
            event_type = event.get("type")

            if event_type == "session.created":
                print(f"Session created: {event.get('id')}")
            elif event_type == "video_chat.delta":
                print(event["delta"], end="", flush=True)
            elif event_type == "video_chat.done":
                print()  # newline after deltas
                usage = event.get("usage")
                if usage:
                    print(f"\n[Usage] prompt={usage['prompt_tokens']}, "
                          f"completion={usage['completion_tokens']}, "
                          f"total={usage['total_tokens']}")
            elif event_type == "error":
                print(f"\nERROR: {event['error']} (code={event.get('code')})",
                      file=sys.stderr)
            else:
                print(f"[{event_type}] {event}")
    except websockets.exceptions.ConnectionClosed:
        pass


async def main(args):
    uri = f"ws://{args.host}:{args.port}/v1/realtime"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        # Start response listener
        recv_task = asyncio.create_task(receive_responses(ws))

        # Wait for session.created, then send session.update
        await asyncio.sleep(0.1)
        await ws.send(json.dumps({
            "type": "session.update",
            "model": args.model,
        }))
        await asyncio.sleep(0.1)

        # Stream frames based on source type
        source = args.source
        if source == "webcam":
            await stream_webcam(ws, args.fps, args.query, args.duration)
        elif Path(source).is_dir():
            await stream_image_dir(ws, source, args.fps, args.query)
        elif Path(source).is_file():
            await stream_video_file(ws, source, args.fps, args.query)
        else:
            print(f"ERROR: Unknown source: {source}", file=sys.stderr)
            return

        # Wait for all responses
        await recv_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream video frames to vLLM realtime API"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--source", default="webcam",
        help="'webcam', path to video file, or directory of images"
    )
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Target frame rate (default: 1)")
    parser.add_argument("--query", default="Describe what you see.",
                        help="Text query to send with the video frames")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration in seconds for webcam capture")
    args = parser.parse_args()

    asyncio.run(main(args))
