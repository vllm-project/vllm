# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-RobotTrack stateful single-frame sliding window.

The client sends a 32-frame window once to establish a stream, then sends one
new frame per step with a monotonic ``frame_index``. The server keeps the
rolling window (``stream_id``) so the per-request payload drops from 32 frames
to 1 frame.

Usage:
    python examples/pooling/robottrack_minicpm_stream.py \
        --model openbmb/MiniCPM-RobotTrack \
        --dino facebook/dinov3-vits16-pretrain-lvd1689m \
        --siglip google/siglip-so400m-patch14-384 \
        --images <directory of jpg frames>
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

HISTORY_FRAMES = 31  # window = history_frames + 1 = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--siglip", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--stream-id", default="cam0")
    parser.add_argument("--max-frames", type=int, default=50)
    args = parser.parse_args()

    from vllm import LLM

    llm = LLM(
        model=args.model,
        runner="pooling",
        dtype="float32",
        max_model_len=512,
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.45,
        trust_remote_code=False,
        hf_overrides={
            "dino_model": args.dino,
            "siglip_model": args.siglip,
            "image_size": 384,
            "max_cached_streams": 8,
        },
    )

    paths = sorted(Path(args.images).glob("*.jpg"))[: args.max_frames]
    window: deque[Image.Image] = deque(maxlen=HISTORY_FRAMES + 1)

    def embed(frames, frame_index):
        mm = {
            "image": {
                "frames": frames,
                "stream_id": args.stream_id,
                "frame_index": frame_index,
            }
        }
        return llm.embed([{"prompt": "Follow the person.", "multi_modal_data": mm}])[
            0
        ].outputs.embedding

    # Establish: send the full 32-frame window once (this is the handshake).
    for path in paths[: HISTORY_FRAMES + 1]:
        window.append(Image.open(path).convert("RGB"))
    sync_accelerator()
    t0 = time.perf_counter()
    traj = embed(list(window), frame_index=HISTORY_FRAMES)
    sync_accelerator()
    print(
        f"establish(32 frames): {(time.perf_counter() - t0) * 1e3:.1f}ms  "
        f"traj[:4]={np.round(traj[:4], 3)}"
    )

    # Steady: send one new frame per step.
    times = []
    for frame_index in range(HISTORY_FRAMES + 1, len(paths)):
        window.append(Image.open(paths[frame_index]).convert("RGB"))
        frames = [np.asarray(window[-1])]  # only the new frame crosses the wire
        sync_accelerator()
        t0 = time.perf_counter()
        traj = embed(frames, frame_index=frame_index)
        sync_accelerator()
        times.append((time.perf_counter() - t0) * 1e3)

    warm = times[10:]
    print(
        f"steady single-frame (n={len(warm)}): "
        f"mean={np.mean(warm):.1f}ms p50={np.median(warm):.1f}ms"
    )


def sync_accelerator():
    import torch

    torch.accelerator.synchronize()


if __name__ == "__main__":
    main()
