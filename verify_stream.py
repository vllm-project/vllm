# SPDX-License-Identifier: Apache-2.0
"""Verify the stateful single-frame stream protocol end-to-end.

Runs the same 50 frames twice on one GPU:
  - stateful: establish a stream with 32 frames, then one new frame per step;
  - stateless: re-send the full 32-frame window every step (the previous path).
Compares the trajectories of matching windows and reports per-step latency for
the stateful (1-frame/request) path.
"""
import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HISTORY = 31


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--siglip", required=True)
    parser.add_argument("--images", required=True)
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
            "frame_cache_size": 64,
            "pixel_cache_size": 64,
            "max_cached_streams": 8,
        },
    )
    paths = sorted(Path(args.images).glob("*.jpg"))[: args.max_frames]
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]

    # ---- stateful path ----
    window = deque(maxlen=HISTORY + 1)
    st_traj, st_times = [], []
    for i, f in enumerate(frames):
        window.append(f)
        if i < HISTORY:
            continue
        if i == HISTORY:
            mm = {"image": {"frames": list(window), "stream_id": "s0",
                            "frame_index": HISTORY}}
        else:
            mm = {"image": {"frames": [f], "stream_id": "s0",
                            "frame_index": i}}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.embed([{"prompt": "Follow the person.", "multi_modal_data": mm}])
        torch.cuda.synchronize()
        st_times.append((time.perf_counter() - t0) * 1e3)
        st_traj.append(np.asarray(out[0].outputs.embedding))

    # ---- stateless reference: full 32-frame window every step ----
    window = deque(maxlen=HISTORY + 1)
    sl_traj = []
    for i, f in enumerate(frames):
        window.append(f)
        if i < HISTORY:
            continue
        mm = {"image": {"frames": list(window)}}
        out = llm.embed([{"prompt": "Follow the person.", "multi_modal_data": mm}])
        sl_traj.append(np.asarray(out[0].outputs.embedding))

    st_traj = np.stack(st_traj)
    sl_traj = np.stack(sl_traj)
    assert st_traj.shape == sl_traj.shape

    # ---- parity: same window (frame N-31..N) must give the same waypoints ----
    diff = np.abs(st_traj - sl_traj)
    print(f"stateful steps={len(st_traj)}  trajectory scale=[{sl_traj.min():.3f}, "
          f"{sl_traj.max():.3f}]")
    print(f"PARITY max|stateful - stateless| = {diff.max():.3e}  "
          f"mean = {diff.mean():.3e}")
    print("per-step max|d|:", " ".join(f"{d:.1e}" for d in diff.max(axis=1)))
    if diff.max() > 5e-2:
        raise RuntimeError("stateful/stateless trajectories diverged")

    warm = np.array(st_times)[max(0, len(st_times) - 10):]
    print(f"stateful steady 1-frame/request: mean={warm.mean():.1f}ms "
          f"p50={np.median(warm):.1f}ms (n={len(warm)})")


if __name__ == "__main__":
    main()
