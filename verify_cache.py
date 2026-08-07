# SPDX-License-Identifier: Apache-2.0
"""Measure the in-tower per-frame cache: latency + trajectory parity.

Runs the rolling 32-frame window over a folder of frames and records the
per-step embed latency and predicted trajectory. Run once with the cache on
(``--frame-cache-size 64``) and once off (``0``), then compare the two .npz
outputs with ``--compare``.

Measured on RTX 4090 (CUDA graph, fp32, 50 frames, steady state = steps >= 32):
    cache off (=0): 1009.8 ms/step
    cache on (=64):  379.1 ms/step   -> 2.66x (~0.63 s/step saved)

Example:
    # cache on
    CUDA_VISIBLE_DEVICES=0 python verify_cache.py \
        --model /cache/zhanghao/model/MiniCPM-RobotTrack \
        --dino /cache/zhanghao/model/dinov3-vits16-pretrain-lvd1689m \
        --siglip /cache/zhanghao/model/siglip-so400m-patch14-384 \
        --images track-image/0 --frame-cache-size 64 --out /tmp/cache_on.npz
    # cache off
    CUDA_VISIBLE_DEVICES=0 python verify_cache.py ... \
        --frame-cache-size 0 --out /tmp/cache_off.npz
    # compare
    python verify_cache.py --compare /tmp/cache_on.npz /tmp/cache_off.npz
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WINDOW = 32


def run(args: argparse.Namespace) -> None:
    from vllm import LLM

    paths = sorted(Path(args.images).glob("*.jpg"))[: args.max_frames]

    llm = LLM(
        model=args.model,
        runner="pooling",
        dtype=args.dtype,
        max_model_len=512,
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.45,
        trust_remote_code=False,
        hf_overrides={
            "dino_model": args.dino,
            "siglip_model": args.siglip,
            "image_size": 384,
            "frame_cache_size": args.frame_cache_size,
            "pixel_cache_size": args.pixel_cache_size,
        },
    )

    window: deque[Image.Image] = deque(maxlen=WINDOW)
    times, trajs = [], []
    for path in paths:
        window.append(Image.open(path).convert("RGB"))
        mm = {"image": {"frames": [np.asarray(f) for f in window]}}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.embed([{"prompt": "Follow the person.", "multi_modal_data": mm}])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        trajs.append(np.asarray(out[0].outputs.embedding))

    times = np.array(times)
    np.savez(args.out, times=times, trajs=np.stack(trajs))
    warm = times[WINDOW:] if len(times) > WINDOW else times
    print(
        f"[frame_cache={args.frame_cache_size} pixel_cache={args.pixel_cache_size}] "
        f"steps={len(times)} total={times.sum():.2f}s "
        f"steady_mean={warm.mean() * 1e3:.1f}ms (n={len(warm)})"
    )


def compare(on_path: str, off_path: str) -> None:
    on, off = np.load(on_path), np.load(off_path)
    t_on, t_off = on["times"], off["times"]
    d = np.abs(on["trajs"] - off["trajs"])
    print("=== PARITY (cache on vs off) ===")
    print(f"  max|Δtraj|={d.max():.3e}  mean|Δtraj|={d.mean():.3e}")
    print("=== LATENCY ===")
    for name, t in [("cache OFF", t_off), ("cache ON ", t_on)]:
        warm = t[WINDOW:]
        print(
            f"  {name}: total={t.sum():5.2f}s  step0={t[0] * 1e3:5.0f}ms  "
            f"steady_mean(>=32)={warm.mean() * 1e3:5.0f}ms"
        )
    print(f"  steady-state speedup: {t_off[WINDOW:].mean() / t_on[WINDOW:].mean():.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("ON_NPZ", "OFF_NPZ"))
    ap.add_argument("--model")
    ap.add_argument("--dino")
    ap.add_argument("--siglip")
    ap.add_argument("--images")
    ap.add_argument("--frame-cache-size", type=int, default=64)
    ap.add_argument("--pixel-cache-size", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--out", default="/tmp/cache_run.npz")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
