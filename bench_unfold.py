#!/usr/bin/env python3
"""
bench_patchify_no_resize.py

Benchmark ONLY patch extraction (no interpolate) for 512x512 non-overlapping patches.

Input resolutions:
    Cartesian product of {512, 1024, 2048, 4096} x {512, 1024, 2048, 4096}

Patch extraction:
    patch = 512, stride = 512  (non-overlapping)

Variants benchmarked:
  1) f_unfold                 : torch.nn.functional.unfold (im2col; materializes)
  2) reshape_perm             : reshape + permute patchify (fast; mostly views)
  3) tensor_unfold_view       : tensor.unfold (view/as_strided) + permute
  4) pixel_unshuffle_patchify : pixel_unshuffle + view/permute (also view-ish; requires divisible)

Correctness:
  - All variants MUST match f_unfold EXACTLY (torch.equal), including patch ordering.

Shape print per case:
    "3, H, W  -->  N, 3, 512, 512"
where N = B * (H/512) * (W/512)

Usage:
  python bench_patchify_no_resize.py --device cuda --iters 200 --warmup 50 --batch 1 --dtype float16
  python bench_patchify_no_resize.py --device cpu  --iters 50  --warmup 10 --batch 1 --dtype float32

Notes:
  - Uses random FLOAT input (not uint8) to avoid dtype limitations in some ops.
  - Requires H and W divisible by 512 (true for the provided RES_SET).
"""

import argparse
import itertools
import math
import statistics
import time
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F


RES_SET = [512, 1024, 2048, 4096]
P = 512  # patch size and stride


# -------------------------
# Timing utils
# -------------------------

def _sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _assert_same(a: torch.Tensor, b: torch.Tensor, name_a: str, name_b: str):
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch {name_a} {tuple(a.shape)} vs {name_b} {tuple(b.shape)}")
    if not torch.equal(a, b):
        diff = (a - b).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        idx = int(diff.argmax().item()) if diff.numel() else 0
        raise AssertionError(
            f"Value mismatch {name_a} vs {name_b}: max_abs_diff={max_abs:.6g} (flat argmax idx={idx})"
        )


# -------------------------
# Patchify variants (no resize)
# -------------------------

@torch.no_grad()
def f_unfold(x: torch.Tensor) -> torch.Tensor:
    # x: (B,C,H,W) -> (B, C*P*P, N) -> (B*N, C, P, P)
    cols = F.unfold(x, kernel_size=P, stride=P)
    return cols.transpose(1, 2).reshape(-1, x.shape[1], P, P)


@torch.no_grad()
def reshape_perm(x: torch.Tensor) -> torch.Tensor:
    # (B,C,H,W) -> (B, H/P, W/P, C, P, P) -> (B*N, C, P, P)
    B, C, H, W = x.shape
    hp, wp = H // P, W // P
    return (
        x.reshape(B, C, hp, P, wp, P)
         .permute(0, 2, 4, 1, 3, 5)
         .reshape(B * hp * wp, C, P, P)
    )


@torch.no_grad()
def tensor_unfold_view(x: torch.Tensor) -> torch.Tensor:
    # unfold returns a strided view in many cases
    u = x.unfold(2, P, P).unfold(3, P, P)  # (B, C, hp, wp, P, P)
    return u.permute(0, 2, 3, 1, 4, 5).reshape(-1, x.shape[1], P, P)


@torch.no_grad()
def pixel_unshuffle_patchify(x: torch.Tensor) -> torch.Tensor:
    # pixel_unshuffle: (B,C,H,W) -> (B, C*P*P, H/P, W/P)
    # then view channels back into (C,P,P) with the correct order
    B, C, H, W = x.shape
    hp, wp = H // P, W // P
    y = F.pixel_unshuffle(x, downscale_factor=P)  # (B, C*P*P, hp, wp)
    # reshape channel dim back to (C, P, P) preserving row-major within patch
    y = y.view(B, C, P, P, hp, wp)               # (B, C, P, P, hp, wp)
    y = y.permute(0, 4, 5, 1, 2, 3)              # (B, hp, wp, C, P, P)
    return y.reshape(B * hp * wp, C, P, P)


VARIANTS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "f_unfold": f_unfold,
    "reshape_perm": reshape_perm,
    "tensor_unfold_view": tensor_unfold_view,
    "pixel_unshuffle_patchify": pixel_unshuffle_patchify,
}


# -------------------------
# Benchmark harness
# -------------------------

def bench_one(fn: Callable[[torch.Tensor], torch.Tensor],
              x: torch.Tensor,
              device: str,
              warmup: int,
              iters: int) -> Tuple[float, float, float]:
    for _ in range(warmup):
        out = fn(x)
        _ = out.numel()
    _sync(device)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(x)
        _ = out.numel()
        _sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)  # ms

    mean = statistics.mean(times)
    p50 = _percentile(times, 0.50)
    p90 = _percentile(times, 0.90)
    return mean, p50, p90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", help="cuda/cuda:0 or cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--channels_last", action="store_true", help="use channels_last memory format")
    ap.add_argument("--no-check", action="store_true", help="disable correctness checks")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    torch.manual_seed(0)

    print("\n" + "=" * 88)
    print("Patchify benchmark (NO resize)")
    print(f"Patch size: {P}x{P} | stride={P}")
    print(f"Input resolutions: {RES_SET} x {RES_SET} (cartesian)")
    print(f"Device: {device} | dtype: {args.dtype} | batch: {args.batch} | iters: {args.iters} | warmup: {args.warmup}")
    if args.channels_last:
        print("Input memory format: channels_last")
    print(f"Correctness checks: {'OFF' if args.no_check else 'ON'}")
    print("=" * 88)

    pairs = list(itertools.product(RES_SET, RES_SET))

    for H, W in pairs:
        if H % P != 0 or W % P != 0:
            print(f"\nSKIP {H}x{W} (not divisible by patch {P})")
            continue

        # Create input
        x = torch.rand((args.batch, 3, H, W), device=device, dtype=dtype)
        if args.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        # Reference output (also used for shape print)
        ref = f_unfold(x)
        hp, wp = H // P, W // P
        expected_patches = args.batch * hp * wp

        # Shape print
        print("\n" + "-" * 88)
        print(f"3, {H}, {W}  -->  {expected_patches}, 3, {P}, {P}")

        # Correctness checks
        if not args.no_check:
            for name, fn in VARIANTS.items():
                out = fn(x)
                _assert_same(ref, out, "f_unfold", name)
            print("Correctness: ✔ all variants identical to f_unfold")
        else:
            print("Correctness: skipped")

        # Benchmark
        results = []
        for name, fn in VARIANTS.items():
            mean, p50, p90 = bench_one(fn, x, device, args.warmup, args.iters)
            results.append((name, mean, p50, p90))
        results.sort(key=lambda t: t[1])

        # Pretty table
        print("-" * 88)
        print(f"{'Rank':<5} {'Variant':<28} {'Mean (ms)':>12} {'P50 (ms)':>12} {'P90 (ms)':>12}")
        print("-" * 88)
        for i, (name, mean, p50, p90) in enumerate(results, 1):
            print(f"{i:<5} {name:<28} {mean:>12.4f} {p50:>12.4f} {p90:>12.4f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()