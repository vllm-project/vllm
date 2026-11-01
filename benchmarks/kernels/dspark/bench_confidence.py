#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone benchmark for the DSpark confidence head fused kernel.

Compares:
  - naive:    torch.cat + .float() + F.linear + sigmoid       (vLLM today, uncompiled)
  - compiled: same wrapped in torch.compile                   (vLLM today, compiled)
  - fused:    TileLang single-kernel implementation           (this RFC)

Metrics:
  - latency      (median of N iters, CUDA events)
  - memory delta (torch.cuda.max_memory_allocated during call)
  - numerical    (max rel err vs fp64 ground truth)

Usage:
    python benchmarks/kernels/dspark/bench_confidence.py
    python benchmarks/kernels/dspark/bench_confidence.py \\
        --n-list 32,512,2048 --H-list 7168 --r-list 8
    python benchmarks/kernels/dspark/bench_confidence.py --fast     # quick sanity
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _pytorch_naive(hidden, markov, weight):
    """Mirrors vLLM current path: cat -> .float() -> linear -> sigmoid.

    Caller is responsible for trimming markov to actual rank r. When r=0
    this function receives just `hidden` and weight has length H.
    """
    if markov.shape[-1] > 0 and weight.shape[0] == hidden.shape[-1] + markov.shape[-1]:
        x = torch.cat([hidden, markov], dim=-1).float()
    else:
        x = hidden.float()
    return torch.sigmoid(F.linear(x, weight.unsqueeze(0)).squeeze(-1))


_pytorch_compiled = torch.compile(_pytorch_naive)


def _reference_fp64(hidden, markov, weight, temp, r):
    h = hidden.to(torch.float64)
    w = weight.to(torch.float64)
    x = torch.cat([h, markov[:, :r].to(torch.float64)], dim=-1) if r > 0 else h
    return torch.sigmoid((x @ w) / temp)


# ---------------------------------------------------------------------------
# Benchmark plumbing
# ---------------------------------------------------------------------------


@dataclass
class Result:
    impl: str
    n: int
    H: int
    r: int
    temp: float
    latency_ms_median: float
    memory_delta_bytes: int
    max_rel_err: float

    @property
    def memory_delta_mb(self) -> float:
        return self.memory_delta_bytes / (1024 * 1024)


def _bench(fn, iters: int, warmup: int = 20) -> float:
    """Median latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _peak_memory_delta_bytes(fn) -> int:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return max(0, peak - base)


def _max_rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (out.double() - ref).abs()
    rel = diff / ref.clamp_min(1e-8)
    return rel.max().item()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_benchmark(n, H, r, temp, iters, check_accuracy=True):
    """Run all 3 implementations at one (n, H, r) point."""
    torch.manual_seed(0)
    hidden = torch.randn(n, H, dtype=torch.bfloat16, device="cuda")
    markov = torch.randn(n, max(r, 1), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(H + r, dtype=torch.float32, device="cuda") * 0.02
    out_fused = torch.zeros(n, dtype=torch.float32, device="cuda")

    # Lazy import so that this script doesn't crash if tilelang isn't there.
    try:
        from vllm.model_executor.kernels.dspark import (  # noqa: F401
            dspark_confidence_torch,
            is_dspark_fused_available,
        )

        have_fused = is_dspark_fused_available()
    except Exception as e:
        print(f"  [warn] fused kernel unavailable: {e}")
        have_fused = False

    ref = _reference_fp64(hidden, markov[:, :r] if r > 0 else markov, weight, temp, r)

    # ---- NAIVE ----
    def naive_call():
        return _pytorch_naive(hidden, markov[:, :r] if r > 0 else markov, weight)

    lat_naive = _bench(naive_call, iters)
    mem_naive = _peak_memory_delta_bytes(naive_call)
    err_naive = _max_rel_err(naive_call(), ref) if check_accuracy else float("nan")

    # ---- COMPILED ----
    def compiled_call():
        return _pytorch_compiled(hidden, markov[:, :r] if r > 0 else markov, weight)

    # torch.compile warmup (compile once, then measure)
    compiled_call()
    torch.cuda.synchronize()
    lat_compiled = _bench(compiled_call, iters)
    mem_compiled = _peak_memory_delta_bytes(compiled_call)
    err_compiled = (
        _max_rel_err(compiled_call(), ref) if check_accuracy else float("nan")
    )

    # ---- FUSED ----
    if have_fused:

        def fused_call():
            torch.ops.vllm.dspark_confidence_tilelang(
                hidden, markov, weight, out_fused, temp
            )
            return out_fused

        # Compile-trigger call once (TileLang JIT caches by static dims).
        fused_call()
        torch.cuda.synchronize()
        lat_fused = _bench(fused_call, iters)
        mem_fused = _peak_memory_delta_bytes(fused_call)
        err_fused = _max_rel_err(out_fused, ref) if check_accuracy else float("nan")
    else:
        lat_fused = float("nan")
        mem_fused = 0
        err_fused = float("nan")

    return [
        Result("naive", n, H, r, temp, lat_naive, mem_naive, err_naive),
        Result("compiled", n, H, r, temp, lat_compiled, mem_compiled, err_compiled),
        Result("fused", n, H, r, temp, lat_fused, mem_fused, err_fused),
    ]


def format_table(results: list[Result]) -> str:
    lines = []
    header = (
        "impl",
        "n",
        "H",
        "r",
        "temp",
        "lat_ms (median)",
        "mem_delta_MB",
        "max_rel_err",
    )
    lines.append(
        f"{header[0]:<10} {header[1]:>6} {header[2]:>6} {header[3]:>3} "
        f"{header[4]:>5} {header[5]:>15} {header[6]:>14} {header[7]:>12}"
    )
    for r in results:
        lines.append(
            f"{r.impl:<10} {r.n:>6d} {r.H:>6d} {r.r:>3d} "
            f"{r.temp:>5.2f} {r.latency_ms_median:>15.4f} "
            f"{r.memory_delta_mb:>14.2f} {r.max_rel_err:>12.3e}"
        )
    return "\n".join(lines)


def summarise(results: list[Result]) -> dict:
    """Group by (n, H, r) and compute speedups."""
    by_shape: dict[tuple[int, int, int], dict[str, Result]] = {}
    for r in results:
        by_shape.setdefault((r.n, r.H, r.r), {})[r.impl] = r

    summary = {"shapes": []}
    for (n, H, r), impls in by_shape.items():
        c = impls.get("compiled")
        f = impls.get("fused")
        if c is None or f is None:
            continue
        speedup = (
            c.latency_ms_median / f.latency_ms_median
            if f.latency_ms_median > 0
            else float("nan")
        )
        mem_ratio = (
            f.memory_delta_bytes / c.memory_delta_bytes
            if c.memory_delta_bytes > 0
            else float("nan")
        )
        err_ratio = f.max_rel_err / max(c.max_rel_err, 1e-12)
        summary["shapes"].append(
            {
                "n": n,
                "H": H,
                "r": r,
                "speedup_lat_vs_compiled": speedup,
                "memory_ratio_vs_compiled": mem_ratio,
                "err_ratio_vs_compiled": err_ratio,
            }
        )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n-list", default="32,128,512,2048,8192", help="comma-separated batch sizes"
    )
    p.add_argument("--H-list", default="2048,7168", help="comma-separated hidden sizes")
    p.add_argument("--r-list", default="8,32", help="comma-separated markov ranks")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument(
        "--no-accuracy",
        action="store_true",
        help="skip numeric accuracy check (faster)",
    )
    p.add_argument("--out", default=None, help="write results to JSON")
    p.add_argument(
        "--fast",
        action="store_true",
        help="quick sanity: a couple of small shapes only",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark requires a GPU.")
        sys.exit(1)

    print(
        f"# Device: {torch.cuda.get_device_name(0)}  "
        f"CC={torch.cuda.get_device_capability(0)}"
    )
    print(f"# PyTorch {torch.__version__}  CUDA {torch.version.cuda}\n")

    if args.fast:
        n_list = [32, 512]
        H_list = [2048]
        r_list = [8]
        args.iters = min(args.iters, 50)
    else:
        n_list = [int(x) for x in args.n_list.split(",")]
        H_list = [int(x) for x in args.H_list.split(",")]
        r_list = [int(x) for x in args.r_list.split(",")]

    all_results: list[Result] = []
    for H in H_list:
        for r in r_list:
            for n in n_list:
                print(f"\n=== n={n} H={H} r={r} temp={args.temp} ===")
                shaped = run_benchmark(
                    n, H, r, args.temp, args.iters, check_accuracy=not args.no_accuracy
                )
                for res in shaped:
                    print(
                        f"  {res.impl:<10}  lat={res.latency_ms_median:>8.4f} ms "
                        f"  memΔ={res.memory_delta_mb:>7.2f} MB "
                        f"  max_rel_err={res.max_rel_err:.3e}"
                    )
                all_results.extend(shaped)

    print("\n\n===== Full table =====")
    print(format_table(all_results))

    print("\n===== Summary vs compiled =====")
    summary = summarise(all_results)
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(0),
                    "cc": torch.cuda.get_device_capability(0),
                    "results": [asdict(r) for r in all_results],
                    "summary": summary,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
