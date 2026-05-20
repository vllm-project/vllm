# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: cpu_topp_sampling kernel, AVX-512 vs AVX2.

Usage:
  numactl -m 5 -N 5 .venv/bin/python benchmarks/kernels/benchmark_cpu_topp.py
  numactl -m 5 -N 5 .venv/bin/python benchmarks/kernels/benchmark_cpu_topp.py --avx2
  numactl -m 5 -N 5 .venv/bin/python benchmarks/kernels/benchmark_cpu_topp.py --both
"""

import argparse
import os
import subprocess
import sys
import time

import torch

CONFIGS = [
    (1, 128256, 0.9),
    (8, 128256, 0.9),
    (32, 128256, 0.9),
    (1, 128256, 0.5),
    (8, 128256, 0.5),
    (32, 128256, 0.5),
]

HEADER = f"{'B':>4}  {'V':>7}  {'top_p':>6}  {'ms/call':>10}  {'GB/s':>8}"
SEP = "-" * 46


def run_bench(warmup: int = 5, iters: int = 20) -> None:
    """Run the benchmark for whichever ISA is already loaded."""
    from vllm._custom_ops import cpu_topp_sampling

    print(HEADER)
    print(SEP)
    for B, V, top_p in CONFIGS:
        logits = torch.randn(B, V, dtype=torch.float32)
        p = torch.full((B,), top_p, dtype=torch.float32)

        for _ in range(warmup):
            cpu_topp_sampling(logits.clone(), p)

        times = []
        for _ in range(iters):
            inp = logits.clone()
            t0 = time.perf_counter()
            cpu_topp_sampling(inp, p)
            times.append(time.perf_counter() - t0)

        times.sort()
        trim = max(1, len(times) // 10)
        ms = sum(times[trim:-trim]) / len(times[trim:-trim]) * 1e3
        bw_gb = 3 * B * V * 4 / (ms * 1e-3) / 1e9
        print(f"{B:>4}  {V:>7}  {top_p:>6.2f}  {ms:>10.3f}  {bw_gb:>8.2f}")


def spawn_for_isa(isa: str, extra_args: list) -> None:
    """Re-invoke this script in a fresh process with VLLM_BENCH_ISA set."""
    env = os.environ.copy()
    env["VLLM_BENCH_ISA"] = isa
    cmd = [sys.executable, __file__] + extra_args
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avx2", action="store_true", help="Benchmark the AVX2 path only"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Benchmark both AVX-512 and AVX2 (spawns subprocesses)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if args.both:
        print("\n=== AVX-512 ===")
        spawn_for_isa(
            "avx512", ["--warmup", str(args.warmup), "--iters", str(args.iters)]
        )
        print("\n=== AVX2 ===")
        spawn_for_isa(
            "avx2", ["--warmup", str(args.warmup), "--iters", str(args.iters)]
        )
        return

    # Determine which lib to load from env (set by spawn_for_isa) or CLI flag.
    isa_env = os.environ.get("VLLM_BENCH_ISA", "")
    use_avx2 = args.avx2 or (isa_env == "avx2")

    _IGNORED = "dynamic module does not define module export function"
    if use_avx2:
        # Load only the AVX2 lib before vllm._custom_ops triggers import_kernels.
        # The lib uses TORCH_LIBRARY_FRAGMENT so a second registration is safe;
        # but we must prevent import_kernels from also loading _C (AVX-512).
        # Patch: monkey-patch current_platform.import_kernels to a no-op, then
        # import _C_AVX2 manually.
        from vllm.platforms import current_platform

        current_platform.import_kernels = lambda: None  # suppress auto-load

        try:
            import vllm._C_AVX2  # noqa: F401
        except ImportError as e:
            if _IGNORED not in str(e):
                raise
        isa_label = "AVX2"
    else:
        isa_label = "AVX-512"

    print(f"\nISA: {isa_label}  (warmup={args.warmup}, iters={args.iters})")
    run_bench(warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
