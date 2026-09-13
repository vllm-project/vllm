# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Micro-benchmark for CPU sampling kernels.

Compares fused Gumbel-max / greedy argmax against the baseline
(softmax → exponential → div → argmax) across vocab and batch sizes.

Usage:
    .venv/bin/python benchmarks/kernels/bench_cpu_sampling.py
    .venv/bin/python benchmarks/kernels/bench_cpu_sampling.py --profile
    .venv/bin/python benchmarks/kernels/bench_cpu_sampling.py --vocab 128256 --batch 16
"""

import argparse
import time

import torch
import vllm._C  # noqa: F401
from torch.profiler import ProfilerActivity, profile, record_function


def baseline_random_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)


def baseline_greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1).view(-1)


def bench_latency(fn, args, n_warmup=20, n_iters=500):
    for _ in range(n_warmup):
        fn(*args)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn(*args)
    elapsed = time.perf_counter() - t0
    return elapsed / n_iters * 1e6  # µs


def run_profile(logits, seeds, n_iters=50):
    """Run torch.profiler and print comparison tables."""
    # Profile baseline random sampling
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof_base_rand:
        for _ in range(n_iters):
            with record_function("baseline_random"):
                baseline_random_sample(logits)

    # Profile fused Gumbel-max
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof_fused:
        for _ in range(n_iters):
            with record_function("fused_gumbel_argmax"):
                torch.ops._C.fused_gumbel_argmax(logits, seeds)

    # Profile baseline greedy
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof_base_grdy:
        for _ in range(n_iters):
            with record_function("baseline_greedy"):
                baseline_greedy_sample(logits)

    # Profile custom greedy
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof_cust_grdy:
        for _ in range(n_iters):
            with record_function("custom_greedy_argmax"):
                torch.ops._C.greedy_argmax(logits)

    B, V = logits.shape
    print(f"\n{'=' * 80}")
    print(f"torch.profiler breakdown  (batch={B}, vocab={V}, iters={n_iters})")
    print(f"{'=' * 80}")

    print("\n--- Baseline Random (softmax → exp → div → argmax) ---")
    print(prof_base_rand.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    print("\n--- Fused Gumbel-max (table lookup + add + argmax) ---")
    print(prof_fused.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    print("\n--- Baseline Greedy (torch.argmax) ---")
    print(prof_base_grdy.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    print("\n--- Custom Greedy (vec_op argmax) ---")
    print(prof_cust_grdy.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    # Summary comparison
    def avg_us(prof, label):
        for e in prof.key_averages():
            if e.key == label:
                return e.cpu_time_total / e.count
        return 0.0

    t_br = avg_us(prof_base_rand, "baseline_random")
    t_fg = avg_us(prof_fused, "fused_gumbel_argmax")
    t_bg = avg_us(prof_base_grdy, "baseline_greedy")
    t_cg = avg_us(prof_cust_grdy, "custom_greedy_argmax")

    print(f"\n{'=' * 60}")
    print(f"  Summary  (batch={B}, vocab={V})")
    print(f"{'=' * 60}")
    print(f"  {'Kernel':<30} {'Avg (µs)':>10} {'Speedup':>10}")
    print(f"  {'-' * 50}")
    print(f"  {'baseline random':<30} {t_br:>10.1f} {'—':>10}")
    print(
        f"  {'fused gumbel-max':<30} {t_fg:>10.1f} "
        f"{t_br / t_fg if t_fg > 0 else 0:>9.2f}x"
    )
    print(f"  {'baseline greedy':<30} {t_bg:>10.1f} {'—':>10}")
    print(
        f"  {'custom greedy':<30} {t_cg:>10.1f} {t_bg / t_cg if t_cg > 0 else 0:>9.2f}x"
    )
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU sampling kernels")
    parser.add_argument(
        "--vocab",
        type=int,
        nargs="+",
        default=[32000, 49152, 128256],
        help="Vocab sizes to benchmark",
    )
    parser.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run torch.profiler and export chrome trace",
    )
    parser.add_argument(
        "--iters", type=int, default=500, help="Iterations per measurement"
    )
    args = parser.parse_args()

    header = (
        f"{'batch':>5}  {'vocab':>7}  "
        f"{'base_rand':>10}  {'fused_rand':>10}  {'rand_spdup':>10}  "
        f"{'base_grdy':>10}  {'cust_grdy':>10}  {'grdy_spdup':>10}"
    )
    units = (
        f"{'':>5}  {'':>7}  "
        f"{'(µs)':>10}  {'(µs)':>10}  {'':>10}  "
        f"{'(µs)':>10}  {'(µs)':>10}  {'':>10}"
    )
    print("\n" + "=" * len(header))
    print("CPU Sampling Kernel Benchmark")
    print("=" * len(header))
    print(header)
    print(units)
    print("-" * len(header))

    for V in args.vocab:
        for B in args.batch:
            logits = torch.randn(B, V, dtype=torch.float32)
            seeds = torch.arange(B, dtype=torch.long)

            t_base_rand = bench_latency(
                baseline_random_sample, (logits,), n_iters=args.iters
            )
            t_fused = bench_latency(
                torch.ops._C.fused_gumbel_argmax, (logits, seeds), n_iters=args.iters
            )

            t_base_grdy = bench_latency(
                baseline_greedy_sample, (logits,), n_iters=args.iters
            )
            t_cust_grdy = bench_latency(
                torch.ops._C.greedy_argmax, (logits,), n_iters=args.iters
            )

            rand_speedup = t_base_rand / t_fused if t_fused > 0 else 0
            grdy_speedup = t_base_grdy / t_cust_grdy if t_cust_grdy > 0 else 0

            print(
                f"{B:>5}  {V:>7}  "
                f"{t_base_rand:>10.1f}  {t_fused:>10.1f}  "
                f"{rand_speedup:>9.2f}x  "
                f"{t_base_grdy:>10.1f}  {t_cust_grdy:>10.1f}  "
                f"{grdy_speedup:>9.2f}x"
            )

    print("-" * len(header))

    if args.profile:
        print("\nRunning torch.profiler (batch=16, vocab=128256) ...")
        logits = torch.randn(16, 128256, dtype=torch.float32)
        seeds = torch.arange(16, dtype=torch.long)

        # warmup
        for _ in range(10):
            baseline_random_sample(logits)
            torch.ops._C.fused_gumbel_argmax(logits, seeds)

        run_profile(logits, seeds)


if __name__ == "__main__":
    main()
