# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: CPU top-p / top-k / joint sampling kernels.

Usage:
  numactl -m 4 -N 4 .venv/bin/python benchmarks/kernels/benchmark_cpu_topp.py
  numactl -m 4 -N 4 .venv/bin/python benchmarks/kernels/benchmark_cpu_topp.py --avx2
  numactl -m 4 -N 4 .venv/bin/python benchmarks/kernels/benchmark_cpu_topp.py --both
  numactl -m 4 -N 4 .venv/bin/python benchmarks/kernels/benchmark_cpu_topp.py \
      --mode topk
  numactl -m 4 -N 4 .venv/bin/python benchmarks/kernels/benchmark_cpu_topp.py \
      --mode joint
  numactl -m 4 -N 4 .venv/bin/python benchmarks/kernels/benchmark_cpu_topp.py \
      --mode all
"""

import argparse
import os
import subprocess
import sys
import time

import torch

CONFIGS_TOPP = [
    (1, 128256, 0.9),
    (8, 128256, 0.9),
    (32, 128256, 0.9),
    (1, 128256, 0.5),
    (8, 128256, 0.5),
    (32, 128256, 0.5),
]

CONFIGS_TOPK = [
    (1, 128256, 50),
    (8, 128256, 50),
    (32, 128256, 50),
    (1, 128256, 1024),
    (8, 128256, 1024),
    (32, 128256, 1024),
]

CONFIGS_JOINT = [
    (1, 128256, 50, 0.9),
    (8, 128256, 50, 0.9),
    (32, 128256, 50, 0.9),
    (1, 128256, 1024, 0.9),
    (8, 128256, 1024, 0.9),
    (32, 128256, 1024, 0.9),
]


def _time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    trim = max(1, len(times) // 10)
    return sum(times[trim:-trim]) / len(times[trim:-trim]) * 1e3


def _ops_ns():
    """Return the torch ops namespace that has the CPU sampling ops."""
    import vllm._custom_ops  # noqa: F401 — triggers import_kernels

    for ns_name in ("_C", "_C_AVX512", "_C_AVX2"):
        ns = getattr(torch.ops, ns_name, None)
        if ns is not None:
            try:
                _ = ns.cpu_topp_sampling
                return ns
            except AttributeError:
                pass
    raise RuntimeError("No CPU sampling ops namespace found")


def run_bench_topp(warmup: int = 5, iters: int = 20) -> None:
    ns = _ops_ns()
    cpu_topp_sampling = ns.cpu_topp_sampling
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    header = (
        f"{'B':>4}  {'V':>7}  {'top_p':>6}"
        f"  {'before_ms':>10}  {'after_ms':>9}  {'speedup':>7}"
    )
    print("\n--- top-p only ---")
    print(header)
    print("-" * 55)
    for B, V, top_p in CONFIGS_TOPP:
        logits = torch.randn(B, V, dtype=torch.float32)
        p_f32 = torch.full((B,), top_p, dtype=torch.float32)
        p_one = torch.full((B,), top_p, dtype=torch.float32)

        before_fn = lambda lg=logits, po=p_one: apply_top_k_top_p_pytorch(  # noqa: E731
            lg.clone(), None, po, allow_cpu_sync=True
        )
        after_fn = lambda lg=logits, pf=p_f32: cpu_topp_sampling(  # noqa: E731
            lg.clone(), pf
        )

        ms_before = _time_fn(before_fn, warmup, iters)
        ms_after = _time_fn(after_fn, warmup, iters)
        speedup = ms_before / ms_after
        print(
            f"{B:>4}  {V:>7}  {top_p:>6.2f}"
            f"  {ms_before:>10.3f}  {ms_after:>9.3f}  {speedup:>7.2f}x"
        )


def run_bench_topk(warmup: int = 5, iters: int = 20) -> None:
    ns = _ops_ns()
    cpu_topk_sampling = ns.cpu_topk_sampling
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_only

    header = (
        f"{'B':>4}  {'V':>7}  {'top_k':>6}"
        f"  {'before_ms':>10}  {'after_ms':>9}  {'speedup':>7}"
    )
    print("\n--- top-k only ---")
    print(header)
    print("-" * 55)
    for B, V, top_k in CONFIGS_TOPK:
        logits = torch.randn(B, V, dtype=torch.float32)
        k_i32 = torch.full((B,), top_k, dtype=torch.int32)

        before_fn = lambda lg=logits, ki=k_i32: apply_top_k_only(  # noqa: E731
            lg.clone(), ki.long()
        )
        after_fn = lambda lg=logits, ki=k_i32: cpu_topk_sampling(  # noqa: E731
            lg.clone(), ki
        )

        ms_before = _time_fn(before_fn, warmup, iters)
        ms_after = _time_fn(after_fn, warmup, iters)
        speedup = ms_before / ms_after
        print(
            f"{B:>4}  {V:>7}  {top_k:>6}"
            f"  {ms_before:>10.3f}  {ms_after:>9.3f}  {speedup:>7.2f}x"
        )


def run_bench_joint(warmup: int = 5, iters: int = 20) -> None:
    ns = _ops_ns()
    cpu_topp_sampling = ns.cpu_topp_sampling
    cpu_topk_topp_sampling = ns.cpu_topk_topp_sampling
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_only

    header = (
        f"{'B':>4}  {'V':>7}  {'k':>5}  {'p':>4}"
        f"  {'before_ms':>10}  {'after_ms':>9}  {'speedup':>7}"
    )
    print("\n--- top-k + top-p (joint) ---")
    print(header)
    print("-" * 62)
    for B, V, top_k, top_p in CONFIGS_JOINT:
        logits = torch.randn(B, V, dtype=torch.float32)
        k_i32 = torch.full((B,), top_k, dtype=torch.int32)
        p_f32 = torch.full((B,), top_p, dtype=torch.float32)

        def before_fn(  # noqa: E731
            lg=logits, ki=k_i32, pf=p_f32
        ):
            tmp = apply_top_k_only(lg.clone(), ki.long())
            cpu_topp_sampling(tmp.contiguous().clone(), pf)

        after_fn = lambda lg=logits, ki=k_i32, pf=p_f32: (  # noqa: E731
            cpu_topk_topp_sampling(lg.clone(), ki, pf)
        )

        ms_before = _time_fn(before_fn, warmup, iters)
        ms_after = _time_fn(after_fn, warmup, iters)
        speedup = ms_before / ms_after
        print(
            f"{B:>4}  {V:>7}  {top_k:>5}  {top_p:>4.2f}"
            f"  {ms_before:>10.3f}  {ms_after:>9.3f}  {speedup:>7.2f}x"
        )


def spawn_for_isa(isa: str, extra_args: list) -> None:
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
        "--both", action="store_true", help="Benchmark both AVX-512 and AVX2"
    )
    parser.add_argument(
        "--mode", choices=["topp", "topk", "joint", "all"], default="all"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if args.both:
        extra = [
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--mode",
            args.mode,
        ]
        print("\n=== AVX-512 ===")
        spawn_for_isa("avx512", extra)
        print("\n=== AVX2 ===")
        spawn_for_isa("avx2", extra)
        return

    isa_env = os.environ.get("VLLM_BENCH_ISA", "")
    use_avx2 = args.avx2 or (isa_env == "avx2")

    _IGNORED = "dynamic module does not define module export function"
    if use_avx2:
        from vllm.platforms import current_platform

        current_platform.import_kernels = lambda: None
        try:
            import vllm._C_AVX2  # noqa: F401
        except ImportError as e:
            if _IGNORED not in str(e):
                raise
        isa_label = "AVX2"
    else:
        isa_label = "AVX-512"

    print(f"\nISA: {isa_label}  (warmup={args.warmup}, iters={args.iters})")

    if args.mode in ("topp", "all"):
        run_bench_topp(warmup=args.warmup, iters=args.iters)
    if args.mode in ("topk", "all"):
        run_bench_topk(warmup=args.warmup, iters=args.iters)
    if args.mode in ("joint", "all"):
        run_bench_joint(warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
