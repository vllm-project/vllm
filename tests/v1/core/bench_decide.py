#!/usr/bin/env python3
"""bench_decide.py — ns-level host microbenchmark of should_load.

Run on the vLLM main checkout (no GPU needed):
    python tests/v1/core/bench_decide.py

Expected: individual-to-tens of ns per call (a pure integer comparison),
far below any scheduling/network cost — evidence for the PR description.
"""

from __future__ import annotations

# Standard
import statistics
import time

# Local
from vllm.config.kv_transfer import KVLoadPolicyConfig
from vllm.v1.core.kv_load_policy import StaticThresholdWindowPolicy

N = 1_000_000


def bench(cfg: KVLoadPolicyConfig, matched: int, label: str) -> None:
    p = StaticThresholdWindowPolicy(cfg)
    # warmup
    for _ in range(10_000):
        p.should_load(matched_tokens=matched)
    t0 = time.perf_counter_ns()
    for _ in range(N):
        p.should_load(matched_tokens=matched)
    dt = time.perf_counter_ns() - t0
    per = dt / N
    print(f"{label:34s} {per:8.1f} ns/call  ({N} iters)")

    # variance via repeated batches
    samples = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        for _ in range(N // 20):
            p.should_load(matched_tokens=matched)
        samples.append((time.perf_counter_ns() - t0) / (N // 20))
    print(
        f"{'':34s} p50={statistics.median(samples):.1f} "
        f"min={min(samples):.1f} max={max(samples):.1f} ns/call"
    )


def main() -> None:
    print(f"StaticThresholdWindowPolicy.should_load microbench, n={N} per config")
    bench(KVLoadPolicyConfig(tau1=1024, tau2=23296), 4096, "window, matched=4096")
    bench(KVLoadPolicyConfig(tau1=1024, tau2=23296), 23552, "window, matched=23552")
    bench(
        KVLoadPolicyConfig(tau1=1024, tau2=None),
        4096,
        "single-threshold, matched=4096",
    )
    bench(
        KVLoadPolicyConfig(tau1=1024, tau2=None),
        512,
        "single-threshold, matched=512",
    )


if __name__ == "__main__":
    main()
