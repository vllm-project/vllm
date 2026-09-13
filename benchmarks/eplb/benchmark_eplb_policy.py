"""Benchmark the batched EPLB policy against layer-by-layer packing."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from vllm.distributed.eplb.policy.batched import BatchedEplbPolicy
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy


def measure(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(7)

    for layers, experts, physical, groups, nodes, ranks in (
        (1, 8, 12, 4, 1, 4),
        (16, 64, 72, 8, 2, 8),
        (58, 256, 288, 8, 1, 8),
        (58, 256, 288, 8, 4, 32),
        (58, 256, 288, 8, 18, 144),
    ):
        weight = torch.randint(0, 1000, (layers, experts), generator=generator)
        arguments = (weight, physical, groups, nodes, ranks)
        old_ms = measure(
            lambda arguments=arguments: DefaultEplbPolicy.rebalance_experts(*arguments),
            args.warmup,
            args.repeats,
        )
        new_ms = measure(
            lambda arguments=arguments: BatchedEplbPolicy.rebalance_experts(*arguments),
            args.warmup,
            args.repeats,
        )
        print(
            f"layers={layers:2d} experts={experts:3d} ranks={ranks:3d} "
            f"old={old_ms:8.3f} ms new={new_ms:8.3f} ms "
            f"speedup={old_ms / new_ms:6.2f}x"
        )


if __name__ == "__main__":
    main()
