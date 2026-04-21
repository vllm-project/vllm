"""CPU microbenchmark: Python vllm.v1.core.block_pool.BlockPool vs Rust vllm_rs.BlockPool.

Exercises the hot methods the scheduler calls every step:
  - get_new_blocks (allocation from free queue + eviction)
  - free_blocks   (ref-count drop + return to queue)
  - touch         (prefix-cache hit path: lift from free queue + inc ref)

Shape of the workload is a coarse approximation of the scheduler's
steady-state decode phase: a stable set of N "live requests" each holding
a block table of length L, with a churn ratio that free/realloc's some
fraction every tick.

Run (after `maturin develop --release` in vllm_rs/):
    python vllm_rs/bench_block_pool.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# Allow running from anywhere: put the vllm repo root (parent of vllm_rs/)
# on sys.path so `import vllm` hits the local checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vllm_rs as rs
from vllm.v1.core.block_pool import BlockPool as PyBlockPool


def time_ns_per_op(fn, iters: int) -> float:
    # Warmup
    fn()
    fn()
    samples = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter_ns() - t0) / iters)
    return statistics.median(samples)


def build_pools(num_blocks: int, caching: bool):
    py = PyBlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=caching,
        hash_block_size=16,
    )
    rbp = rs.BlockPool(num_blocks, caching, 16)
    return py, rbp


def bench_alloc_free(pool, batch: int, blocks_per_req: int):
    """One tick: allocate N batches of K blocks, then free them all."""
    handles = []
    for _ in range(batch):
        handles.append(pool.get_new_blocks(blocks_per_req))
    for h in handles:
        pool.free_blocks(h)


def bench_mixed(pool, batch: int, blocks_per_req: int, churn: int):
    """Steady-state: maintain `batch` live allocations, rotate `churn` per tick."""
    live = [pool.get_new_blocks(blocks_per_req) for _ in range(batch)]
    def step():
        # Free `churn` oldest, allocate `churn` new
        for _ in range(churn):
            pool.free_blocks(live.pop(0))
        for _ in range(churn):
            live.append(pool.get_new_blocks(blocks_per_req))
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-blocks", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256, help="# of live requests")
    ap.add_argument("--blocks-per-req", type=int, default=8)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument(
        "--scenarios",
        default="alloc_free,mixed_10,mixed_1",
        help="comma-separated list from {alloc_free, mixed_N (N = churn)}",
    )
    args = ap.parse_args()

    print(
        f"num_blocks={args.num_blocks}  batch={args.batch}  "
        f"blocks_per_req={args.blocks_per_req}  iters={args.iters}"
    )
    print()
    print(f"{'scenario':<16}{'py (us/tick)':>16}{'rs (us/tick)':>16}{'speedup':>12}")
    print("-" * 60)

    for scn in args.scenarios.split(","):
        for caching in (True, False):
            py, rbp = build_pools(args.num_blocks, caching)

            if scn == "alloc_free":
                py_ns = time_ns_per_op(
                    lambda: bench_alloc_free(py, args.batch, args.blocks_per_req),
                    args.iters,
                )
                rs_ns = time_ns_per_op(
                    lambda: bench_alloc_free(rbp, args.batch, args.blocks_per_req),
                    args.iters,
                )
            elif scn.startswith("mixed_"):
                churn = int(scn.split("_")[1])
                py_step = bench_mixed(py, args.batch, args.blocks_per_req, churn)
                rs_step = bench_mixed(rbp, args.batch, args.blocks_per_req, churn)
                py_ns = time_ns_per_op(py_step, args.iters)
                rs_ns = time_ns_per_op(rs_step, args.iters)
            else:
                print(f"  unknown scenario {scn}")
                continue

            label = f"{scn}/{'cache' if caching else 'nocache'}"
            print(
                f"{label:<16}{py_ns/1e3:>15.2f} {rs_ns/1e3:>15.2f} {py_ns/rs_ns:>11.2f}x"
            )


if __name__ == "__main__":
    main()
