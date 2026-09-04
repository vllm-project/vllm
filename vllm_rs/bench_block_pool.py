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


def bench_realistic(pool, batch: int, blocks_per_req: int, churn: int, is_rust: bool):
    """Includes cache_full_blocks in the hot loop — closer to what
    KVCacheManager.allocate_slots actually does on every scheduling decision.

    Each "churn" step: free oldest request's blocks, allocate new ones, then
    stamp them as full blocks in the prefix cache (common case: allocation
    followed by hashing all the new blocks).
    """
    live = [pool.get_new_blocks(blocks_per_req) for _ in range(batch)]
    counter = [0]
    # Pre-generate unique hash seeds so each call has fresh keys
    def step():
        counter[0] += 1
        c = counter[0]
        for _ in range(churn):
            pool.free_blocks(live.pop(0))
        for i in range(churn):
            blocks = pool.get_new_blocks(blocks_per_req)
            live.append(blocks)
            # Fresh hashes derived from (counter, i, j) so entries don't collide
            block_hashes = [
                (c * 1_000_000 + i * 1024 + j).to_bytes(32, "big")
                for j in range(blocks_per_req)
            ]
            if is_rust:
                pool.cache_full_blocks_fast(blocks, block_hashes, 0, blocks_per_req, 0)
            else:
                # Python reference: same work as cache_full_blocks_fast inline
                from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
                cache_map = pool.cached_block_hash_to_block
                for blk, h in zip(blocks, block_hashes):
                    if blk.is_null:
                        continue
                    if blk.block_hash is not None:
                        continue
                    key = make_block_hash_with_group_id(h, 0)
                    blk.block_hash = key
                    cache_map.insert(key, blk)
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-blocks", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256, help="# of live requests")
    ap.add_argument("--blocks-per-req", type=int, default=8)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument(
        "--scenarios",
        default="alloc_free,mixed_10,mixed_1,realistic_10,realistic_1",
        help="comma-separated list from {alloc_free, mixed_N, realistic_N}",
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
            elif scn.startswith("realistic_"):
                if not caching:
                    # Scheduler only stamps block hashes when caching is on;
                    # realistic_* without caching just duplicates mixed_*.
                    continue
                churn = int(scn.split("_")[1])
                py_step = bench_realistic(py, args.batch, args.blocks_per_req, churn, is_rust=False)
                rs_step = bench_realistic(rbp, args.batch, args.blocks_per_req, churn, is_rust=True)
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
