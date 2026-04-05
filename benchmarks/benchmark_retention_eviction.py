# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simulate agentic multi-turn eviction with retention directives.

This script validates that the two-structure evictor (LRU + priority queue)
correctly protects prioritized blocks under memory pressure. It simulates
a workload where:

  1. Multiple "sessions" allocate blocks (simulating multi-turn agentic
     requests with growing context).
  2. Some blocks receive retention priority (simulating system-prompt and
     tool-call-awaiting blocks that the orchestrator wants to protect).
  3. Memory pressure forces eviction.
  4. We measure whether prioritized blocks survive longer than unprioritized
     ones.

No GPU required — runs against the actual BlockPool implementation.

Usage:
    python benchmarks/benchmark_retention_eviction.py \
        --num-gpu-blocks 1000 --block-size 16 --num-sessions 20
"""

import argparse
import time

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def run_simulation(
    num_gpu_blocks: int,
    block_size: int,
    num_sessions: int,
    blocks_per_session: int,
    priority_fraction: float,
    priority_value: int,
):
    """Run the eviction simulation.

    Args:
        num_gpu_blocks: Total blocks in the pool.
        block_size: Tokens per block (for hash_block_size).
        num_sessions: Number of concurrent sessions to simulate.
        blocks_per_session: Blocks allocated per session.
        priority_fraction: Fraction of each session's blocks that get
            retention priority (0.0 - 1.0).
        priority_value: Priority value to assign (0-100).
    """
    pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=block_size,
    )

    # Track which blocks are prioritized vs not.
    prioritized_block_ids: set[int] = set()
    unprioritized_block_ids: set[int] = set()

    sessions: list[list[KVCacheBlock]] = []
    total_allocated = 0

    print(f"Pool: {num_gpu_blocks} blocks, {num_gpu_blocks - 1} usable")
    print(f"Sessions: {num_sessions}, {blocks_per_session} blocks each")
    print(
        f"Priority: {priority_fraction * 100:.0f}% of blocks at "
        f"priority={priority_value}"
    )
    print(f"Total demand: {num_sessions * blocks_per_session} blocks")
    print()

    # Phase 1: Allocate sessions until we run out of blocks.
    for i in range(num_sessions):
        free = pool.get_num_free_blocks()
        if free < blocks_per_session:
            print(
                f"  Session {i}: only {free} free blocks, "
                f"need {blocks_per_session} — stopping allocation"
            )
            break

        blocks = pool.get_new_blocks(blocks_per_session)
        total_allocated += len(blocks)

        # Mark some blocks as prioritized.
        num_priority = int(len(blocks) * priority_fraction)
        for j, block in enumerate(blocks):
            if j < num_priority:
                block.priority = priority_value
                prioritized_block_ids.add(block.block_id)
            else:
                unprioritized_block_ids.add(block.block_id)

        sessions.append(blocks)
        print(
            f"  Session {i}: allocated {len(blocks)} blocks "
            f"({num_priority} prioritized), "
            f"{pool.get_num_free_blocks()} free remaining"
        )

    print(f"\nTotal allocated: {total_allocated}")
    print(f"  Prioritized: {len(prioritized_block_ids)}")
    print(f"  Unprioritized: {len(unprioritized_block_ids)}")

    # Phase 2: Free all sessions (simulating requests completing).
    for session_blocks in sessions:
        pool.free_blocks(session_blocks)

    print("\nAfter freeing all sessions:")
    print(f"  LRU free list: {pool.free_block_queue.num_free_blocks}")
    print(f"  Priority queue: {pool.priority_eviction_queue.num_blocks}")
    print(f"  Total free: {pool.get_num_free_blocks()}")

    # Phase 3: Allocate under pressure and track which blocks get evicted.
    eviction_target = total_allocated // 2
    print(f"\nEvicting {eviction_target} blocks (allocating new ones)...")

    evicted_prioritized = 0
    evicted_unprioritized = 0
    evicted_other = 0

    t_start = time.monotonic()
    new_blocks = pool.get_new_blocks(eviction_target)
    t_elapsed = time.monotonic() - t_start

    for block in new_blocks:
        bid = block.block_id
        if bid in unprioritized_block_ids:
            evicted_unprioritized += 1
        elif bid in prioritized_block_ids:
            evicted_prioritized += 1
        else:
            evicted_other += 1

    print(f"  Time: {t_elapsed * 1000:.2f} ms")
    print(f"  Evicted from unprioritized: {evicted_unprioritized}")
    print(f"  Evicted from prioritized: {evicted_prioritized}")
    print(f"  Evicted from other (null/fresh): {evicted_other}")

    # Verify correctness: unprioritized blocks should be evicted first.
    total_unprioritized = len(unprioritized_block_ids)
    if eviction_target <= total_unprioritized:
        # Should not have touched any prioritized blocks.
        if evicted_prioritized == 0:
            print("\n  PASS: No prioritized blocks evicted (all evictions from LRU)")
        else:
            print(
                f"\n  FAIL: {evicted_prioritized} prioritized blocks "
                f"evicted when {total_unprioritized} unprioritized "
                f"were available"
            )
    else:
        # Some prioritized blocks must be evicted.
        expected_from_priority = eviction_target - total_unprioritized
        print(
            f"\n  INFO: Needed {expected_from_priority} from priority "
            f"queue (exhausted {total_unprioritized} unprioritized)"
        )
        if evicted_unprioritized == total_unprioritized:
            print("  PASS: All unprioritized evicted before any prioritized")
        else:
            print(
                f"  FAIL: Only {evicted_unprioritized}/{total_unprioritized}"
                f" unprioritized evicted before touching prioritized"
            )

    # Phase 4: Verify remaining priority queue state.
    remaining_in_pq = pool.priority_eviction_queue.num_blocks
    surviving_prioritized = len(prioritized_block_ids) - evicted_prioritized
    print(f"\n  Surviving prioritized blocks: {surviving_prioritized}")
    print(f"  Priority queue size: {remaining_in_pq}")

    return evicted_prioritized == 0 or evicted_unprioritized == len(
        unprioritized_block_ids
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark retention-based eviction")
    parser.add_argument(
        "--num-gpu-blocks", type=int, default=1000, help="Total KV-cache blocks in pool"
    )
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per block")
    parser.add_argument(
        "--num-sessions", type=int, default=20, help="Number of concurrent sessions"
    )
    parser.add_argument(
        "--blocks-per-session", type=int, default=40, help="Blocks per session"
    )
    parser.add_argument(
        "--priority-fraction",
        type=float,
        default=0.3,
        help="Fraction of blocks with priority (0-1)",
    )
    parser.add_argument(
        "--priority-value",
        type=int,
        default=80,
        help="Priority value for retained blocks (0-100)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Retention-Based Eviction Benchmark")
    print("=" * 60)
    print()

    success = run_simulation(
        num_gpu_blocks=args.num_gpu_blocks,
        block_size=args.block_size,
        num_sessions=args.num_sessions,
        blocks_per_session=args.blocks_per_session,
        priority_fraction=args.priority_fraction,
        priority_value=args.priority_value,
    )

    print()
    print("=" * 60)
    print(f"Result: {'PASS' if success else 'FAIL'}")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
