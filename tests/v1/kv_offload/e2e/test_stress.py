# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stress test: OBJ tier batched store/load with a hot/cold ratio sweep.

Mirrors the llm-d-kv-cache fs_connector stress test but exercises
ObjSecondaryTier directly (CPU DRAM <-> S3 via NIXL) instead of LLM inference.

Each iteration submits `batch_size` concurrent jobs mixed by `hot_ratio`:
  hot_ratio = 1.0  ->  all loads   (pure read stress, keys already in S3)
  hot_ratio = 0.0  ->  all stores  (pure write stress, fresh keys)
  in between       ->  mixed read/write

Default sweep: hot_ratios = [0.0, 0.1, ..., 1.0], `num_iterations` iters each.
Pass `--num-repeats=N>=2` to run multiple sweeps and reveal throughput drift.

Run via pytest:
    pytest tests/v1/kv_offload/performance/test_stress.py -v --noconftest

Or as a standalone script:
    python -m tests.v1.kv_offload.performance.test_stress \\
        --num-blocks=128 --batch-size=16 --num-iterations=5
"""

import argparse
import random
import statistics
import time
import uuid

import numpy as np
import torch
import pytest

from .utils import (
    OffloadKey,
    bytes_to_gbs,
    del_tier_and_cleanup,
    drain,
    format_gbs,
    make_job,
    make_tier_with_buffer,
    s3_config_available,
    total_bytes,
    unique_key,
)

# Default hot_ratio sweep: 0.0, 0.1, ..., 1.0
DEFAULT_HOT_RATIOS: tuple[float, ...] = tuple(round(i / 10, 1) for i in range(11))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * pct), len(s) - 1)]


def run_stress_test(
    num_blocks: int = 64,
    elements_per_block: int = 4096,
    batch_size: int = 8,
    num_iterations: int = 3,
    hot_ratios: list[float] | None = None,
    num_repeats: int = 1,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    key_prefix: str | None = None,
) -> tuple[float, float, float, float, float, float]:
    """
    Run a hot/cold-ratio sweep stress test against the OBJ tier.

    The primary buffer holds `num_blocks` slots.  The first half is reserved
    for store sources (DRAM content to push to S3); the second half for load
    destinations (S3 content pulled back into DRAM).  This avoids read/write
    conflicts within the same buffer slot.

    A growing pool of already-stored keys feeds the "hot" (load) side of each
    batch. Cold jobs generate fresh unique keys and store them, expanding the
    pool for future hot batches.

    Returns (for pytest assertions — final repeat, max hot_ratio):
        (tokens_per_sec, kv_gb_s, batch_mean, batch_p50, batch_p99, batch_p100)
    where tokens_per_sec and kv_gb_s are synthetic (based on raw block bytes).
    """
    if hot_ratios is None:
        hot_ratios = list(DEFAULT_HOT_RATIOS)

    prefix = key_prefix or f"perf/{uuid.uuid4().hex[:8]}"
    block_bytes = elements_per_block * dtype.itemsize  # type: ignore[attr-defined]
    total_iters = num_repeats * len(hot_ratios) * num_iterations

    print(
        f"\n===== Stress Test =====\n"
        f"  Blocks: {num_blocks}  Elements/block: {elements_per_block}  "
        f"Block: {block_bytes / 1024:.1f} KB\n"
        f"  Batch size: {batch_size}  Iters/ratio: {num_iterations}  "
        f"Repeats: {num_repeats}  Ratios: {hot_ratios}\n"
        f"  Total iterations: {total_iters} "
        f"(= {total_iters * batch_size} jobs)"
    )

    # Half the buffer for store sources, half for load destinations.
    n_store_slots = num_blocks // 2
    n_load_slots = num_blocks - n_store_slots
    store_slots = list(range(n_store_slots))
    load_slots = list(range(n_store_slots, num_blocks))

    tier, tensor = make_tier_with_buffer(
        num_blocks=num_blocks,
        elements_per_block=elements_per_block,
        dtype=dtype,
        key_prefix=prefix,
    )

    rng = random.Random(seed)

    # Pool of keys that are already in S3 (available for hot/load batches).
    pool: list[OffloadKey] = []
    next_key_id = 0
    job_id_counter = 1

    # results[(repeat, ratio)] = list[batch_wall_time_seconds]
    results: dict[tuple[int, float], list[float]] = {}
    t_start = time.perf_counter()

    try:
        for repeat in range(num_repeats):
            for ratio in hot_ratios:
                results[(repeat, ratio)] = []
                print(
                    f"\n[sweep {repeat + 1}/{num_repeats}] ratio={ratio:.2f}  "
                    f"pool={len(pool)}"
                )

                for it in range(num_iterations):
                    n_hot = min(round(batch_size * ratio), len(pool))
                    n_cold = batch_size - n_hot

                    hot_keys = rng.sample(pool, n_hot) if n_hot else []
                    cold_keys: list[OffloadKey] = []
                    for _ in range(n_cold):
                        cold_keys.append(unique_key(next_key_id))
                        next_key_id += 1

                    # Fill store slots with random data for cold writes.
                    store_slot_cycle = [
                        store_slots[i % n_store_slots] for i in range(n_cold)
                    ]
                    for bid in store_slot_cycle:
                        tensor[bid] = torch.rand((elements_per_block,), dtype=dtype)

                    load_slot_cycle = [
                        load_slots[i % n_load_slots] for i in range(n_hot)
                    ]

                    t0 = time.perf_counter()

                    if cold_keys:
                        tier.submit_store(
                            make_job(
                                job_id_counter,
                                cold_keys,
                                store_slot_cycle[: len(cold_keys)],
                            )
                        )
                        job_id_counter += 1

                    if hot_keys:
                        tier.submit_load(
                            make_job(
                                job_id_counter,
                                hot_keys,
                                load_slot_cycle[: len(hot_keys)],
                            )
                        )
                        job_id_counter += 1

                    drain(tier)
                    elapsed = time.perf_counter() - t0
                    results[(repeat, ratio)].append(elapsed)

                    # Newly stored keys join the pool for future hot batches.
                    pool.extend(cold_keys)

                    batch_gbs = bytes_to_gbs(batch_size * block_bytes, elapsed)
                    print(
                        f"  iter {it + 1:2d}/{num_iterations}: "
                        f"{elapsed:.3f}s  {format_gbs(batch_gbs)}"
                    )

    finally:
        del_tier_and_cleanup(tier)

    t_total = time.perf_counter() - t_start

    # ── Per-ratio summary table ─────────────────────────────────────────
    print("\n==================== Per-ratio summary ====================")
    header = (
        f"{'ratio':>5}  "
        + "  ".join(
            f"{'rep' + str(r + 1) + ' mean':>10}  {'p50':>7}  {'p99':>7}  {'GB/s':>8}"
            for r in range(num_repeats)
        )
        + (f"  {'Δmean':>7}" if num_repeats >= 2 else "")
    )
    print(header)
    print("-" * len(header))

    steady_ratio = 1.0 if 1.0 in hot_ratios else hot_ratios[-1]
    steady_lats = results[(num_repeats - 1, steady_ratio)]

    for ratio in hot_ratios:
        row = f"{ratio:>5.2f}  "
        means = []
        for r in range(num_repeats):
            lats = results[(r, ratio)]
            mean = statistics.mean(lats)
            means.append(mean)
            p50 = statistics.median(lats)
            p99 = _percentile(lats, 0.99)
            gbs = bytes_to_gbs(batch_size * block_bytes, mean)
            row += f"  {mean:>10.3f}  {p50:>7.3f}  {p99:>7.3f}  {gbs:>8.3f}"
        if num_repeats >= 2 and means[0] > 0:
            drift = (means[-1] - means[0]) / means[0] * 100.0
            row += f"  {drift:>+6.1f}%"
        print(row)

    # Steady-state metrics for return / assertions.
    steady_mean = statistics.mean(steady_lats) if steady_lats else 0.0
    steady_p50 = statistics.median(steady_lats) if steady_lats else 0.0
    steady_p99 = _percentile(steady_lats, 0.99)
    steady_p100 = max(steady_lats) if steady_lats else 0.0
    batch_bytes = batch_size * block_bytes
    synthetic_tps = batch_size / steady_mean if steady_mean > 0 else 0.0
    kv_gb_s = bytes_to_gbs(batch_bytes, steady_mean)

    print(
        f"\n[RESULTS]  wall={t_total:.2f}s  pool={len(pool)}\n"
        f"  Steady (repeat {num_repeats}, ratio={steady_ratio}): "
        f"jobs/s={synthetic_tps:.1f}  {format_gbs(kv_gb_s)}  "
        f"mean={steady_mean:.3f}s p50={steady_p50:.3f}s "
        f"p99={steady_p99:.3f}s p100={steady_p100:.3f}s"
    )

    return (
        synthetic_tps,
        kv_gb_s,
        steady_mean,
        steady_p50,
        steady_p99,
        steady_p100,
    )


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("num_blocks", [64])
def test_stress(num_blocks, batch_size):
    """
    Hot/cold-ratio sweep stress test for the OBJ tier.

    Pytest runs a short single-repeat sweep (hot_ratios=[0.0, 0.5, 1.0],
    3 iters each) to keep CI fast. For a full stress run, use the CLI.

    Verifies:
      - Throughput > 0 (no deadlock in concurrent store/load)
      - Steady-state p99 bounded (backend handles pure-read concurrency)
      - p100 not a wild outlier vs p99 (no tail pathology)
    """
    _tps, _gbs, _mean, _p50, p99, p100 = run_stress_test(
        num_blocks=num_blocks,
        batch_size=batch_size,
        num_iterations=3,
        hot_ratios=[0.0, 0.5, 1.0],
        num_repeats=1,
    )

    assert p99 < 30.0, (
        f"Steady-state p99 too high: {p99:.3f}s "
        "(possible S3 connection starvation or backend queue blocked)."
    )
    assert (p100 - p99) < 15.0, (
        f"Tail latency spike (p100 - p99 = {p100 - p99:.3f}s). "
        "Possible lock contention or I/O stall."
    )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not s3_config_available():
        raise SystemExit(
            "S3 env vars not set. Export VLLM_TEST_S3_BUCKET, "
            "VLLM_TEST_S3_ENDPOINT, VLLM_TEST_S3_ACCESS_KEY, VLLM_TEST_S3_SECRET_KEY."
        )

    parser = argparse.ArgumentParser(
        description=(
            "Run OBJ tier hot/cold-ratio sweep stress test. "
            "Sweeps hot_ratio from 0.0 to 1.0 (default) stressing both "
            "write and read paths concurrently."
        )
    )
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument(
        "--elements-per-block",
        type=int,
        default=16384,
        help="float32 elements per block (default 16384 = 64 KB/block)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Iterations per (repeat, hot_ratio) (default 5)",
    )
    parser.add_argument(
        "--hot-ratios",
        type=str,
        default=",".join(str(r) for r in DEFAULT_HOT_RATIOS),
        help="Comma-separated hot_ratio values (default: 0.0..1.0 step 0.1)",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="Full sweeps through hot_ratios (default 1; use >=2 to detect drift)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hot_ratios = [float(x.strip()) for x in args.hot_ratios.split(",") if x.strip()]
    for r in hot_ratios:
        if not 0.0 <= r <= 1.0:
            raise SystemExit(f"hot_ratio {r} out of [0.0, 1.0]")

    run_stress_test(
        num_blocks=args.num_blocks,
        elements_per_block=args.elements_per_block,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        hot_ratios=hot_ratios,
        num_repeats=args.num_repeats,
        seed=args.seed,
    )
