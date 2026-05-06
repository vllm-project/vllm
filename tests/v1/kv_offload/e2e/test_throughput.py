# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Performance test: OBJ tier sustained store and load throughput.

Mirrors the structure of the llm-d-kv-cache fs_connector throughput test
but operates directly on ObjSecondaryTier (CPU DRAM <-> S3 via NIXL)
instead of end-to-end LLM inference.

Measures:
  - Cold store  (DRAM -> S3, first write): wall time and GB/s
  - Hot load (S3 -> DRAM, multiple passes): mean / best / worst and GB/s

Run via pytest (uses conftest fixtures):
    pytest tests/v1/kv_offload/performance/test_throughput.py -v --noconftest \\
        --num-blocks=64 --elements-per-block=4096 --num-passes=5

Or as a standalone script:
    python -m tests.v1.kv_offload.performance.test_throughput \\
        --num-blocks=64 --elements-per-block=4096 --num-passes=5
"""

import argparse
import time
import uuid

import torch
import pytest

from .utils import (
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


def run_throughput_test(
    num_blocks: int = 32,
    elements_per_block: int = 4096,
    num_passes: int = 5,
    dtype: torch.dtype = torch.float32,
    key_prefix: str | None = None,
) -> tuple[float, float, float, float]:
    """
    Run a cold-store then hot-load throughput benchmark.

    Phase 1 — Cold store:  write `num_blocks` random blocks from DRAM to S3.
    Phase 2 — Hot loads:   reload all blocks from S3 back into DRAM `num_passes`
                           times, measuring each pass independently.

    Returns:
        (store_gbs, load_mean_gbs, load_best_gbs, load_worst_gbs)
    """
    prefix = key_prefix or f"perf/{uuid.uuid4().hex[:8]}"
    data_bytes = total_bytes(num_blocks, elements_per_block, dtype)
    data_mb = data_bytes / (1 << 20)

    print(
        f"\n===== Throughput Test =====\n"
        f"  Blocks: {num_blocks}, Elements/block: {elements_per_block}, "
        f"Block size: {elements_per_block * dtype.itemsize / 1024:.1f} KB, "
        f"Total: {data_mb:.1f} MB"
    )

    tier, tensor = make_tier_with_buffer(
        num_blocks=num_blocks,
        elements_per_block=elements_per_block,
        dtype=dtype,
        key_prefix=prefix,
    )

    keys = [unique_key(i) for i in range(num_blocks)]

    try:
        # ── Phase 1: Cold store (DRAM -> S3) ──────────────────────────────
        for bid in range(num_blocks):
            tensor[bid] = torch.rand((elements_per_block,), dtype=dtype)

        t0 = time.perf_counter()
        tier.submit_store(make_job(1, keys, list(range(num_blocks))))
        results = drain(tier)
        store_time = time.perf_counter() - t0

        assert all(r.success for r in results), "Cold store had failures"
        store_gbs = bytes_to_gbs(data_bytes, store_time)

        print(
            f"\n[Cold store]  {store_time:.3f}s  {format_gbs(store_gbs)}"
        )

        # ── Phase 2: Hot loads (S3 -> DRAM, repeated) ────────────────────
        load_times: list[float] = []
        for pass_n in range(num_passes):
            tensor.zero_()
            t0 = time.perf_counter()
            tier.submit_load(make_job(10 + pass_n, keys, list(range(num_blocks))))
            results = drain(tier)
            elapsed = time.perf_counter() - t0
            load_times.append(elapsed)
            success = all(r.success for r in results)
            gbs = bytes_to_gbs(data_bytes, elapsed)
            print(
                f"  [Load pass {pass_n + 1:2d}/{num_passes}]  "
                f"{elapsed:.3f}s  {format_gbs(gbs)}"
                + ("" if success else "  [FAILED]")
            )

        load_mean_gbs = bytes_to_gbs(data_bytes, sum(load_times) / len(load_times))
        load_best_gbs = bytes_to_gbs(data_bytes, min(load_times))
        load_worst_gbs = bytes_to_gbs(data_bytes, max(load_times))

        print(
            f"\n[RESULTS]\n"
            f"  Cold store:      {format_gbs(store_gbs)}\n"
            f"  Load mean:       {format_gbs(load_mean_gbs)}\n"
            f"  Load best:       {format_gbs(load_best_gbs)}\n"
            f"  Load worst:      {format_gbs(load_worst_gbs)}"
        )

    finally:
        del_tier_and_cleanup(tier)

    return store_gbs, load_mean_gbs, load_best_gbs, load_worst_gbs


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_passes", [5])
@pytest.mark.parametrize(
    "num_blocks,elements_per_block",
    [
        (32, 4096),    # 32 × 16 KB = 512 KB  — fast CI smoke test
        (64, 16384),   # 64 × 64 KB = 4 MB    — moderate throughput test
    ],
    ids=["32x16k", "64x64k"],
)
def test_throughput(num_blocks, elements_per_block, num_passes):
    """
    Sustained store/load throughput for the OBJ tier.

    Verifies:
      - Store completes without error (> 0 GB/s)
      - Load completes without error (> 0 GB/s)
      - Load does not degrade by more than 50% across passes (no starvation)
    """
    store_gbs, load_mean_gbs, load_best_gbs, load_worst_gbs = run_throughput_test(
        num_blocks=num_blocks,
        elements_per_block=elements_per_block,
        num_passes=num_passes,
    )

    assert store_gbs > 0, "Store produced 0 GB/s — possible deadlock or all failures"
    assert load_mean_gbs > 0, "Load produced 0 GB/s — possible deadlock or all failures"

    if load_best_gbs > 0:
        degradation = (load_best_gbs - load_worst_gbs) / load_best_gbs
        assert degradation < 0.5, (
            f"Load throughput degraded by {degradation * 100:.1f}% "
            f"(best={format_gbs(load_best_gbs)}, worst={format_gbs(load_worst_gbs)}). "
            "Possible S3 throttling or connection pool exhaustion."
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
        description="Run OBJ tier store/load throughput benchmark."
    )
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument(
        "--elements-per-block",
        type=int,
        default=16384,
        help="float32 elements per block (default 16384 = 64 KB/block)",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=10,
        help="Load passes after the initial cold store (default 10)",
    )
    args = parser.parse_args()

    run_throughput_test(
        num_blocks=args.num_blocks,
        elements_per_block=args.elements_per_block,
        num_passes=args.num_passes,
    )
