# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from unittest.mock import patch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.metrics.stats import KVCacheLifetimeStats


def test_demo_lifetime_stats_flow():
    """Sanity check mirroring the simple demo workflow."""

    stats = KVCacheLifetimeStats()

    stats.add_block_lifetime(2.5)
    stats.add_block_lifetime(3.5)
    stats.add_block_lifetime(4.0)

    assert stats.total_blocks_freed == 3
    assert math.isclose(stats.total_lifetime_seconds, 10.0, rel_tol=1e-6)
    assert math.isclose(stats.average_lifetime_seconds, 10.0 / 3, rel_tol=1e-6)

    stats.reset()
    assert stats.total_blocks_freed == 0
    assert stats.total_lifetime_seconds == 0.0
    assert stats.average_lifetime_seconds == 0.0


@patch("time.monotonic")
def test_demo_block_pool_tracking(mock_monotonic):
    """Replicate the demo's allocation/free lifecycle expectations."""

    mock_monotonic.side_effect = [100.0, 105.0]

    pool = BlockPool(
        num_gpu_blocks=5, enable_caching=False, enable_kv_cache_events=False
    )

    blocks = pool.get_new_blocks(2)
    assert all(block.allocation_time == 100.0 for block in blocks)

    pool.free_blocks(blocks)

    stats = pool.get_lifetime_stats()
    assert stats.total_blocks_freed == 2
    assert math.isclose(stats.total_lifetime_seconds, 10.0, rel_tol=1e-6)
    assert math.isclose(stats.average_lifetime_seconds, 5.0, rel_tol=1e-6)


def test_demo_prometheus_average_derivation():
    """Validate that Prometheus-style averages match the stored average."""

    stats = KVCacheLifetimeStats()
    for lifetime in (10.0, 20.0, 30.0):
        stats.add_block_lifetime(lifetime)

    derived_average = stats.total_lifetime_seconds / stats.total_blocks_freed
    assert math.isclose(derived_average, stats.average_lifetime_seconds, rel_tol=1e-6)
