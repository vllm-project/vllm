# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-free prefetch: reservation, deferral and publication."""

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.test_config import (
    _make_vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    _ConnectorMetricName,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.prefetch import (
    PrefetchOutcome,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.outputs import KVConnectorOutput

BLOCK_SIZE = 16
# One GPU block per offloaded chunk in this fixture's geometry; asserted in
# test_fixture_geometry so the expectations below stay honest if it changes.
BLOCKS_PER_CHUNK = 1


def _kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=256,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=4,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )


def _make_scheduler(
    *,
    enabled: bool = True,
    reserve_blocks: int = 0,
    num_gpu_blocks: int = 64,
    enable_caching: bool = True,
) -> tuple[OffloadingConnectorScheduler, MockOffloadingSpec, BlockPool]:
    extra_config = {
        "enable_request_free_prefetch": enabled,
        "prefetch_reserve_blocks": reserve_blocks,
    }
    vllm_config = _make_vllm_config(extra_config=extra_config)
    vllm_config.cache_config.enable_prefix_caching = enable_caching
    vllm_config.speculative_config = None
    kv_cache_config = _kv_cache_config()
    spec = MockOffloadingSpec(build_offloading_config(vllm_config, kv_cache_config))
    sched = OffloadingConnectorScheduler(spec, vllm_config, kv_cache_config)
    pool = BlockPool(
        num_gpu_blocks, enable_caching=enable_caching, hash_block_size=BLOCK_SIZE
    )
    sched.bind_gpu_block_pool(pool)
    return sched, spec, pool


def _hashes(num_blocks: int) -> list[BlockHash]:
    return [BlockHash(f"h{i}".encode().ljust(32, b"\0")) for i in range(num_blocks)]


def _complete(sched: OffloadingConnectorScheduler, job_id: int) -> None:
    meta = OffloadingWorkerMetadata()
    meta.mark_completed(job_id)
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))


def test_prefetch_is_unsupported_unless_enabled():
    sched, _, _ = _make_scheduler(enabled=False)
    assert sched.request_free_prefetch(_hashes(8)) is PrefetchOutcome.UNSUPPORTED


def test_prefetch_is_unsupported_without_prefix_caching():
    sched, _, _ = _make_scheduler(enable_caching=False)
    assert sched.request_free_prefetch(_hashes(8)) is PrefetchOutcome.UNSUPPORTED


def test_prefetch_completes_when_nothing_is_resident():
    sched, spec, pool = _make_scheduler()
    spec.manager.lookup.return_value = LookupResult.MISS
    free_before = pool.get_num_free_blocks()

    assert sched.request_free_prefetch(_hashes(8)) is PrefetchOutcome.COMPLETED

    assert pool.get_num_free_blocks() == free_before
    assert not sched._current_batch_load_jobs


def test_fixture_geometry():
    """The expectations below assume one GPU block per offloaded chunk."""
    sched, _, _ = _make_scheduler()
    group = sched.config.kv_group_configs[0]
    assert group.hashes_per_chunk == BLOCKS_PER_CHUNK
    assert group.tokens_per_chunk // group.tokens_per_block == BLOCKS_PER_CHUNK


def test_prefetch_completes_on_a_partial_chunk():
    sched, spec, _ = _make_scheduler()
    spec.manager.lookup.return_value = LookupResult.HIT
    # Fewer blocks than one chunk: no whole chunk was ever offloaded, so
    # there is nothing that could be resolved.
    assert (
        sched.request_free_prefetch(_hashes(BLOCKS_PER_CHUNK - 1))
        is PrefetchOutcome.COMPLETED
    )
    spec.manager.lookup.assert_not_called()


def test_prefetch_reserves_blocks_and_submits_a_load():
    sched, spec, pool = _make_scheduler()
    spec.manager.lookup.return_value = LookupResult.HIT
    hashes = _hashes(2 * BLOCKS_PER_CHUNK)
    free_before = pool.get_num_free_blocks()

    assert sched.request_free_prefetch(hashes) is PrefetchOutcome.ACCEPTED

    # Destinations are held, so nothing can read them mid-transfer.
    assert pool.get_num_free_blocks() == free_before - 2 * BLOCKS_PER_CHUNK
    assert len(sched._current_batch_load_jobs) == 1
    job_id, job = next(iter(sched._current_batch_load_jobs.items()))
    assert job.req_id.startswith("__kv_prefetch_")
    assert len(job.dst_spec.block_ids) == 2 * BLOCKS_PER_CHUNK
    # The load is not attributed to any request.
    assert job.req_id not in sched._req_status
    for block_id in job.dst_spec.block_ids:
        assert pool.blocks[block_id].block_hash is None


def test_prefetch_stops_at_the_first_missing_chunk():
    sched, spec, pool = _make_scheduler()
    spec.manager.lookup.side_effect = [LookupResult.HIT, LookupResult.MISS]
    free_before = pool.get_num_free_blocks()

    assert (
        sched.request_free_prefetch(_hashes(2 * BLOCKS_PER_CHUNK))
        is PrefetchOutcome.ACCEPTED
    )

    # Only the first chunk is loaded: a hole would not serve a prefix lookup.
    assert pool.get_num_free_blocks() == free_before - BLOCKS_PER_CHUNK


def test_prefetch_defers_behind_the_reserve():
    num_blocks = 16
    sched, spec, pool = _make_scheduler(
        num_gpu_blocks=num_blocks, reserve_blocks=num_blocks - 1
    )
    spec.manager.lookup.return_value = LookupResult.HIT
    free_before = pool.get_num_free_blocks()

    assert (
        sched.request_free_prefetch(_hashes(BLOCKS_PER_CHUNK))
        is PrefetchOutcome.DEFERRED
    )

    # Deferral is free: nothing reserved, nothing submitted, nothing evicted.
    assert pool.get_num_free_blocks() == free_before
    assert not sched._current_batch_load_jobs
    stats = sched.get_stats()
    assert stats is not None
    assert stats.reduce()[_ConnectorMetricName.PREFETCH_DEFERRED] == 1


def test_completed_prefetch_publishes_blocks_into_the_prefix_cache():
    sched, spec, pool = _make_scheduler()
    spec.manager.lookup.return_value = LookupResult.HIT
    hashes = _hashes(BLOCKS_PER_CHUNK)
    free_before = pool.get_num_free_blocks()

    assert sched.request_free_prefetch(hashes) is PrefetchOutcome.ACCEPTED
    job_id = next(iter(sched._current_batch_load_jobs))
    _complete(sched, job_id)

    # Published blocks are readable by a later request's prefix lookup...
    for i, block_hash in enumerate(hashes):
        cached = pool.get_cached_block(block_hash, kv_cache_group_ids=[0])
        assert cached is not None, i
    # ...and count as free again, as last-resort eviction candidates.
    assert pool.get_num_free_blocks() == free_before
    assert not sched._prefetch_jobs
    assert not sched._jobs
    spec.manager.complete_load.assert_called_once()


def test_reset_cache_drops_in_flight_prefetch_blocks():
    sched, spec, pool = _make_scheduler()
    spec.manager.lookup.return_value = LookupResult.HIT
    hashes = _hashes(BLOCKS_PER_CHUNK)
    free_before = pool.get_num_free_blocks()

    assert sched.request_free_prefetch(hashes) is PrefetchOutcome.ACCEPTED
    sched._current_batch_load_jobs.clear()
    sched.reset_cache()

    assert pool.get_num_free_blocks() == free_before
    assert not sched._prefetch_jobs
    for block_hash in hashes:
        assert (
            pool.cached_block_hash_to_block.get_one_block(
                make_block_hash_with_group_id(block_hash, 0)
            )
            is None
        )


@pytest.mark.parametrize("num_chunks", [1, 3])
def test_prefetched_blocks_are_the_last_to_be_reused(num_chunks: int):
    """An unused prefetch must be the cheapest thing in the pool to lose."""
    sched, spec, pool = _make_scheduler(num_gpu_blocks=32)
    spec.manager.lookup.return_value = LookupResult.HIT
    hashes = _hashes(num_chunks * BLOCKS_PER_CHUNK)

    assert sched.request_free_prefetch(hashes) is PrefetchOutcome.ACCEPTED
    job_id = next(iter(sched._current_batch_load_jobs))
    _complete(sched, job_id)

    prefetched = {
        pool.get_cached_block(h, kv_cache_group_ids=[0])[0].block_id for h in hashes
    }
    # Every other free block is handed out before any prefetched one, and the
    # prefetched blocks are still readable meanwhile.
    others = pool.get_new_blocks(pool.get_num_free_blocks() - len(prefetched))
    assert prefetched.isdisjoint(block.block_id for block in others)
    for block_hash in hashes:
        assert pool.get_cached_block(block_hash, kv_cache_group_ids=[0]) is not None

    # Only once nothing else is left do they go, leaving the prefix cache.
    last = pool.get_new_blocks(len(prefetched))
    assert {block.block_id for block in last} == prefetched
    for block_hash in hashes:
        assert pool.get_cached_block(block_hash, kv_cache_group_ids=[0]) is None
