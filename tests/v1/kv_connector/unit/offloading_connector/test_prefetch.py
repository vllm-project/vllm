# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-free prefetch: submission outcomes, reservation, completion."""

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
    PrefetchCompletionStatus,
    PrefetchSubmitOutcome,
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
GEOMETRIES = [1, 4]  # GPU blocks per offloaded chunk


def _kv_cache_config(num_groups: int = 1) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=256,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [f"layer{g}"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=4,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
            for g in range(num_groups)
        ],
    )


def _make_scheduler(
    *,
    enabled: bool = True,
    reserve_blocks: int = 0,
    num_gpu_blocks: int = 64,
    enable_caching: bool = True,
    blocks_per_chunk: int = 1,
    num_groups: int = 1,
) -> tuple[OffloadingConnectorScheduler, MockOffloadingSpec, BlockPool]:
    extra_config = {
        "enable_request_free_prefetch": enabled,
        "prefetch_reserve_blocks": reserve_blocks,
        "blocks_per_chunk": blocks_per_chunk,
    }
    vllm_config = _make_vllm_config(extra_config=extra_config)
    vllm_config.cache_config.enable_prefix_caching = enable_caching
    vllm_config.speculative_config = None
    kv_cache_config = _kv_cache_config(num_groups)
    spec = MockOffloadingSpec(build_offloading_config(vllm_config, kv_cache_config))
    sched = OffloadingConnectorScheduler(spec, vllm_config, kv_cache_config)
    pool = BlockPool(
        num_gpu_blocks, enable_caching=enable_caching, hash_block_size=BLOCK_SIZE
    )
    sched.bind_gpu_block_pool(pool)
    return sched, spec, pool


def _hashes(num_blocks: int, tag: str = "h") -> list[BlockHash]:
    return [BlockHash(f"{tag}{i}".encode().ljust(32, b"\0")) for i in range(num_blocks)]


def _complete(sched: OffloadingConnectorScheduler, job_id: int) -> None:
    meta = OffloadingWorkerMetadata()
    meta.mark_completed(job_id)
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))


def _put_on_gpu(pool: BlockPool, hashes: list[BlockHash]) -> None:
    """Make `hashes` resident in the GPU prefix cache, as a request would."""
    blocks = pool.get_new_blocks(len(hashes))
    pool.cache_prefetched_blocks(
        blocks, hashes, 0, BLOCK_SIZE, on_reuse=lambda block: None
    )


def _is_cached(pool: BlockPool, block_hash: BlockHash) -> bool:
    return pool.get_cached_block(block_hash, kv_cache_group_ids=[0]) is not None


@pytest.mark.parametrize("bpc", GEOMETRIES)
def test_fixture_geometry(bpc: int):
    sched, _, _ = _make_scheduler(blocks_per_chunk=bpc)
    group = sched.config.kv_group_configs[0]
    assert group.hashes_per_chunk == bpc
    assert group.tokens_per_chunk // group.tokens_per_block == bpc


# ---- UNSUPPORTED ------------------------------------------------------------


def test_unsupported_unless_enabled():
    sched, _, _ = _make_scheduler(enabled=False)
    result = sched.request_free_prefetch(_hashes(8))
    assert result.outcome is PrefetchSubmitOutcome.UNSUPPORTED
    assert result.prefetch_id is None


def test_unsupported_without_prefix_caching():
    sched, _, _ = _make_scheduler(enable_caching=False)
    result = sched.request_free_prefetch(_hashes(8))
    assert result.outcome is PrefetchSubmitOutcome.UNSUPPORTED


def test_unsupported_with_more_than_one_kv_cache_group():
    """A prefix-cache hit needs every group, so one group's load is not
    usable residency; the primitive says so instead of loading."""
    sched, spec, _ = _make_scheduler(num_groups=2)
    spec.manager.lookup.return_value = LookupResult.HIT
    result = sched.request_free_prefetch(_hashes(8))
    assert result.outcome is PrefetchSubmitOutcome.UNSUPPORTED
    assert not sched._current_batch_load_jobs
    spec.manager.lookup.assert_not_called()


# ---- MISSING vs ALREADY_AVAILABLE: nothing loadable is not "done" ------------


@pytest.mark.parametrize("bpc", GEOMETRIES)
def test_missing_when_nothing_is_offloaded(bpc: int):
    sched, spec, pool = _make_scheduler(blocks_per_chunk=bpc)
    spec.manager.lookup.return_value = LookupResult.MISS
    free_before = pool.get_num_free_blocks()

    result = sched.request_free_prefetch(_hashes(2 * bpc))

    assert result.outcome is PrefetchSubmitOutcome.MISSING
    assert result.num_blocks_requested == 2 * bpc
    assert result.num_blocks_available == 0
    assert result.num_blocks_unresolved == 2 * bpc
    assert pool.get_num_free_blocks() == free_before
    assert not sched._current_batch_load_jobs
    assert sched.take_prefetch_completions() == []


def test_missing_on_a_partial_chunk():
    """Fewer blocks than one offloaded chunk: never offloaded under this key
    scheme, so it cannot resolve - and it is not on the GPU either."""
    bpc = 4
    sched, spec, _ = _make_scheduler(blocks_per_chunk=bpc)
    spec.manager.lookup.return_value = LookupResult.HIT
    result = sched.request_free_prefetch(_hashes(bpc - 1))
    assert result.outcome is PrefetchSubmitOutcome.MISSING
    spec.manager.lookup.assert_not_called()


@pytest.mark.parametrize("bpc", GEOMETRIES)
def test_already_available_only_when_every_block_is_on_the_gpu(bpc: int):
    sched, spec, pool = _make_scheduler(blocks_per_chunk=bpc)
    hashes = _hashes(2 * bpc)
    _put_on_gpu(pool, hashes)

    result = sched.request_free_prefetch(hashes)

    assert result.outcome is PrefetchSubmitOutcome.ALREADY_AVAILABLE
    assert result.num_blocks_available == 2 * bpc
    assert result.num_blocks_unresolved == 0
    spec.manager.lookup.assert_not_called()
    assert not sched._current_batch_load_jobs


def test_empty_target_is_vacuously_available():
    sched, spec, _ = _make_scheduler()
    result = sched.request_free_prefetch([])
    assert result.outcome is PrefetchSubmitOutcome.ALREADY_AVAILABLE
    assert result.num_blocks_requested == 0


@pytest.mark.parametrize("bpc", GEOMETRIES)
def test_gpu_resident_prefix_with_nothing_offloaded_after_it_is_missing(bpc: int):
    """Partly on the GPU and the rest nowhere: not available, not loadable."""
    sched, spec, pool = _make_scheduler(blocks_per_chunk=bpc)
    hashes = _hashes(2 * bpc)
    _put_on_gpu(pool, hashes[:bpc])
    spec.manager.lookup.return_value = LookupResult.MISS

    result = sched.request_free_prefetch(hashes)

    assert result.outcome is PrefetchSubmitOutcome.MISSING
    assert result.num_blocks_available == bpc
    assert result.num_blocks_unresolved == bpc


# ---- DEFERRED: infeasible now, nothing mutated ----------------------------------


@pytest.mark.parametrize("bpc", GEOMETRIES)
def test_deferred_behind_the_reserve(bpc: int):
    num_blocks = 16
    sched, spec, pool = _make_scheduler(
        num_gpu_blocks=num_blocks, reserve_blocks=num_blocks - bpc, blocks_per_chunk=bpc
    )
    spec.manager.lookup.return_value = LookupResult.HIT
    free_before = pool.get_num_free_blocks()

    result = sched.request_free_prefetch(_hashes(2 * bpc))

    assert result.outcome is PrefetchSubmitOutcome.DEFERRED
    # Deferral is free: nothing reserved, nothing submitted, nothing evicted.
    assert pool.get_num_free_blocks() == free_before
    assert not sched._current_batch_load_jobs
    stats = sched.get_stats()
    assert stats is not None
    assert stats.reduce()[_ConnectorMetricName.PREFETCH_DEFERRED] == 1


@pytest.mark.parametrize("pending", [LookupResult.HIT_PENDING, LookupResult.RETRY])
def test_deferred_while_the_offloaded_tier_cannot_serve_yet(pending: LookupResult):
    """A store still in flight or a backend retry is not a missing target."""
    sched, spec, pool = _make_scheduler()
    spec.manager.lookup.return_value = pending
    free_before = pool.get_num_free_blocks()

    result = sched.request_free_prefetch(_hashes(4))

    assert result.outcome is PrefetchSubmitOutcome.DEFERRED
    assert pool.get_num_free_blocks() == free_before
    assert not sched._current_batch_load_jobs


def test_deferral_never_evicts_cached_blocks():
    """Free-but-cached blocks count as free; a prefetch that does not fit
    behind the reserve must still leave them in the prefix cache."""
    sched, spec, pool = _make_scheduler(num_gpu_blocks=8, reserve_blocks=4)
    resident = _hashes(4, tag="r")
    _put_on_gpu(pool, resident)
    spec.manager.lookup.return_value = LookupResult.HIT

    result = sched.request_free_prefetch(_hashes(6))

    assert result.outcome is PrefetchSubmitOutcome.DEFERRED
    assert all(_is_cached(pool, h) for h in resident)


def test_reservation_never_takes_a_referenced_block():
    """Blocks a request holds are untouchable; unreferenced cached blocks are
    reclaimed LRU-first, exactly as by a request's allocation (documented)."""
    sched, spec, pool = _make_scheduler(num_gpu_blocks=12)
    held = pool.get_new_blocks(4)  # a running request's blocks
    idle = _hashes(4, tag="idle")
    _put_on_gpu(pool, idle)  # unreferenced, still cached
    spec.manager.lookup.return_value = LookupResult.HIT

    result = sched.request_free_prefetch(_hashes(6))

    assert result.outcome is PrefetchSubmitOutcome.ACCEPTED
    ((_, job),) = sched._current_batch_load_jobs.items()
    assert {b.block_id for b in held}.isdisjoint(job.dst_spec.block_ids)
    assert all(b.ref_cnt == 1 for b in held)
    # 11 usable blocks (block 0 is the null block): 4 held, 7 free of which 4
    # idle-cached. Taking 6 reclaims at least 3 idle cached blocks.
    assert sum(_is_cached(pool, h) for h in idle) <= 1


# ---- ACCEPTED: reserved and submitted, not yet usable ---------------------------


@pytest.mark.parametrize("bpc", GEOMETRIES)
def test_accepted_reserves_blocks_and_submits_one_load(bpc: int):
    sched, spec, pool = _make_scheduler(blocks_per_chunk=bpc)
    spec.manager.lookup.return_value = LookupResult.HIT
    hashes = _hashes(2 * bpc)
    free_before = pool.get_num_free_blocks()

    result = sched.request_free_prefetch(hashes)

    assert result.outcome is PrefetchSubmitOutcome.ACCEPTED
    assert result.prefetch_id is not None
    assert result.num_blocks_to_load == 2 * bpc
    assert result.num_blocks_unresolved == 0
    # Destinations are held, unhashed: nothing can read them mid-transfer, and
    # accepting is not availability.
    assert pool.get_num_free_blocks() == free_before - 2 * bpc
    assert not any(_is_cached(pool, h) for h in hashes)
    assert sched.take_prefetch_completions() == []
    ((job_id, job),) = sched._current_batch_load_jobs.items()
    assert job.req_id == result.prefetch_id
    assert job.req_id not in sched._req_status  # no request attached
    assert job.dst_spec.block_indices == [0]
    assert list(job.dst_spec.group_sizes) == [2 * bpc]
    for block_id in job.dst_spec.block_ids:
        assert pool.blocks[block_id].block_hash is None


@pytest.mark.parametrize("bpc", GEOMETRIES)
def test_load_stops_at_the_first_missing_chunk(bpc: int):
    sched, spec, pool = _make_scheduler(blocks_per_chunk=bpc)
    spec.manager.lookup.side_effect = [LookupResult.HIT, LookupResult.MISS]
    free_before = pool.get_num_free_blocks()

    result = sched.request_free_prefetch(_hashes(3 * bpc))

    # A hole would not serve a prefix lookup: only the first chunk loads, and
    # the rest is reported unresolved rather than implied done.
    assert result.outcome is PrefetchSubmitOutcome.ACCEPTED
    assert result.num_blocks_to_load == bpc
    assert result.num_blocks_unresolved == 2 * bpc
    assert pool.get_num_free_blocks() == free_before - bpc


@pytest.mark.parametrize("bpc", GEOMETRIES)
def test_gpu_resident_leading_chunks_are_skipped(bpc: int):
    """Chunks already usable on the GPU are not loaded again."""
    sched, spec, pool = _make_scheduler(blocks_per_chunk=bpc)
    hashes = _hashes(3 * bpc)
    _put_on_gpu(pool, hashes[:bpc])
    spec.manager.lookup.return_value = LookupResult.HIT

    result = sched.request_free_prefetch(hashes)

    assert result.outcome is PrefetchSubmitOutcome.ACCEPTED
    assert result.num_blocks_available == bpc
    assert result.num_blocks_to_load == 2 * bpc
    ((_, job),) = sched._current_batch_load_jobs.items()
    # The load lands at logical block index bpc: the worker uses it to align
    # the offloaded chunk with the first destination block.
    assert job.dst_spec.block_indices == [bpc]
    assert spec.manager.lookup.call_count == 2


# ---- COMPLETED: only after publication ------------------------------------------


@pytest.mark.parametrize("bpc", GEOMETRIES)
def test_completion_is_reported_after_publication(bpc: int):
    sched, spec, pool = _make_scheduler(blocks_per_chunk=bpc)
    spec.manager.lookup.return_value = LookupResult.HIT
    hashes = _hashes(bpc)
    free_before = pool.get_num_free_blocks()

    result = sched.request_free_prefetch(hashes)
    job_id = next(iter(sched._current_batch_load_jobs))
    assert sched.take_prefetch_completions() == []
    _complete(sched, job_id)

    (completion,) = sched.take_prefetch_completions()
    assert completion.prefetch_id == result.prefetch_id
    assert completion.status is PrefetchCompletionStatus.COMPLETED
    assert completion.num_blocks_published == bpc
    # COMPLETED means usable residency: a later prefix lookup finds every block
    assert all(_is_cached(pool, h) for h in hashes)
    # ...and the blocks count as free again, as last-resort eviction candidates.
    assert pool.get_num_free_blocks() == free_before
    assert not sched._prefetch_jobs
    assert not sched._jobs
    spec.manager.complete_load.assert_called_once()
    # Each completion is reported once.
    assert sched.take_prefetch_completions() == []
    stats = sched.get_stats()
    assert stats is not None
    assert stats.reduce()[_ConnectorMetricName.PREFETCH_BLOCKS_PUBLISHED] == bpc


def test_block_cached_by_a_request_mid_flight_is_not_duplicated():
    sched, spec, pool = _make_scheduler()
    spec.manager.lookup.return_value = LookupResult.HIT
    hashes = _hashes(2)

    sched.request_free_prefetch(hashes)
    job_id = next(iter(sched._current_batch_load_jobs))
    # A request computes the first block's content while the load is running.
    _put_on_gpu(pool, hashes[:1])
    _complete(sched, job_id)

    (completion,) = sched.take_prefetch_completions()
    assert completion.status is PrefetchCompletionStatus.COMPLETED
    assert completion.num_blocks_published == 1
    assert completion.num_blocks_already_cached == 1
    key = make_block_hash_with_group_id(hashes[0], 0)
    entry = pool.cached_block_hash_to_block._cache[key]
    # One block per hash is stored bare; duplicates would turn it into a dict.
    assert not isinstance(entry, dict)


def test_reset_cache_cancels_in_flight_prefetches():
    sched, spec, pool = _make_scheduler()
    spec.manager.lookup.return_value = LookupResult.HIT
    hashes = _hashes(2)
    free_before = pool.get_num_free_blocks()

    result = sched.request_free_prefetch(hashes)
    sched._current_batch_load_jobs.clear()
    sched.reset_cache()

    (completion,) = sched.take_prefetch_completions()
    assert completion.prefetch_id == result.prefetch_id
    assert completion.status is PrefetchCompletionStatus.CANCELLED
    assert completion.num_blocks_published == 0
    assert pool.get_num_free_blocks() == free_before
    assert not sched._prefetch_jobs
    assert not any(_is_cached(pool, h) for h in hashes)


@pytest.mark.parametrize("num_chunks", [1, 3])
def test_prefetched_blocks_are_the_last_to_be_reused(num_chunks: int):
    """An unused prefetch must be the cheapest thing in the pool to lose."""
    sched, spec, pool = _make_scheduler(num_gpu_blocks=32)
    spec.manager.lookup.return_value = LookupResult.HIT
    hashes = _hashes(num_chunks)

    sched.request_free_prefetch(hashes)
    job_id = next(iter(sched._current_batch_load_jobs))
    _complete(sched, job_id)

    prefetched = {
        pool.get_cached_block(h, kv_cache_group_ids=[0])[0].block_id for h in hashes
    }
    # Every other free block is handed out before any prefetched one, and the
    # prefetched blocks are still readable meanwhile.
    others = pool.get_new_blocks(pool.get_num_free_blocks() - len(prefetched))
    assert prefetched.isdisjoint(block.block_id for block in others)
    assert all(_is_cached(pool, h) for h in hashes)

    # Only once nothing else is left do they go, leaving the prefix cache.
    last = pool.get_new_blocks(len(prefetched))
    assert {block.block_id for block in last} == prefetched
    assert not any(_is_cached(pool, h) for h in hashes)
