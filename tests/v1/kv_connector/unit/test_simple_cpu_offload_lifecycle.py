# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle and scheduling tests for SimpleCPUOffloadConnector internals."""

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.manager import (
    RequestState,
    SimpleCPUOffloadScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload.worker import (
    SimpleCPUOffloadWorker,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import create_request


def _build_scheduler_output(
    num_scheduled_tokens: dict[str, int],
    finished_req_ids: set[str] | None = None,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids or set(),
        free_encoder_mm_hashes=[],
        kv_connector_metadata=SimpleCPUOffloadMetadata(),
    )


def _create_scheduler_manager(
    block_size: int = 16,
    cpu_capacity_gb: float = 1.0,
    num_gpu_blocks: int = 64,
    lazy_mode: bool = False,
) -> SimpleCPUOffloadScheduler:
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
        ),
        scheduler_config=SimpleNamespace(
            max_num_scheduled_tokens=256,
        ),
    )

    num_kv_heads = 8
    head_size = 64
    dtype = torch.float16
    page_size_bytes = 2 * block_size * num_kv_heads * head_size * 2

    kv_cache_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[KVCacheTensor(size=page_size_bytes, shared_by=["layer"])],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            )
        ],
    )
    manager = SimpleCPUOffloadScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=int(cpu_capacity_gb * (1024**3)),
        lazy_offload=lazy_mode,
    )
    return manager


class _MockBlock:
    """Mock GPU block with block_hash attribute for skipping logic."""

    def __init__(self, block_hash=None):
        self.block_hash = block_hash


class _MockKVCacheBlocks:
    """Mock KVCacheBlocks with blocks attribute for update_state_after_alloc."""

    def __init__(self, block_ids: list[int], num_computed_blocks: int = 0):
        self._block_ids = block_ids
        # First num_computed_blocks have block_hash set (already computed)
        self.blocks = [
            [
                _MockBlock(block_hash="computed" if i < num_computed_blocks else None)
                for i in range(len(block_ids))
            ]
        ]

    def get_block_ids(self):
        return [self._block_ids]


def _create_gpu_block_pool(num_blocks: int = 64) -> BlockPool:
    return BlockPool(num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=16)


# ============================================================
# Eager store tests
# ============================================================


def test_eager_store_tracks_request_state():
    """update_state_after_alloc creates RequestState in _reqs_to_store."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=64, block_size=16)
    req_id = request.request_id

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([0, 1, 2, 3]),
        num_external_tokens=0,
    )

    assert req_id in manager._reqs_to_store
    state = manager._reqs_to_store[req_id]
    assert isinstance(state, RequestState)
    assert state.gpu_block_ids == [[0, 1, 2, 3]]
    assert state.num_stored_blocks == 0
    assert req_id not in manager._reqs_to_load


def test_eager_store_accumulates_across_steps():
    """num_stored_blocks advances across scheduler steps."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=64, block_size=16)
    req_id = request.request_id

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([0, 1, 2, 3]),
        num_external_tokens=0,
    )

    request.num_computed_tokens = 0
    manager.build_connector_meta(_build_scheduler_output({req_id: 16}))
    assert manager._reqs_to_store[req_id].num_stored_blocks == 1

    request.num_computed_tokens = 16
    manager.build_connector_meta(_build_scheduler_output({req_id: 16}))
    assert manager._reqs_to_store[req_id].num_stored_blocks == 2


def test_cached_blocks_advance_store_cursor():
    """Blocks already in CPU cache should advance cursor without re-storing."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    # Pre-cache the first block hash
    cached_block = manager.cpu_block_pool.get_new_blocks(1)[0]
    cached_block.block_hash = make_block_hash_with_group_id(
        request.block_hashes[0], group_id=0
    )
    manager.cpu_block_pool.cached_block_hash_to_block.insert(
        cached_block.block_hash, cached_block
    )
    manager.cpu_block_pool.free_blocks([cached_block])

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([0, 1]),
        num_external_tokens=0,
    )
    request.num_computed_tokens = 0

    meta = manager.build_connector_meta(_build_scheduler_output({req_id: 32}))
    # Block 0 already cached, only block 1 should be stored
    assert meta.store_gpu_blocks == [1]
    assert meta.store_event >= 0
    assert manager._reqs_to_store[req_id].num_stored_blocks == 2

    # Repeating should not re-issue any store
    meta2 = manager.build_connector_meta(_build_scheduler_output({req_id: 32}))
    assert meta2.store_event == -1
    assert meta2.store_gpu_blocks == []


def test_store_touches_gpu_blocks_to_prevent_freeing():
    """GPU blocks should have ref_cnt incremented during store."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    gpu_block_pool = _create_gpu_block_pool()
    manager.bind_gpu_block_pool(gpu_block_pool)

    gpu_blocks = gpu_block_pool.get_new_blocks(2)
    gpu_block_ids = [b.block_id for b in gpu_blocks]
    assert all(gpu_block_pool.blocks[bid].ref_cnt == 1 for bid in gpu_block_ids)

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks(gpu_block_ids),
        num_external_tokens=0,
    )
    request.num_computed_tokens = 0

    meta = manager.build_connector_meta(_build_scheduler_output({req_id: 32}))
    assert meta.store_event >= 0
    # GPU blocks touched: ref_cnt 1 -> 2
    for bid in meta.store_gpu_blocks:
        assert gpu_block_pool.blocks[bid].ref_cnt == 2


def test_store_completion_caches_and_releases_refs():
    """Store completion should cache CPU blocks and release GPU refs."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    gpu_block_pool = _create_gpu_block_pool()
    manager.bind_gpu_block_pool(gpu_block_pool)

    gpu_blocks = gpu_block_pool.get_new_blocks(2)
    gpu_block_ids = [b.block_id for b in gpu_blocks]

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks(gpu_block_ids),
        num_external_tokens=0,
    )
    request.num_computed_tokens = 0

    meta = manager.build_connector_meta(_build_scheduler_output({req_id: 32}))
    stored_gpu_ids = list(meta.store_gpu_blocks)
    assert all(gpu_block_pool.blocks[bid].ref_cnt == 2 for bid in stored_gpu_ids)

    manager.update_connector_output(
        KVConnectorOutput(
            finished_sending={f"__store_done_{meta.store_event}"},
            finished_recving=None,
            invalid_block_ids=set(),
        )
    )

    # GPU ref_cnt decremented back to 1
    for bid in stored_gpu_ids:
        assert gpu_block_pool.blocks[bid].ref_cnt == 1

    # Blocks should be cached in CPU pool
    for bh in request.block_hashes[:2]:
        cached = manager.cpu_block_pool.get_cached_block(bh, kv_cache_group_ids=[0])
        assert cached is not None


def test_preemption_with_inflight_store():
    """Preemption: kv_cache_manager frees (ref_cnt 2->1),
    store completion frees connector ref (1->0)."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    gpu_block_pool = _create_gpu_block_pool()
    manager.bind_gpu_block_pool(gpu_block_pool)

    gpu_blocks = gpu_block_pool.get_new_blocks(2)
    gpu_block_ids = [b.block_id for b in gpu_blocks]

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks(gpu_block_ids),
        num_external_tokens=0,
    )
    request.num_computed_tokens = 0

    meta = manager.build_connector_meta(_build_scheduler_output({req_id: 32}))
    stored_ids = list(meta.store_gpu_blocks)
    assert all(gpu_block_pool.blocks[bid].ref_cnt == 2 for bid in stored_ids)

    # Preemption: kv_cache_manager.free() decrements ref_cnt 2 -> 1
    gpu_block_pool.free_blocks(gpu_blocks)
    assert all(gpu_block_pool.blocks[bid].ref_cnt == 1 for bid in stored_ids)
    free_count_before = gpu_block_pool.get_num_free_blocks()

    # Store completes: connector frees ref_cnt 1 -> 0
    manager.update_connector_output(
        KVConnectorOutput(
            finished_sending={f"__store_done_{meta.store_event}"},
            finished_recving=None,
            invalid_block_ids=set(),
        )
    )

    for bid in stored_ids:
        assert gpu_block_pool.blocks[bid].ref_cnt == 0
    assert gpu_block_pool.get_num_free_blocks() > free_count_before


def test_multistep_gpu_block_accumulation():
    """GPU blocks from multiple steps accumulate and are all freed."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=64, block_size=16)
    req_id = request.request_id

    gpu_block_pool = _create_gpu_block_pool()
    manager.bind_gpu_block_pool(gpu_block_pool)

    gpu_blocks = gpu_block_pool.get_new_blocks(4)
    gpu_block_ids = [b.block_id for b in gpu_blocks]

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks(gpu_block_ids),
        num_external_tokens=0,
    )

    request.num_computed_tokens = 0
    meta1 = manager.build_connector_meta(_build_scheduler_output({req_id: 16}))
    step1_gpu_ids = list(meta1.store_gpu_blocks)
    assert len(step1_gpu_ids) > 0

    request.num_computed_tokens = 16
    meta2 = manager.build_connector_meta(_build_scheduler_output({req_id: 16}))
    step2_gpu_ids = list(meta2.store_gpu_blocks)

    all_gpu_ids = step1_gpu_ids + step2_gpu_ids
    assert len(all_gpu_ids) > len(step1_gpu_ids)
    for bid in all_gpu_ids:
        assert gpu_block_pool.blocks[bid].ref_cnt == 2

    # Complete both store jobs
    manager.update_connector_output(
        KVConnectorOutput(
            finished_sending={
                f"__store_done_{meta1.store_event}",
                f"__store_done_{meta2.store_event}",
            },
            finished_recving=None,
            invalid_block_ids=set(),
        )
    )

    for bid in all_gpu_ids:
        assert gpu_block_pool.blocks[bid].ref_cnt == 1


# ============================================================
# Load tests
# ============================================================


def test_load_touch_refs_are_released_on_finished_recving():
    """CPU (and GPU) touch refs should be released when load completes."""
    manager = _create_scheduler_manager()
    gpu_block_pool = _create_gpu_block_pool()
    manager.bind_gpu_block_pool(gpu_block_pool)

    request = create_request(num_tokens=16, block_size=16)
    req_id = request.request_id

    # Cache a CPU block
    cpu_block = manager.cpu_block_pool.get_new_blocks(1)[0]
    cpu_block.block_hash = make_block_hash_with_group_id(
        request.block_hashes[0], group_id=0
    )
    manager.cpu_block_pool.cached_block_hash_to_block.insert(
        cpu_block.block_hash, cpu_block
    )
    manager.cpu_block_pool.free_blocks([cpu_block])
    assert cpu_block.ref_cnt == 0

    # Allocate a GPU block for the load target
    gpu_block = gpu_block_pool.get_new_blocks(1)[0]
    gpu_block_id = gpu_block.block_id
    assert gpu_block_pool.blocks[gpu_block_id].ref_cnt == 1

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([gpu_block_id]),
        num_external_tokens=16,
    )
    # CPU block touched: ref_cnt 0 -> 1
    assert cpu_block.ref_cnt == 1
    # GPU block touched: ref_cnt 1 -> 2
    assert gpu_block_pool.blocks[gpu_block_id].ref_cnt == 2
    assert req_id in manager._reqs_to_load

    # Build connector meta to assign load_event
    meta = manager.build_connector_meta(_build_scheduler_output({}))
    assert meta.load_event >= 0

    # Simulate load completion
    output = KVConnectorOutput(
        finished_sending=None,
        finished_recving={req_id},
        invalid_block_ids=set(),
    )
    manager.update_connector_output(output)
    assert cpu_block.ref_cnt == 0
    assert gpu_block_pool.blocks[gpu_block_id].ref_cnt == 1
    assert req_id not in manager._reqs_to_load


def test_request_finished_releases_inflight_load_refs():
    """request_finished should release CPU/GPU touch refs for loads."""
    manager = _create_scheduler_manager()
    gpu_block_pool = _create_gpu_block_pool()
    manager.bind_gpu_block_pool(gpu_block_pool)

    request = create_request(num_tokens=16, block_size=16)
    req_id = request.request_id

    # Cache a CPU block
    cpu_block = manager.cpu_block_pool.get_new_blocks(1)[0]
    cpu_block.block_hash = make_block_hash_with_group_id(
        request.block_hashes[0], group_id=0
    )
    manager.cpu_block_pool.cached_block_hash_to_block.insert(
        cpu_block.block_hash, cpu_block
    )
    manager.cpu_block_pool.free_blocks([cpu_block])

    gpu_block = gpu_block_pool.get_new_blocks(1)[0]

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([gpu_block.block_id]),
        num_external_tokens=16,
    )
    assert cpu_block.ref_cnt == 1
    assert gpu_block_pool.blocks[gpu_block.block_id].ref_cnt == 2

    # Load not yet submitted (no load_event), so request_finished cleans up directly
    is_async, _ = manager.request_finished(request, block_ids=[gpu_block.block_id])
    assert not is_async
    assert cpu_block.ref_cnt == 0
    assert gpu_block_pool.blocks[gpu_block.block_id].ref_cnt == 1
    assert req_id not in manager._reqs_to_load


def test_load_skips_locally_computed_blocks():
    """External load should skip blocks already computed on GPU."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=64, block_size=16)
    req_id = request.request_id

    # Cache all 4 blocks in CPU
    for bh in request.block_hashes[:4]:
        cpu_block = manager.cpu_block_pool.get_new_blocks(1)[0]
        cpu_block.block_hash = make_block_hash_with_group_id(bh, group_id=0)
        manager.cpu_block_pool.cached_block_hash_to_block.insert(
            cpu_block.block_hash, cpu_block
        )
        manager.cpu_block_pool.free_blocks([cpu_block])

    # GPU already has blocks 0,1 computed (32 tokens)
    num_local_computed = 32
    num_new, is_async = manager.get_num_new_matched_tokens(request, num_local_computed)
    assert num_new == 32  # blocks 2,3 from CPU
    assert is_async is True

    # GPU block layout: [10, 11, 12, 13] for blocks 0-3
    # Blocks 0,1 have block_hash set (num_computed_blocks=2)
    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([10, 11, 12, 13], num_computed_blocks=2),
        num_external_tokens=32,
    )

    # Should load into GPU blocks 12, 13 (not 10, 11)
    state = manager._reqs_to_load[req_id]
    assert state.load_transfer is not None
    assert state.load_transfer.gpu_block_ids == [12, 13]
    assert len(state.load_transfer.cpu_block_ids) == 2


# ============================================================
# Finished flag / deferred cleanup tests
# ============================================================


def test_finished_flag_defers_store_cleanup():
    """request_finished sets finished=True when store is in-flight,
    cleanup happens on store completion."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    gpu_block_pool = _create_gpu_block_pool()
    manager.bind_gpu_block_pool(gpu_block_pool)

    gpu_blocks = gpu_block_pool.get_new_blocks(2)
    gpu_block_ids = [b.block_id for b in gpu_blocks]

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks(gpu_block_ids),
        num_external_tokens=0,
    )
    request.num_computed_tokens = 0

    meta = manager.build_connector_meta(_build_scheduler_output({req_id: 32}))
    assert meta.store_event >= 0

    # Request finishes while store is in-flight
    is_async, _ = manager.request_finished(request, block_ids=gpu_block_ids)
    assert not is_async
    # State should still be in _reqs_to_store with finished=True
    assert req_id in manager._reqs_to_store
    assert manager._reqs_to_store[req_id].finished is True

    # Store completes -> deferred cleanup triggers
    manager.update_connector_output(
        KVConnectorOutput(
            finished_sending={f"__store_done_{meta.store_event}"},
            finished_recving=None,
            invalid_block_ids=set(),
        )
    )
    # Now request should be cleaned up
    assert req_id not in manager._reqs_to_store


def test_finished_flag_defers_load_cleanup():
    """request_finished sets finished=True when load is in-flight,
    cleanup happens on load completion."""
    manager = _create_scheduler_manager()
    gpu_block_pool = _create_gpu_block_pool()
    manager.bind_gpu_block_pool(gpu_block_pool)

    request = create_request(num_tokens=16, block_size=16)
    req_id = request.request_id

    cpu_block = manager.cpu_block_pool.get_new_blocks(1)[0]
    cpu_block.block_hash = make_block_hash_with_group_id(
        request.block_hashes[0], group_id=0
    )
    manager.cpu_block_pool.cached_block_hash_to_block.insert(
        cpu_block.block_hash, cpu_block
    )
    manager.cpu_block_pool.free_blocks([cpu_block])

    gpu_block = gpu_block_pool.get_new_blocks(1)[0]
    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([gpu_block.block_id]),
        num_external_tokens=16,
    )

    # Assign load_event (simulates build_connector_meta)
    meta = manager.build_connector_meta(_build_scheduler_output({}))
    assert meta.load_event >= 0

    # Request finishes while load is in-flight
    is_async, _ = manager.request_finished(request, block_ids=[gpu_block.block_id])
    assert not is_async
    # State should still be in _reqs_to_load with finished=True
    assert req_id in manager._reqs_to_load
    assert manager._reqs_to_load[req_id].finished is True

    # Load completes -> cleanup triggers
    output = KVConnectorOutput(
        finished_sending=None,
        finished_recving={req_id},
        invalid_block_ids=set(),
    )
    manager.update_connector_output(output)
    assert req_id not in manager._reqs_to_load
    assert cpu_block.ref_cnt == 0
    assert gpu_block_pool.blocks[gpu_block.block_id].ref_cnt == 1


def test_request_finished_no_inflight_cleans_immediately():
    """request_finished cleans up immediately when nothing is in-flight."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([0, 1]),
        num_external_tokens=0,
    )
    assert req_id in manager._reqs_to_store

    is_async, _ = manager.request_finished(request, block_ids=[0, 1])
    assert not is_async
    assert req_id not in manager._reqs_to_store


def test_request_finished_returns_false_with_inflight_stores():
    """request_finished() always returns False; ref_cnt protects blocks."""
    manager = _create_scheduler_manager()
    request = create_request(num_tokens=32, block_size=16)
    req_id = request.request_id

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([0, 1]),
        num_external_tokens=0,
    )
    request.num_computed_tokens = 0

    # Create a store job so there's an in-flight transfer
    manager.build_connector_meta(_build_scheduler_output({req_id: 32}))

    is_async, params = manager.request_finished(request, block_ids=[0, 1])
    assert is_async is False
    assert params is None
    # Store state deferred (finished=True)
    assert req_id in manager._reqs_to_store


# ============================================================
# Lazy store tests
# ============================================================


def test_lazy_store_touches_and_releases_gpu_blocks():
    """Lazy mode: GPU eviction candidates are touched during store,
    freed on completion."""
    manager = _create_scheduler_manager(lazy_mode=True)

    # Small pool: 1 null + 4 usable
    gpu_block_pool = _create_gpu_block_pool(num_blocks=5)
    manager.bind_gpu_block_pool(gpu_block_pool)

    all_blocks = gpu_block_pool.get_new_blocks(4)
    hashed_blocks = all_blocks[:2]
    unhashed_blocks = all_blocks[2:]
    for i, block in enumerate(hashed_blocks):
        block.block_hash = make_block_hash_with_group_id(
            BlockHash(b"hash" + str(i).encode()), group_id=0
        )
        gpu_block_pool.cached_block_hash_to_block.insert(block.block_hash, block)

    gpu_block_pool.free_blocks(hashed_blocks)
    gpu_block_pool.free_blocks(unhashed_blocks)
    gpu_block_ids = [b.block_id for b in hashed_blocks]
    assert all(gpu_block_pool.blocks[bid].ref_cnt == 0 for bid in gpu_block_ids)

    meta = manager.build_connector_meta(_build_scheduler_output({"some_req": 32}))
    assert meta.store_event >= 0, "Expected a lazy store job"

    for bid in meta.store_gpu_blocks:
        assert gpu_block_pool.blocks[bid].ref_cnt > 0

    manager.update_connector_output(
        KVConnectorOutput(
            finished_sending={f"__store_done_{meta.store_event}"},
            finished_recving=None,
            invalid_block_ids=set(),
        )
    )

    for bid in meta.store_gpu_blocks:
        assert gpu_block_pool.blocks[bid].ref_cnt == 0


def test_lazy_mode_does_not_track_reqs_to_store():
    """Lazy mode should not populate _reqs_to_store."""
    manager = _create_scheduler_manager(lazy_mode=True)
    request = create_request(num_tokens=32, block_size=16)

    manager.update_state_after_alloc(
        request=request,
        blocks=_MockKVCacheBlocks([0, 1]),
        num_external_tokens=0,
    )
    assert request.request_id not in manager._reqs_to_store


# ============================================================
# Worker tests
# ============================================================


def test_worker_wait_for_save_queues_store_jobs():
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    worker = SimpleCPUOffloadWorker(
        vllm_config=vllm_config,
        kv_cache_config=None,
        cpu_capacity_bytes=1024 * 1024,
    )

    worker.bind_connector_metadata(
        SimpleCPUOffloadMetadata(
            store_event=0,
            store_gpu_blocks=[1, 2],
            store_cpu_blocks=[3, 4],
        )
    )

    with patch.object(
        type(worker), "_is_initialized", new_callable=PropertyMock, return_value=True
    ):
        worker.wait_for_save()

    assert len(worker._pending_store_jobs) == 1
    job_idx, src, dst = worker._pending_store_jobs[0]
    assert job_idx == 0
    assert src == [1, 2]
    assert dst == [3, 4]
    assert not worker._store_events


def test_worker_start_load_submits_pending_stores_without_metadata():
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    worker = SimpleCPUOffloadWorker(
        vllm_config=vllm_config,
        kv_cache_config=None,
        cpu_capacity_bytes=1024 * 1024,
    )
    worker.clear_connector_metadata()

    called = {"count": 0}

    def _fake_submit():
        called["count"] += 1

    worker._submit_pending_stores = _fake_submit  # type: ignore[method-assign]
    with patch.object(
        type(worker), "_is_initialized", new_callable=PropertyMock, return_value=True
    ):
        worker.start_load_kv()

    assert called["count"] == 1


def test_connector_emits_finished_sending_immediately_on_store_completion():
    """finished_sending emits job-index sentinels as soon as stores complete."""
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
        SimpleCPUOffloadConnector,
    )

    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    connector.scheduler_manager = None
    connector._connector_metadata = None
    connector._pending_load_job_indices = set()
    connector._pending_store_job_indices = {0}

    mock_worker = MagicMock()
    mock_worker.get_finished.return_value = ({"__store_done_0"}, None)
    connector.worker_handler = mock_worker

    finished_sending, finished_recving = connector.get_finished(set())
    assert finished_sending == {"__store_done_0"}
    assert finished_recving is None
