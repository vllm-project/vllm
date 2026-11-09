# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import (
    MemoryRuntime,
    _EmbeddedRuntime,
    _hybrid_kv_cache_config,
    _kv_cache_config,
    _SchedulerHandle,
    _vllm_config,
    _WorkerHandle,
    make_request,
    make_scheduler_output,
    mori_has_umbp,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.umbp import (
    scheduler as umbp_scheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    BlockTransferPlan,
    KVLayoutPlanner,
    LoadSpec,
    RankTopology,
    RequestTracker,
    StoreEventResult,
    TransferJobState,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    EmbeddedRuntime,
    UMBPRuntimeConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.factory import (
    UMBPRuntimeFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SparseCacheRole,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

mori_available = importlib.util.find_spec("mori") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and mori_available),
    reason="UMBP is only available on ROCm with the mori package installed",
)


def test_mamba_align_states_store_only_from_boundary_handoffs():
    """Positional Mamba blocks may be live or speculative; only core's
    boundary hand-offs name a committed state block."""
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        _hybrid_kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("mamba-align")),
    )
    request = make_request(
        "m",
        num_tokens=33,
        block_ids=([3, 4, 5], [6, 7, 8]),
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([3, 4, 5], [6, 7, 8])),
        0,
    )

    def output(offloads):
        return make_scheduler_output(
            new_reqs=[
                SimpleNamespace(
                    req_id="m", num_computed_tokens=0, block_ids=([3, 4, 5], [6, 7, 8])
                )
            ],
            scheduled_tokens={"m": 33},
            block_state=SimpleNamespace(boundary_state_offloads=offloads),
        )

    positional = scheduler.build_connector_meta(output({}))
    assert {
        (plan.group_id, plan.block_id) for plan in positional.iter_store_plans()
    } == {
        (0, 3),
        (0, 4),
    }

    scheduler._request_trackers["m"].saved_tokens = 0
    handed_off = scheduler.build_connector_meta(output({"m": [(1, 9, 32)]}))
    handed_off_plans = list(handed_off.iter_store_plans())
    assert (1, 9) in {(plan.group_id, plan.block_id) for plan in handed_off_plans}
    assert not any(
        plan.group_id == 1 and plan.block_id in (6, 7, 8) for plan in handed_off_plans
    )


def test_restored_hybrid_prefix_rechecks_only_positional_groups():
    """A restored request's residency check must skip boundary-only groups."""
    handle = _SchedulerHandle([])
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        _hybrid_kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("mamba-restore")),
    )
    request = make_request(
        "r",
        num_tokens=48,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([3, 4, 5], [6, 7, 8]),
    )
    tracker = scheduler._tracker_for_request("r")
    tracker.load_spec = LoadSpec(0, 32)

    plans = scheduler._store_plans(
        request, tracker, 48, block_ids_override=([3, 4, 5], [6, 7, 8])
    )

    assert {plan.group_id for plan in plans} <= {0}
    assert all(":g1:" not in key for query in handle.queries for key in query)


def test_store_plans_store_each_groups_complete_blocks():
    config = _kv_cache_config()
    config.kv_cache_groups.append(
        KVCacheGroupSpec(
            ["other"], replace(config.kv_cache_groups[0].kv_cache_spec, block_size=32)
        )
    )
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        config,
        _SchedulerHandle({}),
        BlockIdentityCodec(UMBPNamespace("group-blocks")),
    )
    request = make_request(
        "r",
        num_tokens=48,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([1, 2, 3], [4, 5]),
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids), 0
    )
    tracker = scheduler._request_trackers["r"]

    # The third 16-token block is complete although the second 32-token one
    # is not; it must not wait for the larger group.
    plans = scheduler._store_plans(request, tracker, 48)
    assert sorted((plan.group_id, plan.block_id) for plan in plans) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
    ]

    request.block_hashes.append(b"d")
    request.block_ids[0].append(6)
    plans = scheduler._store_plans(request, tracker, 64)
    assert sorted((plan.group_id, plan.block_id) for plan in plans) == [(0, 6), (1, 5)]


def test_store_plans_apply_core_reachable_block_mask(monkeypatch):
    config = _kv_cache_config()
    full = config.kv_cache_groups[0].kv_cache_spec
    sliding = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=full.num_kv_heads,
        head_size=full.head_size,
        dtype=full.dtype,
        sliding_window=32,
    )
    config.kv_cache_groups.append(KVCacheGroupSpec(["sliding"], sliding))
    original = KVCacheSpecRegistry.get_manager_class

    class SparseStoreManager:
        @classmethod
        def reachable_block_mask(cls, **kwargs):
            assert kwargs["start_block"] == 0
            assert kwargs["end_block"] == 4
            assert kwargs["alignment_tokens"] == 16
            assert kwargs["reachable_boundaries"] == (63,)
            return [False, True, False, True]

    def manager_for(spec, role=None):
        return (
            SparseStoreManager
            if isinstance(spec, SlidingWindowSpec)
            else original(spec, role)
        )

    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}),
        config,
        _SchedulerHandle({}),
        BlockIdentityCodec(UMBPNamespace("store-mask")),
    )
    monkeypatch.setattr(KVCacheSpecRegistry, "get_manager_class", manager_for)
    request = make_request(
        "masked",
        num_tokens=64,
        num_prompt_tokens=64,
        block_hashes=[b"a", b"b", b"c", b"d"],
        block_ids=([1, 2, 3, 4], [5, 6, 7, 8]),
    )

    plans = scheduler._store_plans(request, RequestTracker(), 64)

    assert [plan.block_id for plan in plans if plan.group_id == 0] == [1, 2, 3, 4]
    assert [plan.block_id for plan in plans if plan.group_id == 1] == [6, 8]


@pytest.mark.parametrize("use_eagle", [False, True])
def test_single_sliding_window_stores_a_replayable_tail(use_eagle):
    config = _kv_cache_config()
    full = config.kv_cache_groups[0].kv_cache_spec
    config.kv_cache_groups[0].kv_cache_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=full.num_kv_heads,
        head_size=full.head_size,
        dtype=full.dtype,
        sliding_window=32,
    )
    config.prefix_cache_retention_interval = 0
    vllm_config = _vllm_config({})
    vllm_config.speculative_config = SimpleNamespace(
        use_eagle_block_drop=lambda: use_eagle
    )
    codec = BlockIdentityCodec(UMBPNamespace("swa-replay"))
    handle = _SchedulerHandle({})
    scheduler = UMBPStoreConnectorScheduler(vllm_config, config, handle, codec)
    request = make_request(
        "swa",
        num_tokens=65,
        block_hashes=[b"a", b"b", b"c", b"d"],
        block_ids=([1, 2, 3, 4],),
    )
    try:
        plans = scheduler._store_plans(request, RequestTracker(), 64)
        assert plans.block_ids == ([2, 3, 4] if use_eagle else [3, 4])
        handle.hits = dict.fromkeys(plans.keys, True)
        matched, async_load = scheduler.get_num_new_matched_tokens(request, 0)
        assert (matched, async_load) == (48 if use_eagle else 64, True)
        destinations = ([0, 5, 6] if use_eagle else [0, 0, 5, 6],)
        scheduler.update_state_after_alloc(
            request,
            SimpleNamespace(get_block_ids=lambda group_ids: destinations),
            matched,
        )
        loaded = scheduler._pending_loads[request.request_id]
        assert loaded.block_ids == [5, 6]
        expected_hashes = [b"b", b"c"] if use_eagle else [b"c", b"d"]
        assert loaded.keys == [codec.key(block_hash) for block_hash in expected_hashes]
    finally:
        scheduler.close()


@pytest.mark.parametrize("lookup_async", [False, True])
@pytest.mark.parametrize("chunk_object_resident", [False, True])
def test_external_chunked_local_hit_preserves_absolute_positions(
    chunk_object_resident, lookup_async
):
    """A chunked-local hit is claimed only through resident absolute chunks.

    With a 16-token local prefix, claiming token 80 needs the group-1 object
    for tokens 64-79. When it is missing, the safe boundary is the token-64
    chunk start, so 48 external tokens are claimed and nothing loads it.
    """
    config = _kv_cache_config()
    full = config.kv_cache_groups[0].kv_cache_spec
    config.kv_cache_groups.append(
        KVCacheGroupSpec(
            ["chunked"],
            ChunkedLocalAttentionSpec(
                block_size=16,
                num_kv_heads=full.num_kv_heads,
                head_size=full.head_size,
                dtype=full.dtype,
                attention_chunk_size=64,
            ),
        )
    )
    codec = BlockIdentityCodec(UMBPNamespace("absolute-chunk"))
    hashes = [b"a", b"b", b"c", b"d", b"e"]
    hits = {
        codec.key(block_hash, group_id): True
        for group_id in (0, 1)
        for block_hash in hashes[1:]
    }
    if not chunk_object_resident:
        del hits[codec.key(b"e", 1)]
    handle = _SchedulerHandle(hits)
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"lookup_async": lookup_async}), config, handle, codec
    )
    pool = BlockPool(8, True, 16)
    for group_id in (0, 1):
        identity = make_block_hash_with_group_id(hashes[0], group_id)
        pool.cached_block_hash_to_block.insert(identity, pool.blocks[group_id + 1])
    scheduler.bind_gpu_block_pool(pool)
    request = make_request(
        "chunked",
        num_tokens=81,
        block_hashes=hashes,
        block_ids=([1, 2, 3, 4, 5], [0, 0, 0, 0, 6]),
    )
    external_tokens = 64 if chunk_object_resident else 48
    try:
        result = scheduler.get_num_new_matched_tokens(request, 16)
        deadline = time.monotonic() + 5
        while result[0] is None and time.monotonic() < deadline:
            time.sleep(0.001)
            result = scheduler.get_num_new_matched_tokens(request, 16)
        assert result == (external_tokens, True)
        scheduler.update_state_after_alloc(
            request,
            SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids),
            external_tokens,
        )
        plans = scheduler._pending_loads[request.request_id]
        expected = [(0, 2), (0, 3), (0, 4)]
        if chunk_object_resident:
            expected += [(0, 5), (1, 6)]
        assert [(plan.group_id, plan.block_id) for plan in plans] == expected
        loaded_keys = [plan.key for plan in plans]
        assert (codec.key(b"e", 1) in loaded_keys) == chunk_object_resident
    finally:
        scheduler.close()


def test_eagle_lookup_includes_the_aligned_prompt_proof_block():
    config = _kv_cache_config()
    full = config.kv_cache_groups[0].kv_cache_spec
    config.kv_cache_groups.append(
        KVCacheGroupSpec(
            ["sliding"],
            SlidingWindowSpec(
                block_size=16,
                num_kv_heads=full.num_kv_heads,
                head_size=full.head_size,
                dtype=full.dtype,
                sliding_window=32,
            ),
            is_eagle_group=True,
        )
    )
    config.prefix_cache_retention_interval = 0
    vllm_config = _vllm_config({})
    vllm_config.speculative_config = SimpleNamespace(use_eagle_block_drop=lambda: True)
    codec = BlockIdentityCodec(UMBPNamespace("eagle-proof"))
    handle = _SchedulerHandle({})
    scheduler = UMBPStoreConnectorScheduler(vllm_config, config, handle, codec)
    request = make_request(
        "eagle",
        num_tokens=64,
        block_hashes=[b"a", b"b", b"c", b"d"],
        block_ids=([1, 2, 3, 4], [5, 6, 7, 8]),
    )
    try:
        plans = scheduler._store_plans(request, RequestTracker(), 64)
        handle.hits = dict.fromkeys(plans.keys, True)
        assert scheduler.get_num_new_matched_tokens(request, 0) == (48, True)
        assert codec.key(b"d", 1) in handle.queries[-1]
        scheduler.update_state_after_alloc(
            request,
            SimpleNamespace(
                get_block_ids=lambda group_ids: (
                    [1, 2, 3],
                    [0, 5, 6],
                )
            ),
            48,
        )
        loaded = scheduler._pending_loads[request.request_id]
        assert loaded.keys == [
            codec.key(b"a", 0),
            codec.key(b"b", 0),
            codec.key(b"c", 0),
            codec.key(b"b", 1),
            codec.key(b"c", 1),
        ]
    finally:
        scheduler.close()


def test_chunk_boundary_hit_without_transfers_is_synchronous():
    config = _kv_cache_config()
    full = config.kv_cache_groups[0].kv_cache_spec
    config.kv_cache_groups[0].kv_cache_spec = ChunkedLocalAttentionSpec(
        block_size=16,
        num_kv_heads=full.num_kv_heads,
        head_size=full.head_size,
        dtype=full.dtype,
        attention_chunk_size=64,
    )
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({}),
        config,
        _SchedulerHandle({}),
        BlockIdentityCodec(UMBPNamespace("empty-chunk")),
    )
    request = make_request(
        "chunk",
        num_tokens=65,
        block_hashes=[b"a", b"b", b"c", b"d"],
    )
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (64, False)
        scheduler.update_state_after_alloc(
            request,
            SimpleNamespace(get_block_ids=lambda group_ids: ([0, 0, 0, 0],)),
            64,
        )
        assert not scheduler._pending_loads[request.request_id]
    finally:
        scheduler.close()


@pytest.mark.parametrize("external_checkpoint", [False, True])
@pytest.mark.parametrize("lookup_async", [False, True])
def test_hybrid_external_hit_keeps_local_sparse_checkpoint(
    external_checkpoint,
    lookup_async,
):
    config = _hybrid_kv_cache_config()
    config.kv_cache_groups.insert(
        1,
        KVCacheGroupSpec(
            ["untransferred"],
            config.kv_cache_groups[0].kv_cache_spec,
            enable_kv_transfer=False,
        ),
    )
    codec = BlockIdentityCodec(UMBPNamespace("local-checkpoint"))
    handle = _SchedulerHandle(
        {
            codec.key(b"c", 0): True,
            codec.key(b"d", 0): True,
            codec.key(b"d", 2): external_checkpoint,
        }
    )
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"lookup_async": lookup_async}),
        config,
        handle,
        codec,
    )
    pool = BlockPool(8, True, 16)
    for block_id, (group_id, block_hash) in enumerate(
        [(0, b"a"), (0, b"b"), (2, b"b")],
        start=1,
    ):
        identity = make_block_hash_with_group_id(block_hash, group_id)
        pool.cached_block_hash_to_block.insert(identity, pool.blocks[block_id])
    scheduler.bind_local_cache_lookup(
        lambda group_id, block_hash: bool(pool.get_cached_block(block_hash, [group_id]))
    )
    request = make_request(
        "hybrid",
        num_tokens=65,
        block_hashes=[b"a", b"b", b"c", b"d"],
    )
    try:
        result = scheduler.get_num_new_matched_tokens(request, 32)
        deadline = time.monotonic() + 5
        while result[0] is None and time.monotonic() < deadline:
            time.sleep(0.001)
            result = scheduler.get_num_new_matched_tokens(request, 32)
        assert result == ((32, True) if external_checkpoint else (0, False))
        if external_checkpoint:
            scheduler.update_state_after_alloc(
                request,
                SimpleNamespace(
                    get_block_ids=lambda group_ids: (
                        [1, 2, 3, 4],
                        [0, 0, 0, 5],
                    )
                ),
                32,
            )
            plans = scheduler._pending_loads[request.request_id]
            assert [(plan.group_id, plan.block_id) for plan in plans] == [
                (0, 3),
                (0, 4),
                (2, 5),
            ]
    finally:
        scheduler.close()


def test_draft_groups_must_be_restorable(monkeypatch):
    monkeypatch.setitem(
        UMBPRuntimeFactory._builders, "embedded", lambda config: _EmbeddedRuntime()
    )
    config = _kv_cache_config()
    config.kv_cache_groups.append(
        KVCacheGroupSpec(
            ["draft"],
            config.kv_cache_groups[0].kv_cache_spec,
            is_eagle_group=True,
            enable_kv_transfer=False,
        )
    )
    with pytest.raises(NotImplementedError, match="draft-model KV"):
        UMBPStoreConnector(
            _vllm_config({"mode": "embedded"}), KVConnectorRole.SCHEDULER, config
        )


def test_hybrid_model_loads_asynchronously_even_when_sync_is_requested():
    codec = BlockIdentityCodec(UMBPNamespace("hybrid-sync"))
    block_hashes = [b"a", b"b"]
    hits = {
        codec.key(block_hashes[0], 0): True,
        codec.key(block_hashes[1], 0): True,
        codec.key(block_hashes[1], 1): True,
    }
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        _hybrid_kv_cache_config(),
        _SchedulerHandle(hits),
        codec,
    )
    try:
        request = make_request("hybrid-sync", block_hashes=block_hashes, num_tokens=33)

        assert scheduler.get_num_new_matched_tokens(request, 0) == (32, True)
    finally:
        scheduler.close()


@pytest.mark.parametrize("lookup_async", [False, True])
def test_hybrid_lookup_uses_latest_mamba_checkpoint(lookup_async, monkeypatch):
    codec = BlockIdentityCodec(UMBPNamespace("hybrid-hit-window"))
    block_hashes = [b"a", b"b"]
    hits = {
        codec.key(block_hashes[0], 0): True,
        codec.key(block_hashes[1], 0): True,
        codec.key(block_hashes[1], 1): True,
    }
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lookup_async": lookup_async}),
        _hybrid_kv_cache_config(),
        _SchedulerHandle(hits),
        codec,
    )
    try:
        pool = scheduler._lookup_coordinator.block_pool
        for operation in (
            "get_new_blocks",
            "_insert_block_hash",
            "touch",
            "free_blocks",
        ):
            monkeypatch.setattr(
                pool,
                operation,
                lambda *a, **kw: pytest.fail("lookup must be read-only"),
            )
        request = make_request(
            "hybrid-lookup", block_hashes=block_hashes, num_tokens=33
        )

        matched, is_async = scheduler.get_num_new_matched_tokens(request, 0)
        deadline = time.monotonic() + 5
        while matched is None and time.monotonic() < deadline:
            time.sleep(0.001)
            matched, is_async = scheduler.get_num_new_matched_tokens(request, 0)

        assert (matched, is_async) == (32, True)
        spec = scheduler._load_specs[request.request_id]
        assert spec.block_hashes_by_group == (
            (block_hashes[0], block_hashes[1]),
            (None, block_hashes[1]),
        )

        tracker = RequestTracker()
        tracker.load_spec = spec
        plans = scheduler._load_plans_for_external_tokens(
            request,
            tracker,
            ([10, 11], [NULL_BLOCK_ID, 12]),
            matched,
        )
        assert [(plan.group_id, plan.block_id, plan.key) for plan in plans] == [
            (0, 10, codec.key(block_hashes[0], 0)),
            (0, 11, codec.key(block_hashes[1], 0)),
            (1, 12, codec.key(block_hashes[1], 1)),
        ]
    finally:
        scheduler.close()


@pytest.mark.parametrize("failure_stage", ["lookup", "match"])
def test_hybrid_shadow_lookup_cleans_up_after_failure(monkeypatch, failure_stage):
    """A failed query cannot leak shadow hits or consume pool capacity."""
    handle = _SchedulerHandle({})
    codec = BlockIdentityCodec(UMBPNamespace("shadow-cleanup"))
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({}), _hybrid_kv_cache_config(), handle, codec
    )
    request = make_request("r", num_tokens=33)
    handle.hits = {codec.key(h, g): True for h in request.block_hashes for g in (0, 1)}
    coordinator = scheduler._lookup_coordinator
    pool = coordinator.block_pool
    free_blocks = pool.get_num_free_blocks()

    def fail(*args, **kwargs):
        raise RuntimeError("injected shadow failure")

    try:
        with monkeypatch.context() as patch:
            if failure_stage == "lookup":
                lookup = pool.get_cached_block

                def fail_after_lookup(*args, **kwargs):
                    lookup(*args, **kwargs)
                    fail()

                patch.setattr(pool, "get_cached_block", fail_after_lookup)
            else:
                patch.setattr(coordinator, "find_longest_cache_hit", fail)
            with pytest.raises(RuntimeError, match="injected shadow failure"):
                scheduler.get_num_new_matched_tokens(request, 0)
        assert pool.get_num_free_blocks() == free_blocks
        assert len(pool.cached_block_hash_to_block) == 0
        assert not pool.cached_block_hashes_by_block
        assert pool.hits == []
        assert pool.hit_blocks == {}
        handle.hits = {}
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        handle.hits = {
            codec.key(h, g): True for h in request.block_hashes for g in (0, 1)
        }
        assert scheduler.get_num_new_matched_tokens(request, 0) == (32, True)
        assert pool.get_num_free_blocks() == free_blocks
        assert len(pool.cached_block_hash_to_block) == 0
    finally:
        scheduler.close()


@pytest.mark.parametrize("indexer", [False, True])
def test_sparse_attention_indexer_forces_async_load(indexer):
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        cache_role=SparseCacheRole.INDEXER if indexer else SparseCacheRole.SPARSE,
    )
    config = replace(
        _kv_cache_config(),
        kv_cache_groups=[KVCacheGroupSpec(["layer1", "layer2"], spec)],
    )
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        config,
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("indexer")),
    )

    # The indexer reads its restored cache before any wait_for_layer_load.
    try:
        assert scheduler.load_async is indexer
    finally:
        scheduler.close()


@pytest.mark.parametrize("boundary", [0, 3, 8, 11, 12, 16])
def test_scheduler_accepts_only_computed_hash_aligned_tail(boundary, monkeypatch):
    # Core hashes requests in 4-token units.
    monkeypatch.setattr(
        umbp_scheduler, "resolve_kv_cache_block_sizes", lambda *args: (16, 4)
    )
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("partial")),
    )
    request = make_request(
        "partial",
        num_tokens=12,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([7],),
        num_computed_tokens=12,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([7],)),
        0,
    )

    pool = BlockPool(8, True, 4)
    scheduler.bind_gpu_block_pool(pool)
    assert not scheduler.register_finished_partial_tail(
        request, ([7],), [(0, 7, boundary)]
    )
    metadata = scheduler.build_connector_meta(make_scheduler_output())

    assert len(metadata.store_plans) == (1 if boundary == 12 else 0)
    if metadata.store_plans:
        tail = metadata.store_plans[0]
        assert tail.block_id == 7
        assert tail.key == scheduler.codec.key(b"c", tail.group_id)


def test_scheduler_stores_exact_hybrid_boundary_state():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        _hybrid_kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("hybrid")),
    )
    request = make_request(
        "hybrid",
        block_ids=([1, 2], [8, 9]),
        num_computed_tokens=16,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2], [8, 9])),
        0,
    )
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(
            block_state=SimpleNamespace(
                boundary_state_offloads={
                    "hybrid": [
                        (1, 9, 16),
                        (1, NULL_BLOCK_ID, 16),
                    ]
                }
            ),
        )
    )

    assert len(metadata.store_plans) == 1
    plan = metadata.store_plans[0]
    assert plan.group_id == 1
    assert plan.block_id == 9
    assert plan.key == scheduler.codec.key(b"a", 1)


@pytest.mark.parametrize("local_tokens", [0, 16])
def test_scheduler_partial_prefix_loads_full_page_at_absolute_index(
    local_tokens, monkeypatch
):
    # Core hashes requests in 4-token units.
    monkeypatch.setattr(
        umbp_scheduler, "resolve_kv_cache_block_sizes", lambda *args: (16, 4)
    )
    config = _kv_cache_config()
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "mode": "embedded",
                "load_async": False,
                "enable_partial_hash_hits": True,
            }
        ),
        config,
        _SchedulerHandle(
            [False, False, True] + [False] * (4 if local_tokens == 0 else 0)
        ),
        BlockIdentityCodec(UMBPNamespace("partial-prefix")),
    )
    request = make_request(
        "partial-prefix",
        num_tokens=32,
        block_hashes=[b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h"],
    )

    assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (12, False)
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(
            get_block_ids=lambda group_ids: ([6, 7] if local_tokens else [7],)
        ),
        12,
    )
    plans = scheduler._pending_loads["partial-prefix"]

    assert len(plans) == 1
    assert plans[0].block_id == 7


def test_hybrid_fine_hash_lookup_is_not_limited_by_gpu_page_count():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {"load_async": False, "enable_partial_hash_hits": True},
            prefix_match_unit=4,
        ),
        _hybrid_kv_cache_config(),
        _SchedulerHandle([True] * 62),
        BlockIdentityCodec(UMBPNamespace("many-hashes")),
    )
    try:
        request = make_request(
            "many-hashes",
            num_tokens=128,
            block_hashes=[bytes([index]) for index in range(32)],
        )
        for _ in range(2):
            # Mamba-style groups always load asynchronously.
            assert scheduler.get_num_new_matched_tokens(request, 0) == (124, True)
    finally:
        scheduler.close()


def test_finished_hybrid_tail_pins_full_pages_until_all_workers_finish():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {"mode": "embedded", "load_async": False},
            prefix_match_unit=4,
            world_size=2,
        ),
        _hybrid_kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("multi-partial")),
    )
    request = make_request(
        "multi-partial",
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([7], [9]),
        num_computed_tokens=12,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([7], [9])),
        0,
    )

    pool = BlockPool(16, True, 4)
    scheduler.bind_gpu_block_pool(pool)
    blocks = [pool.blocks[7], pool.blocks[9]]
    pool.touch(blocks)
    assert not scheduler.register_finished_partial_tail(
        request,
        ([7], [9]),
        [(1, 9, 12)],
    )
    assert [block.ref_cnt for block in blocks] == [2, 2]
    assert scheduler.request_finished(request, ([7], [9])) == (False, None)
    pool.free_blocks(blocks)
    assert [block.ref_cnt for block in blocks] == [1, 1]
    assert scheduler.has_pending_push_work()
    metadata = scheduler.build_connector_meta(
        make_scheduler_output(finished={request.request_id})
    )

    assert len(metadata.store_plans) == 2
    assert {(plan.group_id, plan.block_id) for plan in metadata.store_plans} == {
        (0, 7),
        (1, 9),
    }
    assert [block.ref_cnt for block in blocks] == [1, 1]
    for rank in range(2):
        scheduler.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                    store_events={metadata.store_event: StoreEventResult(1)}
                )
            )
        )
        assert [block.ref_cnt for block in blocks] == ([1, 1] if rank == 0 else [0, 0])
    assert not scheduler.has_pending_push_work()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not mori_has_umbp(), reason="MORI was built without UMBP"
            ),
        ),
    ],
)
def test_finished_hybrid_tail_restores_every_byte(socket_dir, device):
    """A hash-only tail survives request cleanup, then restores byte for byte."""
    if device == "cpu":
        runtime = MemoryRuntime()
    else:
        runtime = EmbeddedRuntime.from_config(
            UMBPRuntimeConfig(
                "embedded", {"capacity_bytes": 16 << 20, "lookup_dir": socket_dir}
            )
        )
    config = _hybrid_kv_cache_config()
    groups = config.kv_cache_groups
    config = replace(
        config,
        kv_cache_tensors=[
            KVCacheTensor(
                size=8 * group.kv_cache_spec.page_size_bytes,
                layers=group.layer_names,
                layer_stride=8 * group.kv_cache_spec.page_size_bytes,
                block_stride=group.kv_cache_spec.page_size_bytes,
            )
            for group in groups
        ],
    )
    topology = RankTopology()
    layout = KVLayoutPlanner.from_kv_cache_config(config)
    descriptor = layout.describe(topology)
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {"enable_partial_hash_hits": True, "load_async": True},
            prefix_match_unit=4,
        ),
        config,
        runtime.create_scheduler_handle("tail", topology, descriptor),
        BlockIdentityCodec(UMBPNamespace("tail")),
    )
    worker = UMBPStoreConnectorWorker(
        runtime.create_worker_handle("tail", topology, descriptor), layout
    )
    pool = BlockPool(8, True, 4)
    scheduler.bind_gpu_block_pool(pool)
    caches = {
        group.layer_names[0]: torch.zeros(
            (8, group.kv_cache_spec.page_size_bytes), dtype=torch.uint8, device=device
        )
        for group in groups
    }
    # Distinct int32 words, so a K/V swap or a shifted range cannot match.
    pages = {
        name: torch.arange(
            group << 16,
            (group << 16) + cache.shape[1] // 4,
            dtype=torch.int32,
            device=device,
        ).view(torch.uint8)
        for group, (name, cache) in enumerate(caches.items())
    }
    for block_id, (name, cache) in enumerate(caches.items(), 1):
        cache[block_id] = pages[name]
    worker.register_kv_caches(caches)
    request = make_request(
        "tail",
        num_tokens=13,
        num_computed_tokens=12,
        block_hashes=[b"a", b"b", b"c"],
    )
    source_ids = ([1], [2])
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: source_ids), 0
    )
    sources = [pool.blocks[1], pool.blocks[2]]
    pool.touch(sources)
    assert not scheduler.register_finished_partial_tail(
        request, source_ids, [(1, 2, 12)]
    )
    assert scheduler.request_finished(request, source_ids) == (False, None)
    pool.free_blocks(sources)
    output = make_scheduler_output(finished={"tail"})
    try:
        metadata = scheduler.build_connector_meta(output)
        assert len(metadata.store_plans) == 2
        worker.enqueue_stores(metadata)
        worker.wait_for_save()
        deadline = time.monotonic() + 10
        while scheduler.has_pending_push_work():
            assert time.monotonic() < deadline, "store completion was not reported"
            worker.get_finished(set())
            scheduler.update_connector_output(
                SimpleNamespace(
                    kv_connector_worker_meta=worker.build_connector_worker_meta()
                )
            )
            time.sleep(0.001)
        assert [block.ref_cnt for block in sources] == [0, 0]
        for cache in caches.values():
            cache.zero_()

        request.request_id = "replay"
        assert scheduler.get_num_new_matched_tokens(request, 0) == (12, True)
        scheduler.update_state_after_alloc(
            request, SimpleNamespace(get_block_ids=lambda group_ids: ([3], [4])), 12
        )
        output.finished_req_ids = set()
        worker.start_load_kv(None, scheduler.build_connector_meta(output))
        deadline = time.monotonic() + 10
        while (finished := worker.get_finished(set())) == (None, None):
            assert time.monotonic() < deadline, "load completion was not reported"
            time.sleep(0.001)
        assert finished == (None, {"replay"})
        assert worker.get_failed_recving() == set()
        for block_id, (name, cache) in enumerate(caches.items(), 3):
            expected = torch.zeros_like(cache)
            expected[block_id] = pages[name]
            assert torch.equal(cache, expected), name
        counters = worker.get_kv_connector_stats().reduce()
        page_bytes = sum(group.kv_cache_spec.page_size_bytes for group in groups)
        assert counters["load_num_bytes"] == page_bytes
        assert counters["store_num_bytes"] == page_bytes
    finally:
        worker.close()
        scheduler.close()


@pytest.mark.parametrize("failure", [None, "missing-rank", "exception", "short"])
@pytest.mark.parametrize("restored_tokens", [32, 64])
@pytest.mark.parametrize("null_block", [False, True])
def test_eager_restored_prefix_rechecks_residency_per_group_and_rank(
    failure, restored_tokens, null_block, monkeypatch
):
    """Only currently complete objects may skip store; fresh suffixes still store."""
    config = _kv_cache_config()
    config.kv_cache_groups.append(
        KVCacheGroupSpec(
            ["other"], replace(config.kv_cache_groups[0].kv_cache_spec, block_size=32)
        )
    )
    codec = BlockIdentityCodec(UMBPNamespace("eager-residency"))
    topology = RankTopology(tp_size=2)
    handle = _SchedulerHandle({})
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded"}, tensor_parallel_size=2, world_size=2),
        config,
        handle,
        codec,
        topology,
    )
    request = make_request(
        "req",
        num_tokens=65,
        block_hashes=[b"a", b"b", b"c", b"d"],
        block_ids=([1, 2, 3, 4], [NULL_BLOCK_ID if null_block else 5, 6]),
        num_computed_tokens=63,
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids), 0
    )
    tracker = scheduler._request_trackers["req"]
    # The group hit windows can be sparse; a scalar watermark cannot deduplicate.
    tracker.load_spec = LoadSpec(0, restored_tokens, ((None, b"b"), (b"b",)))
    keys = [
        key
        for group, block_size in scheduler.group_block_sizes.items()
        for end in range(block_size, restored_tokens + 1, block_size)
        if request.block_ids[group][end // block_size - 1] != NULL_BLOCK_ID
        for block_hash in (request.block_hashes[end // 16 - 1],)
        for key in codec.keys_for_topology(block_hash, topology, (group,))
    ]
    handle.hits = dict.fromkeys(keys, True)
    if failure == "missing-rank":
        handle.hits[keys[1]] = False
    elif failure == "short":
        handle.hits = [True]
    elif failure == "exception":

        def unavailable(keys):
            raise TimeoutError("lookup unavailable")

        monkeypatch.setattr(handle, "lookup", unavailable)
    output = make_scheduler_output(new_reqs=[request], scheduled_tokens={"req": 1})
    created = []

    def record_plan(**kwargs):
        plan = BlockTransferPlan(**kwargs)
        created.append(plan)
        return plan

    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler.BlockTransferPlan",
        record_plan,
    )
    try:
        meta = scheduler.build_connector_meta(output)
        expected = {(0, b"c"), (0, b"d"), (1, b"d")} if restored_tokens == 32 else set()
        if failure == "missing-rank":
            expected.add((0, b"a"))
        elif failure is not None:
            expected = {(0, h) for h in request.block_hashes} | {(1, b"b"), (1, b"d")}
        if null_block:
            expected.discard((1, b"b"))
        assert {p.key for p in meta.iter_store_plans()} == {
            scheduler.codec.key(h, g) for g, h in expected
        }
        assert len(meta.store_requests.get("req", [])) == len(expected)
        # The scheduler keeps eager stores columnar; plans are materialized
        # only by compatibility iteration or a runtime fallback.
        assert created == []
    finally:
        scheduler.close()


def test_several_cache_groups_load_asynchronously():
    config = _kv_cache_config()
    config.kv_cache_groups.append(
        KVCacheGroupSpec(["other"], config.kv_cache_groups[0].kv_cache_spec)
    )
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "load_async": False}),
        config,
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("groups-async")),
    )

    assert scheduler.load_async


@pytest.mark.parametrize("several_groups", [False, True])
def test_several_cache_groups_report_failed_loads_per_request(several_groups):
    """Core maps block-level load failures to requests only with one group."""

    class _FailingLoadHandle(_WorkerHandle):
        def load(self, plans):
            job = TransferJobState(tuple(plans))
            job.start()
            job.fail([plan.key for plan in plans], "evicted")
            return job

    worker = UMBPStoreConnectorWorker(
        _FailingLoadHandle(), report_failed_requests=several_groups
    )
    plan = BlockTransferPlan("key", 3, request_id="req", group_id=0)
    worker.start_load_kv(
        None, UMBPConnectorMetadata(async_load=True, load_requests={"req": [plan]})
    )

    assert worker.get_finished(set()) == (None, {"req"})
    assert worker.get_failed_recving() == ({"req"} if several_groups else set())
    assert worker.get_block_ids_with_load_errors() == (set() if several_groups else {3})
