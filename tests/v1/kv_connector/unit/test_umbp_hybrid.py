# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import (
    _EmbeddedRuntime,
    _hybrid_kv_cache_config,
    _kv_cache_config,
    _SchedulerHandle,
    _vllm_config,
    _WorkerHandle,
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
    LoadSpec,
    RankTopology,
    RequestTracker,
    StoreEventResult,
    TransferJobState,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
    UMBPNamespace,
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
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SparseCacheRole,
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
    request = SimpleNamespace(
        request_id="m",
        req_id="m",
        num_tokens=33,
        block_hashes=[b"a", b"b"],
        block_ids=([3, 4, 5], [6, 7, 8]),
        num_computed_tokens=0,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([3, 4, 5], [6, 7, 8])),
        0,
    )

    def output(offloads):
        return SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="m", num_computed_tokens=0, block_ids=([3, 4, 5], [6, 7, 8])
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={"m": 33},
            kv_connector_block_state=SimpleNamespace(boundary_state_offloads=offloads),
        )

    positional = scheduler.build_connector_meta(output({}))
    assert {(plan.group_id, plan.block_id) for plan in positional.store_plans} == {
        (0, 3),
        (0, 4),
    }

    scheduler._request_trackers["m"].saved_tokens = 0
    handed_off = scheduler.build_connector_meta(output({"m": [(1, 9, 32)]}))
    assert (1, 9) in {(plan.group_id, plan.block_id) for plan in handed_off.store_plans}
    assert not any(
        plan.group_id == 1 and plan.block_id in (6, 7, 8)
        for plan in handed_off.store_plans
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
    request = SimpleNamespace(
        request_id="r",
        req_id="r",
        num_tokens=48,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([3, 4, 5], [6, 7, 8]),
        num_computed_tokens=0,
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
    request = SimpleNamespace(
        request_id="r",
        req_id="r",
        num_tokens=48,
        block_hashes=[b"a", b"b", b"c"],
        block_ids=([1, 2, 3], [4, 5]),
        num_computed_tokens=0,
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
    request = SimpleNamespace(
        request_id="hybrid-sync", block_hashes=block_hashes, num_tokens=33
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, True)
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
    pool = scheduler._lookup_coordinator.block_pool
    for operation in ("get_new_blocks", "_insert_block_hash", "touch", "free_blocks"):
        monkeypatch.setattr(
            pool, operation, lambda *a, **kw: pytest.fail("lookup must be read-only")
        )
    request = SimpleNamespace(
        request_id="hybrid-lookup",
        block_hashes=block_hashes,
        num_tokens=33,
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
    scheduler.close()


@pytest.mark.parametrize("failure_stage", ["lookup", "match"])
def test_hybrid_shadow_lookup_cleans_up_after_failure(monkeypatch, failure_stage):
    """A failed query cannot leak shadow hits or consume pool capacity."""
    handle = _SchedulerHandle({})
    codec = BlockIdentityCodec(UMBPNamespace("shadow-cleanup"))
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({}), _hybrid_kv_cache_config(), handle, codec
    )
    request = SimpleNamespace(request_id="r", num_tokens=33, block_hashes=[b"a", b"b"])
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
    assert scheduler.load_async is indexer
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
    request = SimpleNamespace(
        request_id="partial",
        req_id="partial",
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
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={},
        )
    )

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
    request = SimpleNamespace(
        request_id="hybrid",
        req_id="hybrid",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
        block_ids=([1, 2], [8, 9]),
        num_computed_tokens=16,
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([1, 2], [8, 9])),
        0,
    )
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={},
            kv_connector_block_state=SimpleNamespace(
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
    request = SimpleNamespace(
        request_id="partial-prefix",
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
    request = SimpleNamespace(
        request_id="many-hashes",
        num_tokens=128,
        block_hashes=[bytes([index]) for index in range(32)],
    )
    for _ in range(2):
        # Mamba-style groups always load asynchronously.
        assert scheduler.get_num_new_matched_tokens(request, 0) == (124, True)
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
    request = SimpleNamespace(
        request_id="multi-partial",
        req_id="multi-partial",
        num_tokens=32,
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
        SimpleNamespace(
            finished_req_ids={request.request_id},
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={},
        )
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
    request = SimpleNamespace(
        request_id="req",
        req_id="req",
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
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"req": 1},
    )
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
        assert {p.key for p in meta.store_plans} == {
            scheduler.codec.key(h, g) for g, h in expected
        }
        assert meta.store_requests.get("req", []) == meta.store_plans
        # Resident blocks must not allocate plans that will only be discarded.
        assert len(created) == len(expected)
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
