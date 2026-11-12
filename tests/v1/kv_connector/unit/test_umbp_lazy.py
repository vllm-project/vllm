# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from types import SimpleNamespace

import pytest

import vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler as umbp_scheduler
from tests.v1.kv_connector.umbp_test_utils import (
    _hybrid_kv_cache_config,
    _kv_cache_config,
    _SchedulerHandle,
    _vllm_config,
    make_scheduler_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    RankTopology,
    StoreEventResult,
    UMBPConnectorWorkerMetadata,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)
from vllm.platforms import current_platform
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

mori_available = importlib.util.find_spec("mori") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and mori_available),
    reason="UMBP is only available on ROCm with the mori package installed",
)


def _block(block_id: int, block_hash: bytes):
    return SimpleNamespace(
        block_id=block_id,
        block_hash=make_block_hash_with_group_id(block_hash, 0),
        is_null=False,
        ref_cnt=0,
    )


def _pool(blocks, free_blocks, aliases=None):
    released = []
    return (
        SimpleNamespace(
            blocks=blocks,
            cached_block_hashes_by_block=aliases or {},
            free_block_queue=SimpleNamespace(
                iter_blocks_after=lambda cursor: iter(free_blocks),
                popleft_n=lambda n: [],
                prepend_n=lambda blocks: None,
            ),
            touch=lambda selected: None,
            free_blocks=lambda selected, evict_first=False: released.extend(selected),
        ),
        released,
    )


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        umbp_scheduler, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    return now


def _complete_store(scheduler, metadata, failed_objects=()):
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                store_events={
                    metadata.store_event: StoreEventResult(1, set(failed_objects))
                }
            )
        )
    )


def test_lazy_offload_defers_store_until_blocks_enter_free_queue():
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("lazy")),
    )
    request = SimpleNamespace(
        request_id="lazy",
        req_id="lazy",
        num_tokens=32,
        num_prompt_tokens=32,
        num_computed_tokens=32,
        num_in_flight_tokens=0,
        block_hashes=[b"a", b"b"],
        block_ids=([4, 5],),
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: ([4, 5],)),
        0,
    )
    active_output = make_scheduler_output()
    active_output.scheduled_new_reqs = [request]
    active_output.num_scheduled_tokens = {"lazy": 32}

    assert scheduler.build_connector_meta(active_output).store_plans == []

    block = _block(4, b"a")
    # Block 5 is the finished request's chain tail; settling walks the chain.
    pool, released = _pool([None, None, None, None, block, _block(5, b"b")], [block])
    scheduler.bind_gpu_block_pool(pool)
    scheduler.request_finished(request, ([4, 5],))

    metadata = scheduler.build_connector_meta(make_scheduler_output())

    assert [plan.block_id for plan in metadata.store_plans] == [4]
    assert metadata.store_event == 0
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                store_events={0: StoreEventResult(1)}
            )
        )
    )
    assert released == [block]
    # Pinning re-arms the scan; the next step finds nothing new.
    assert scheduler.build_connector_meta(make_scheduler_output()).store_plans == []
    assert not scheduler.has_pending_push_work()
    scheduler.close()


def test_lazy_offload_rechecks_residency_after_the_pool_evicts(clock):
    handle = _SchedulerHandle([])
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("lazy-residency")),
    )
    block = _block(4, b"a")
    pool, _ = _pool([None, None, None, None, block], [block])
    scheduler.bind_gpu_block_pool(pool)

    first = scheduler.build_connector_meta(make_scheduler_output())
    assert len(first.store_plans) == 1
    _complete_store(scheduler, first)
    handle.hits = [True]
    # The completed store confirmed the key, so steps before the recheck do not
    # look it up again.
    for _ in range(3):
        assert scheduler.build_connector_meta(make_scheduler_output()).store_plans == []
    assert len(handle.queries) == 1

    # The pool evicts the object; the next recheck finds it gone.
    handle.hits = [False]
    clock[0] += umbp_scheduler._LAZY_RECHECK_S
    again = scheduler.build_connector_meta(make_scheduler_output())

    assert [plan.key for plan in again.store_plans] == [first.store_plans[0].key]
    assert len(handle.queries) == 2
    scheduler.close()


def test_lazy_offload_repairs_a_missing_nonzero_rank():
    codec = BlockIdentityCodec(UMBPNamespace("lazy-rank-residency"))
    topology = RankTopology(tp_size=2)
    rank_keys = codec.keys_for_topology(b"a", topology, (0,))
    handle = _SchedulerHandle({rank_keys[0]: True})
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"lazy_offload": True}, tensor_parallel_size=2, world_size=2),
        _kv_cache_config(),
        handle,
        codec,
        topology,
    )
    block = _block(4, b"a")
    pool, _ = _pool([None, None, None, None, block], [block])
    scheduler.bind_gpu_block_pool(pool)
    try:
        metadata = scheduler.build_connector_meta(make_scheduler_output())
        assert [plan.block_id for plan in metadata.store_plans] == [4]
        assert handle.queries == [list(rank_keys)]
        scheduler.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                    store_events={metadata.store_event: StoreEventResult(2)},
                ),
            )
        )
        handle.hits = dict.fromkeys(rank_keys, True)
        assert not scheduler.build_connector_meta(make_scheduler_output()).store_plans
    finally:
        scheduler.close()


def test_lazy_offload_backs_off_while_the_pool_is_unavailable(clock):
    """A handle reports a rank it cannot ask as `None`, not as a miss."""
    handle = _SchedulerHandle([None])
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("lazy-unavailable")),
    )
    block = _block(4, b"a")
    pool, _ = _pool([None, None, None, None, block], [block])
    scheduler.bind_gpu_block_pool(pool)

    assert scheduler.build_connector_meta(make_scheduler_output()).store_plans == []
    assert scheduler._lazy_lookup_failures == 1
    # Until the backoff passes, steps neither look up nor keep the engine busy.
    assert scheduler.build_connector_meta(make_scheduler_output()).store_plans == []
    assert len(handle.queries) == 1
    assert not scheduler.has_pending_push_work()

    handle.hits = [False]
    clock[0] += umbp_scheduler._LAZY_RETRY_INITIAL_S
    retried = scheduler.build_connector_meta(make_scheduler_output())

    assert [plan.block_id for plan in retried.store_plans] == [4]
    assert scheduler._lazy_lookup_failures == 0
    scheduler.close()


def test_lazy_offload_backs_off_failed_stores_on_a_dead_server(clock):
    """A killed standalone server answers lookups with misses and fails stores."""
    handle = _SchedulerHandle([False])
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "standalone", "lazy_offload": True}),
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("lazy-dead-server")),
    )
    block = _block(4, b"a")
    pool, _ = _pool([None, None, None, None, block], [block])
    scheduler.bind_gpu_block_pool(pool)

    first = scheduler.build_connector_meta(make_scheduler_output())
    assert [plan.block_id for plan in first.store_plans] == [4]
    _complete_store(scheduler, first, failed_objects={(0, b"a")})
    # Until the backoff passes, the window is not stored again.
    assert scheduler.build_connector_meta(make_scheduler_output()).store_plans == []
    assert len(handle.queries) == 1
    assert not scheduler.has_pending_push_work()

    clock[0] += umbp_scheduler._LAZY_RETRY_INITIAL_S
    retried = scheduler.build_connector_meta(make_scheduler_output())

    assert [plan.block_id for plan in retried.store_plans] == [4]
    scheduler.close()


def test_lazy_offload_includes_secondary_block_hashes():
    handle = _SchedulerHandle([])
    codec = BlockIdentityCodec(UMBPNamespace("lazy-alias"))
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        handle,
        codec,
    )
    block = _block(4, b"a")
    secondary = make_block_hash_with_group_id(b"b", 0)
    pool, _ = _pool(
        [None, None, None, None, block],
        [block],
        {block.block_id: {secondary}},
    )
    scheduler.bind_gpu_block_pool(pool)

    metadata = scheduler.build_connector_meta(make_scheduler_output())

    assert {(plan.block_id, plan.key) for plan in metadata.store_plans} == {
        (4, codec.key(b"a", 0)),
        (4, codec.key(b"b", 0)),
    }
    scheduler.close()


@pytest.mark.parametrize("aliased_head", [False, True])
def test_lazy_offload_closes_full_attention_prefix(aliased_head):
    codec = BlockIdentityCodec(UMBPNamespace("lazy-prefix"))
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        codec,
    )
    head = _block(3, b"a")
    tail = _block(4, b"b")
    aliases = {}
    if aliased_head:
        aliases[head.block_id] = {head.block_hash}
        head.block_hash = make_block_hash_with_group_id(b"other", 0)
    pool, _ = _pool([None, None, None, head, tail], [tail], aliases)
    scheduler.bind_gpu_block_pool(pool)
    request = SimpleNamespace(
        request_id="lazy-prefix",
        num_tokens=33,
        num_prompt_tokens=33,
        num_computed_tokens=33,
        num_in_flight_tokens=0,
        block_hashes=[b"a", b"b"],
    )
    scheduler.request_finished(request, ([3, 4],))

    metadata = scheduler.build_connector_meta(make_scheduler_output())

    assert [(plan.block_id, plan.key) for plan in metadata.store_plans] == [
        (4, codec.key(b"b", 0)),
        (3, codec.key(b"a", 0)),
    ]
    scheduler.close()


def test_lazy_offload_drops_recycled_prefix_chain_tail():
    codec = BlockIdentityCodec(UMBPNamespace("lazy-recycled-prefix"))
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        codec,
    )
    head = _block(3, b"old-head")
    tail = _block(4, b"old-tail")
    pool, _ = _pool([None, None, None, head, tail], [tail])
    scheduler.bind_gpu_block_pool(pool)
    request = SimpleNamespace(
        request_id="old-request",
        num_tokens=32,
        num_prompt_tokens=32,
        num_computed_tokens=32,
        num_in_flight_tokens=0,
        block_hashes=[b"old-head", b"old-tail"],
    )
    scheduler.request_finished(request, ([3, 4],))

    # Core has recycled the tail block for a different prefix. The remembered
    # chain no longer belongs to this free-queue entry and must not pull in its
    # old head block.
    tail.block_hash = make_block_hash_with_group_id(b"new-tail", 0)
    metadata = scheduler.build_connector_meta(make_scheduler_output())

    assert [(plan.block_id, plan.key) for plan in metadata.store_plans] == [
        (4, codec.key(b"new-tail", 0))
    ]
    assert 4 not in scheduler._lazy_prefix_chains_by_block
    scheduler.close()


def test_lazy_target_matches_cache_group_geometry():
    assert (
        UMBPStoreConnectorScheduler._estimate_lazy_target_blocks(
            _hybrid_kv_cache_config(),
            max_num_batched_tokens=64,
            dcp_size=1,
        )
        == 12
    )


def test_lazy_offload_is_disabled_for_consumer_role():
    config = _vllm_config({"mode": "embedded", "lazy_offload": True})
    config.kv_transfer_config.kv_role = "kv_consumer"

    scheduler = UMBPStoreConnectorScheduler(
        config,
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("consumer")),
    )

    assert not scheduler.store_enabled
    assert not scheduler.lazy_offload
    scheduler.close()


@pytest.mark.parametrize("value", [-1, 1.5, "4"])
def test_lazy_offload_rejects_invalid_scan_limit(value):
    with pytest.raises(ValueError, match="lazy_offload_max_blocks"):
        UMBPStoreConnectorScheduler(
            _vllm_config(
                {
                    "mode": "embedded",
                    "lazy_offload": True,
                    "lazy_offload_max_blocks": value,
                }
            ),
            _kv_cache_config(),
            _SchedulerHandle([]),
            BlockIdentityCodec(UMBPNamespace("invalid-lazy-limit")),
        )


def test_lazy_offload_looks_up_tp_replicated_groups_once():
    codec = BlockIdentityCodec(
        UMBPNamespace("lazy-replicated"), replicated_groups=frozenset({0})
    )
    topology = RankTopology(tp_size=2)
    keys = codec.keys_for_topology(b"a", topology, (0,))
    handle = _SchedulerHandle({})
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"lazy_offload": True}, tensor_parallel_size=2, world_size=2),
        _kv_cache_config(),
        handle,
        codec,
        topology,
    )
    block = _block(4, b"a")
    pool, _ = _pool([None, None, None, None, block], [block])
    scheduler.bind_gpu_block_pool(pool)
    try:
        metadata = scheduler.build_connector_meta(make_scheduler_output())

        assert len(keys) == 1
        assert handle.queries == [list(keys)]
        assert [plan.block_id for plan in metadata.store_plans] == [4]
        assert scheduler._lazy_lookup_failures == 0
    finally:
        scheduler.close()


def _lazy_scheduler_on_real_pool(max_blocks):
    pool = BlockPool(num_gpu_blocks=40, enable_caching=True, hash_block_size=16)
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "mode": "embedded",
                "lazy_offload": True,
                "lazy_offload_max_blocks": max_blocks,
            }
        ),
        _kv_cache_config(),
        _SchedulerHandle({}),
        BlockIdentityCodec(UMBPNamespace("lazy-free-order")),
    )
    scheduler.bind_gpu_block_pool(pool)
    return scheduler, pool


def _finish_cached_request(scheduler, pool, request_id, num_blocks):
    blocks = pool.get_new_blocks(num_blocks)
    request = SimpleNamespace(
        request_id=request_id,
        req_id=request_id,
        num_tokens=16 * num_blocks,
        num_prompt_tokens=16 * num_blocks,
        num_computed_tokens=16 * num_blocks,
        num_in_flight_tokens=0,
        block_hashes=[
            BlockHash(f"{request_id}-{index}".encode()) for index in range(num_blocks)
        ],
    )
    pool.cache_full_blocks(request, blocks, 0, num_blocks, 16, 0)
    scheduler.request_finished(request, ([block.block_id for block in blocks],))
    pool.free_blocks(reversed(blocks))
    return blocks


def _free_queue(pool):
    return [block.block_id for block in pool.free_block_queue.iter_blocks_after(None)]


def test_lazy_store_returns_blocks_behind_free_empty_blocks():
    scheduler, pool = _lazy_scheduler_on_real_pool(max_blocks=40)
    a = _finish_cached_request(scheduler, pool, "a", 4)
    b = _finish_cached_request(scheduler, pool, "b", 4)
    before = _free_queue(pool)
    try:
        metadata = scheduler.build_connector_meta(make_scheduler_output())
        assert {plan.block_id for plan in metadata.store_plans} == {
            block.block_id for block in a + b
        }
        _complete_store(scheduler, metadata)

        assert _free_queue(pool) == before
        # A one-block allocation takes an empty block, not A's oldest one.
        assert pool.get_new_blocks(1)[0].block_hash is None
        assert all(block.block_hash is not None for block in a)
    finally:
        scheduler.close()


def test_lazy_store_frees_pulled_in_prefix_blocks_to_the_mru_end():
    scheduler, pool = _lazy_scheduler_on_real_pool(max_blocks=1)
    head, tail = _finish_cached_request(scheduler, pool, "x", 2)
    others = [
        _finish_cached_request(scheduler, pool, f"c{index}", 1)[0] for index in range(7)
    ]
    # Another request hits the prefix head, so it moves to the MRU end.
    pool.touch([head])
    pool.free_blocks([head])
    # Use up the empty blocks, so the tail is next to be evicted.
    empty = [
        block
        for block in pool.free_block_queue.iter_blocks_after(None)
        if block.block_hash is None
    ]
    pool.get_new_blocks(len(empty))
    assert _free_queue(pool) == [
        tail.block_id,
        *(other.block_id for other in others),
        head.block_id,
    ]
    try:
        metadata = scheduler.build_connector_meta(make_scheduler_output())
        assert [plan.block_id for plan in metadata.store_plans] == [
            tail.block_id,
            head.block_id,
        ]
        _complete_store(scheduler, metadata)

        assert _free_queue(pool) == [
            tail.block_id,
            *(other.block_id for other in others),
            head.block_id,
        ]
    finally:
        scheduler.close()
