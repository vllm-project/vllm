# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.umbp_test_utils import (
    _hybrid_kv_cache_config,
    _kv_cache_config,
    _SchedulerHandle,
    _vllm_config,
    make_scheduler_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    StoreEventResult,
    UMBPConnectorWorkerMetadata,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

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
                iter_blocks_after=lambda cursor: iter(free_blocks)
            ),
            touch=lambda selected: None,
            free_blocks=lambda selected: released.extend(selected),
        ),
        released,
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
    pool, released = _pool([None, None, None, None, block], [block])
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
    assert not scheduler.has_pending_push_work()
    scheduler.close()


def test_lazy_offload_rechecks_authoritative_runtime_residency():
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
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                store_events={first.store_event: StoreEventResult(1)}
            )
        )
    )
    handle.hits = [True]
    second = scheduler.build_connector_meta(make_scheduler_output())
    handle.hits = [False]
    third = scheduler.build_connector_meta(make_scheduler_output())

    assert len(first.store_plans) == 1
    assert second.store_plans == []
    assert [plan.key for plan in third.store_plans] == [first.store_plans[0].key]
    assert len(handle.queries) == 3
    scheduler.close()


@pytest.mark.parametrize("failure", ["exception", "wrong-length"])
def test_lazy_lookup_failure_is_retried(failure, monkeypatch):
    handle = _SchedulerHandle([])
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lazy_offload": True}),
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("lazy-retry")),
    )
    block = _block(4, b"a")
    pool, _ = _pool([None, None, None, None, block], [block])
    scheduler.bind_gpu_block_pool(pool)
    scheduler._lazy_scan_pending = True
    lookup = handle.lookup
    if failure == "exception":

        def fail_lookup(keys):
            del keys
            raise TimeoutError("unavailable")

        monkeypatch.setattr(handle, "lookup", fail_lookup)
    else:
        handle.hits = [False, False]

    failed = scheduler.build_connector_meta(make_scheduler_output())

    assert failed.store_plans == []
    assert scheduler._lazy_scan_pending

    monkeypatch.setattr(handle, "lookup", lookup)
    handle.hits = [False]
    retried = scheduler.build_connector_meta(make_scheduler_output())

    assert [plan.block_id for plan in retried.store_plans] == [4]
    assert not scheduler._lazy_scan_pending
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
