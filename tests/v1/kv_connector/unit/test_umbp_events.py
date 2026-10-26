# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from concurrent.futures import Future
from contextlib import suppress
from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.umbp_test_utils import (
    _kv_cache_config,
    _SchedulerHandle,
    _vllm_config,
    _WorkerHandle,
    make_request,
    make_scheduler_output,
)
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
    UMBPStoreKVEvents,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    BlockTransferPlan,
    RankTopology,
    StoreEventResult,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.stats import (
    UMBPStoreConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
)
from vllm.platforms import current_platform
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    maybe_convert_block_hash,
)

mori_available = importlib.util.find_spec("mori") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and mori_available),
    reason="UMBP is only available on ROCm with the mori package installed",
)


def _finish_event_checks(scheduler):
    events = []
    while scheduler._event_lookup is not None:
        with suppress(RuntimeError):
            scheduler._event_lookup.future.result(timeout=5)
        events.extend(scheduler.take_events())
    return events


def _event_scheduler(*, ranks=1, enabled=True, limit=16):
    config = _vllm_config(
        {"kv_event_max_objects": limit},
        tensor_parallel_size=ranks,
        world_size=ranks,
    )
    config.kv_events_config = SimpleNamespace(enable_kv_cache_events=enabled)
    handle = _SchedulerHandle({})
    scheduler = UMBPStoreConnectorScheduler(
        config,
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("event-residency")),
        RankTopology(tp_size=ranks),
    )
    scheduler.bind_gpu_block_pool(BlockPool(8, True, 16))
    return scheduler, handle


def _schedule_store(scheduler, block_hash=b"a"):
    request = make_request(
        f"store-{scheduler._store_event_counter}",
        num_tokens=17,
        block_hashes=[block_hash],
        block_ids=([1],),
    )
    scheduler.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids),
        0,
    )
    return scheduler.build_connector_meta(
        make_scheduler_output(new_reqs=[request], scheduled_tokens={request.req_id: 16})
    )


def _complete_store(scheduler, event_id, workers=1):
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                store_events={event_id: StoreEventResult(workers)}
            )
        )
    )


def test_worker_events_publish_removal_from_any_rank_once():
    from vllm.distributed.kv_events import BlockRemoved
    from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
        UMBPStoreKVEvents,
    )

    def removed(block_hash):
        return BlockRemoved(
            block_hashes=[block_hash],
            medium="CPU",
        )

    def step(rank_events):
        # As KVOutputAggregator combines one container per worker.
        combined = UMBPStoreKVEvents(rank_events[0])
        for events in rank_events[1:]:
            combined.add_events(events)
            combined.increment_workers(1)
        connector.update_connector_output(SimpleNamespace(kv_cache_events=combined))
        return list(connector.take_events())

    connector = object.__new__(UMBPStoreConnector)
    connector.connector_scheduler = SimpleNamespace(
        update_connector_output=lambda output: None,
        take_events=lambda: [],
    )
    connector._kv_cache_events = None
    a, b = removed(b"a"), removed(b"b")

    assert step([[a, b], [a]]) == [a, b]
    assert step([[], [b]]) == [b]
    assert step([[], []]) == []


def test_kv_events_preserve_repeated_events_across_steps():
    from vllm.distributed.kv_events import BlockRemoved
    from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
        UMBPStoreKVEvents,
    )

    connector = object.__new__(UMBPStoreConnector)
    connector.connector_scheduler = SimpleNamespace(
        update_connector_output=lambda output: None,
        take_events=lambda: [],
    )
    connector._kv_cache_events = None
    removed = BlockRemoved(block_hashes=[b"a"], medium="CPU")
    other = BlockRemoved(block_hashes=[b"b"], medium="CPU")

    for event in (removed, other, removed):
        connector.update_connector_output(
            SimpleNamespace(kv_cache_events=UMBPStoreKVEvents([event]))
        )

    assert list(connector.take_events()) == [removed, other, removed]


@pytest.mark.parametrize("enable_events", [False, True])
def test_worker_reports_store_barrier_without_block_stored(enable_events):
    worker = UMBPStoreConnectorWorker(
        _WorkerHandle(), enable_kv_cache_events=enable_events
    )
    plan = BlockTransferPlan(
        key="event-key",
        block_id=3,
        block_hash=b"event-hash",
        parent_block_hash=b"parent",
        token_ids=(1, 2, 3, 4),
        block_size=4,
        group_id=0,
    )
    metadata = UMBPConnectorMetadata(
        store_plans=[plan],
        store_requests={"req": [plan]},
        store_event=7,
    )

    worker.enqueue_stores(metadata)
    worker.wait_for_save()
    result = worker.build_connector_worker_meta()

    assert result.store_events == {7: StoreEventResult(1)}
    assert worker.get_kv_events() == []


def test_store_event_failure_identity_aggregates_across_workers():
    first = StoreEventResult(1, {(0, b"failed-a")})
    second = StoreEventResult(1, {(1, b"failed-b")})

    metadata = UMBPConnectorWorkerMetadata(store_events={7: first})
    metadata.aggregate(UMBPConnectorWorkerMetadata(store_events={7: second}))

    assert metadata.store_events[7] == StoreEventResult(
        2,
        {(0, b"failed-a"), (1, b"failed-b")},
    )


def test_scheduler_emits_only_objects_stored_by_every_worker():
    config = _vllm_config(
        {"mode": "embedded"},
        tensor_parallel_size=2,
        world_size=2,
    )
    config.kv_events_config = SimpleNamespace(enable_kv_cache_events=True)
    scheduler = UMBPStoreConnectorScheduler(
        config,
        _kv_cache_config(),
        _SchedulerHandle([True, True]),
        BlockIdentityCodec(UMBPNamespace("store-quorum")),
        RankTopology(tp_size=2),
    )
    try:
        request = SimpleNamespace(
            request_id="r",
            req_id="r",
            num_tokens=32,
            num_prompt_tokens=32,
            num_computed_tokens=0,
            block_hashes=[b"a", b"b"],
            block_ids=([1, 2],),
            all_token_ids=list(range(32)),
        )
        scheduler.update_state_after_alloc(
            request,
            SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids),
            0,
        )
        metadata = scheduler.build_connector_meta(
            SimpleNamespace(
                finished_req_ids=set(),
                preempted_req_ids=set(),
                scheduled_new_reqs=[request],
                scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                num_scheduled_tokens={"r": 32},
            )
        )

        scheduler.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                    store_events={metadata.store_event: StoreEventResult(1)}
                )
            )
        )
        assert scheduler.take_events() == []

        scheduler.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=UMBPConnectorWorkerMetadata(
                    store_events={
                        metadata.store_event: StoreEventResult(1, {(0, b"a")})
                    }
                )
            )
        )
        events = scheduler.take_events() + _finish_event_checks(scheduler)
        [event] = events
        assert event.block_hashes == [maybe_convert_block_hash(b"b")]
        assert event.token_ids == list(range(16, 32))
    finally:
        scheduler.close()


def test_store_quorum_rechecks_eviction_from_an_earlier_step():
    scheduler, handle = _event_scheduler(ranks=2)
    try:
        metadata = _schedule_store(scheduler)
        keys = scheduler.codec.keys_for_topology(b"a", scheduler.topology, (0,))
        handle.hits.update(dict.fromkeys(keys, True))

        _complete_store(scheduler, metadata.store_event)
        assert scheduler.take_events() == []
        assert scheduler._gpu_block_pool.blocks[1].ref_cnt == 1
        handle.hits[keys[0]] = False
        _complete_store(scheduler, metadata.store_event)

        assert scheduler._gpu_block_pool.blocks[1].ref_cnt == 0
        assert scheduler.take_events() + _finish_event_checks(scheduler) == []
        assert scheduler._event_objects == {}
    finally:
        scheduler.close()


def test_old_residency_result_cannot_invalidate_a_new_store(monkeypatch):
    scheduler, _ = _event_scheduler()
    try:
        futures = []

        def submit(*args):
            future: Future[list[bool]] = Future()
            futures.append(future)
            return future

        monkeypatch.setattr(scheduler._lookup_executor, "submit", submit)
        first = _schedule_store(scheduler)
        _complete_store(scheduler, first.store_event)
        second = _schedule_store(scheduler)
        _complete_store(scheduler, second.store_event)
        futures[0].set_result([False])
        assert scheduler.take_events() == []
        futures[1].set_result([True])

        [event] = scheduler.take_events()
        assert event.block_hashes == [maybe_convert_block_hash(b"a")]
        assert scheduler._event_objects[(0, b"a")].published
    finally:
        scheduler.close()


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize(
    "failure", ["exception", "short-result", "long-result", "reset"]
)
def test_failed_or_reset_residency_check_cannot_publish_block_stored(
    failure, ranks, monkeypatch
):
    """A failed residency check or one outliving reset_store() emits nothing."""
    scheduler, _ = _event_scheduler(ranks=ranks)
    try:
        futures = []

        def submit(*args):
            future: Future[list[bool]] = Future()
            futures.append(future)
            return future

        monkeypatch.setattr(scheduler._lookup_executor, "submit", submit)
        metadata = _schedule_store(scheduler)
        _complete_store(scheduler, metadata.store_event, workers=ranks)

        # Transfer pins are released when the store completes, before residency.
        assert scheduler._gpu_block_pool.blocks[1].ref_cnt == 0
        [future] = futures
        if failure == "exception":
            future.set_exception(RuntimeError("injected residency failure"))
        elif failure == "short-result":
            future.set_result([True] * (ranks - 1))
        elif failure == "long-result":
            future.set_result([True] * (ranks + 1))
        else:
            # The lookup is already running, so reset cannot cancel it.
            assert future.set_running_or_notify_cancel()
            assert scheduler.reset_store()
            future.set_result([True] * ranks)

        assert scheduler.take_events() == []
        assert scheduler._event_lookup is None
        assert scheduler._event_objects == {}
        assert scheduler._pending_event_objects == {}
        assert not scheduler.has_pending_push_work()
        assert len(futures) == 1
        assert scheduler._gpu_block_pool.blocks[1].ref_cnt == 0
        if failure == "reset":
            # A store after reset publishes only from its own residency check.
            metadata = _schedule_store(scheduler)
            _complete_store(scheduler, metadata.store_event, workers=ranks)
            futures[1].set_result([True] * ranks)
            [event] = scheduler.take_events()
            assert isinstance(event, BlockStored)
            assert event.block_hashes == [maybe_convert_block_hash(b"a")]
    finally:
        scheduler.close()


def test_worker_removal_invalidates_an_older_residency_success(monkeypatch):
    scheduler, _ = _event_scheduler()
    try:
        futures = []

        def submit(*args):
            future: Future[list[bool]] = Future()
            futures.append(future)
            return future

        monkeypatch.setattr(scheduler._lookup_executor, "submit", submit)
        metadata = _schedule_store(scheduler)
        _complete_store(scheduler, metadata.store_event)
        futures[0].set_result([True])
        connector = object.__new__(UMBPStoreConnector)
        connector.connector_scheduler = scheduler
        connector._kv_cache_events = None
        removed = BlockRemoved(
            block_hashes=[maybe_convert_block_hash(b"a")], medium="CPU", group_idx=0
        )

        connector.update_connector_output(
            SimpleNamespace(
                kv_cache_events=UMBPStoreKVEvents([removed]),
                kv_connector_worker_meta=UMBPConnectorWorkerMetadata(),
            )
        )

        assert list(connector.take_events()) == [removed]
        new_store = _schedule_store(scheduler)
        _complete_store(scheduler, new_store.store_event)
        futures[1].set_result([True])
        assert len(scheduler.take_events()) == 1
    finally:
        scheduler.close()


def test_published_objects_are_rechecked_while_new_stores_arrive():
    scheduler, handle = _event_scheduler()
    try:
        first = _schedule_store(scheduler)
        handle.hits[scheduler.codec.key(b"a")] = True
        _complete_store(scheduler, first.store_event)
        assert len(scheduler.take_events() + _finish_event_checks(scheduler)) == 1
        handle.hits.clear()
        handle.hits[scheduler.codec.key(b"b")] = True
        scheduler._next_event_reconcile = 0
        second = _schedule_store(scheduler, b"b")
        _complete_store(scheduler, second.store_event)

        events = scheduler.take_events() + _finish_event_checks(scheduler)

        assert any(isinstance(event, BlockRemoved) for event in events)
        assert (0, b"a") not in scheduler._event_objects
        assert scheduler._event_objects[(0, b"b")].published
    finally:
        scheduler.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_kv_event_tracking_is_bounded_under_unique_stores(enabled):
    scheduler, handle = _event_scheduler(enabled=enabled, limit=2)
    try:
        events = []
        for index in range(10):
            block_hash = bytes([index + 1])
            metadata = _schedule_store(scheduler, block_hash)
            handle.hits[scheduler.codec.key(block_hash)] = True
            _complete_store(scheduler, metadata.store_event)
            events.extend(scheduler.take_events() + _finish_event_checks(scheduler))

        assert len(scheduler._event_objects) == (2 if enabled else 0)
        assert len(events) == (2 if enabled else 0)
        assert scheduler._event_publications_suppressed == (8 if enabled else 0)
        assert all(record.plan is None for record in scheduler._event_objects.values())
        if not enabled:
            assert handle.queries == []
    finally:
        scheduler.close()


def test_published_objects_are_removed_after_authoritative_miss():
    scheduler, handle = _event_scheduler()
    try:
        metadata = _schedule_store(scheduler)
        handle.hits[scheduler.codec.key(b"a")] = True
        _complete_store(scheduler, metadata.store_event)
        assert len(scheduler.take_events() + _finish_event_checks(scheduler)) == 1
        handle.hits.clear()
        scheduler._next_event_reconcile = 0

        events = scheduler.take_events() + _finish_event_checks(scheduler)

        assert len(events) == 1 and type(events[0]).__name__ == "BlockRemoved"
        assert events[0].block_hashes == [maybe_convert_block_hash(b"a")]
        assert scheduler._event_objects == {}
    finally:
        scheduler.close()


@pytest.mark.parametrize("enable_events", [False, True])
def test_worker_emits_block_removed_for_runtime_eviction(enable_events):
    class _EvictingWorkerHandle(_WorkerHandle):
        evicted_keys = ["umbp:vllm:v1:test:tp0:pcp0:dcp0:pp0:g2:65766963746564"]

        def take_evicted_keys(self):
            keys, self.evicted_keys = self.evicted_keys, []
            return keys

    handle = _EvictingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle, enable_kv_cache_events=enable_events)

    events = worker.get_kv_events()
    assert handle.evicted_keys == []
    if not enable_events:
        assert events == []
        return
    [event] = events

    assert event.block_hashes == [maybe_convert_block_hash(b"evicted")]
    assert event.group_idx == 2
    assert event.medium == "CPU"


def test_umbp_stats_aggregate_and_reduce():
    first = UMBPStoreConnectorStats()
    first.record("load", submitted=2, completed=1, failed=1, num_bytes=64)
    second = UMBPStoreConnectorStats(
        {"load": {"completed": 1, "num_bytes": 32, "unknown": 7}, "store": {}}
    )

    merged = first.aggregate(second)

    assert merged.reduce() == {
        "load_submitted": 2,
        "load_completed": 2,
        "load_failed": 1,
        "load_num_bytes": 96,
        "store_submitted": 0,
        "store_completed": 0,
        "store_failed": 0,
        "store_num_bytes": 0,
    }
    assert first.data["load"]["completed"] == 1
    assert second.data["load"] == {"completed": 1, "num_bytes": 32, "unknown": 7}
