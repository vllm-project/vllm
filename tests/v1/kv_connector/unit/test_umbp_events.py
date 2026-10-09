# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.umbp_test_utils import (
    _kv_cache_config,
    _SchedulerHandle,
    _vllm_config,
    _WorkerHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
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
from vllm.v1.core.kv_cache_utils import (
    maybe_convert_block_hash,
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
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("store-quorum")),
        RankTopology(tp_size=2),
    )
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
                store_events={metadata.store_event: StoreEventResult(1, {(0, b"a")})}
            )
        )
    )
    [event] = scheduler.take_events()
    assert event.block_hashes == [maybe_convert_block_hash(b"b")]
    assert event.token_ids == list(range(16, 32))


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
