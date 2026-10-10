# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.umbp_test_utils import (
    _WorkerHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
    UMBPStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockTransferPlan,
    UMBPConnectorMetadata,
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


def test_kv_events_are_published_once_every_rank_reported_them():
    from vllm.distributed.kv_events import BlockStored
    from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import (
        UMBPStoreKVEvents,
    )

    def stored(block_hash):
        return BlockStored(
            block_hashes=[block_hash],
            parent_block_hash=None,
            token_ids=[],
            block_size=16,
            lora_id=None,
            medium="CPU",
            lora_name=None,
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
        update_connector_output=lambda output: None
    )
    connector._kv_cache_events = None
    a, b = stored(b"a"), stored(b"b")

    assert step([[a, b], [a]]) == [a]
    assert step([[], [b]]) == [b]
    assert step([[], []]) == []


@pytest.mark.parametrize("enable_events", [False, True])
def test_worker_emits_block_stored_event_after_store_completion(enable_events):
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
    )

    worker.enqueue_stores(metadata)
    worker.wait_for_save()
    events = worker.get_kv_events()
    result = worker.build_connector_worker_meta()

    assert result.store_events == {}
    if not enable_events:
        assert events == []
        return
    assert len(events) == 1
    assert events[0].block_hashes == [maybe_convert_block_hash(b"event-hash")]
    assert events[0].parent_block_hash == maybe_convert_block_hash(b"parent")
    assert events[0].token_ids == [1, 2, 3, 4]
    assert worker.get_kv_events() == []


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
