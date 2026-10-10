# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side event bookkeeping tests for SimpleCPUOffloadConnector."""

from typing import cast

import pytest
import torch

from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker


class _FakeEvent:
    def __init__(self, complete: bool) -> None:
        self.complete = complete
        self.query_count = 0

    def query(self) -> bool:
        self.query_count += 1
        return self.complete


@pytest.mark.skip_global_cleanup
def test_get_finished_reports_only_completed_prefixes():
    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    load_events = [_FakeEvent(True), _FakeEvent(False), _FakeEvent(True)]
    store_events = [_FakeEvent(True), _FakeEvent(False), _FakeEvent(True)]
    worker._load_events.extend(
        (i, cast(torch.Event, event)) for i, event in enumerate(load_events)
    )
    worker._store_events.extend(
        (i, cast(torch.Event, event)) for i, event in enumerate(store_events)
    )

    load_event_to_reqs = {i: [f"req-{i}"] for i in range(len(load_events))}
    for event_idx in range(len(load_events)):
        worker.bind_connector_metadata(
            SimpleCPUOffloadMetadata(
                load_event=event_idx,
                load_event_to_reqs=load_event_to_reqs,
                store_event=event_idx,
            )
        )

    assert worker.get_finished(set()) == (None, {"req-0"})
    worker_meta = worker.build_connector_worker_meta()
    assert worker_meta is not None
    assert worker_meta.completed_store_events == {0: 1}
    assert [event.query_count for event in load_events] == [1, 1, 0]
    assert [event.query_count for event in store_events] == [1, 1, 0]

    load_events[1].complete = True
    store_events[1].complete = True
    assert worker.get_finished(set()) == (None, {"req-1", "req-2"})
    worker_meta = worker.build_connector_worker_meta()
    assert worker_meta is not None
    assert worker_meta.completed_store_events == {1: 1, 2: 1}
    assert worker.build_connector_worker_meta() is None
    assert worker.get_finished(set()) == (None, None)
    assert [event.query_count for event in load_events] == [1, 2, 1]
    assert [event.query_count for event in store_events] == [1, 2, 1]
