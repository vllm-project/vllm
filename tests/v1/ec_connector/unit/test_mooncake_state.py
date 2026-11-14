# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    ECMooncakeScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.state import (
    SchedulerTransferState,
    SchedulerTransferTable,
)


def test_missing_push_event_expires_and_notifies_once():
    table = SchedulerTransferTable(resident_capacity=0, tombstone_ttl=10)
    record = table.wait_for_event("request:0", "request", "media", deadline=1)
    assert table.expire(now=0, terminal_limit=1) == []
    assert table.take_unavailable_requests() == set()

    assert table.expire(now=1, terminal_limit=1) == [record]
    assert record.state is SchedulerTransferState.UNAVAILABLE
    assert table.take_unavailable_requests() == {"request"}
    assert table.expire(now=2, terminal_limit=1) == []
    assert table.take_unavailable_requests() == set()

    table.expire(now=11, terminal_limit=1)
    assert table.get("request:0") is None


def test_finished_request_cancels_waiter_without_transfer_metadata():
    scheduler = object.__new__(ECMooncakeScheduler)
    scheduler._is_consumer = True
    scheduler._is_producer = False
    scheduler._unresolved_transfer_ids = 0
    scheduler._transfers = SchedulerTransferTable(resident_capacity=0, tombstone_ttl=10)
    record = scheduler._transfers.wait_for_event(
        "request:0", "request", "media", deadline=100
    )
    scheduler._event_ready_shards = {}
    scheduler._pending_cancels = {}
    scheduler._control_addr = "unused"
    submitted = []
    scheduler._control_executor = SimpleNamespace(
        submit=lambda *args: submitted.append(args)
    )
    request = SimpleNamespace(
        request_id="request",
        mm_features=[SimpleNamespace(identifier="media")],
        ec_transfer_params=None,
    )

    assert scheduler.request_finished(request) == (False, None)
    assert record.state is SchedulerTransferState.CANCELLED
    assert len(submitted) == 1
    assert submitted[0][1:] == ("unused", "request:0")
    scheduler.request_finished(request)
    assert len(submitted) == 1
