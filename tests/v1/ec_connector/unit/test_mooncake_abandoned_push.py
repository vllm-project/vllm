# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A Mooncake EC push no producer will write must fail its requests now.

A follower shares the destination of a concurrent push of the same mm_hash,
and its own producer is told not to write. If that writer's producer abandons
the push, the follower's requests used to wait out `push_wait_timeout_s`.
"""

import pytest
import torch
import zmq

from vllm.distributed.ec_transfer.ec_connector.mooncake.control import (
    ConsumerControlServer,
    make_cancel_request,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeLoadSpec,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    ConsumerReservationManager,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.state import (
    SchedulerTransferState,
    SchedulerTransferTable,
)
from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.cpu_test

_HASH = "hash-A"


class _FakeTransfer:
    def register_memory(self, tensor):
        return 0

    def unregister_memory(self, tensor):
        return True

    def local_session(self):
        return "session"


def _manager() -> ConsumerReservationManager:
    pool = ConsumerMemoryPool(1 << 20, _FakeTransfer())
    pool.prepare(torch.device("cpu"))
    return ConsumerReservationManager(pool, lease_ttl=60.0, tombstone_limit=16)


def _writer_and_follower(manager):
    writer, writes = manager.reserve(
        "t-writer", _HASH, 8, (4,), "float16", torch.float16
    )
    follower, follower_writes = manager.reserve(
        "t-follower", _HASH, 8, (4,), "float16", torch.float16
    )
    assert writes and not follower_writes
    return writer, follower


@pytest.mark.parametrize(
    "refresh, abandoned",
    [
        (False, {"t-writer", "t-follower"}),
        # A refreshing producer re-reserves its own transfer at once.
        (True, {"t-follower"}),
    ],
    ids=["batch-failure", "refresh"],
)
def test_abandoned_writer_reports_its_followers(refresh, abandoned):
    manager = _manager()
    writer, _ = _writer_and_follower(manager)

    manager.cancel("t-writer", writer.reservation_id, abandon=True, refresh=refresh)

    assert {tid for tid, _ in manager.drain_abandoned()} == abandoned
    assert manager.drain_abandoned() == []


def test_abandoned_follower_is_reported():
    manager = _manager()
    _, follower = _writer_and_follower(manager)
    manager.cancel("t-follower", follower.reservation_id, abandon=True)
    assert manager.drain_abandoned() == [("t-follower", _HASH)]


def test_consumer_cancel_is_not_reported():
    """The Consumer already stopped waiting on what it cancelled itself."""
    manager = _manager()
    _writer_and_follower(manager)
    manager.cancel("t-follower", "")
    manager.cancel("t-writer", "")
    assert manager.drain_abandoned() == []


def _table() -> SchedulerTransferTable:
    return SchedulerTransferTable(resident_capacity=1 << 20, tombstone_ttl=60.0)


def test_abandoned_report_fails_the_waiting_request():
    table = _table()
    table.wait_for_event("t-1", "req-1", _HASH, deadline=100.0)

    assert table.observe_abandoned("t-1", _HASH, now=1.0)

    assert table.get("t-1").state is SchedulerTransferState.UNAVAILABLE
    assert table.take_unavailable_requests() == {"req-1"}


def test_abandoned_report_before_the_request_fails_it_on_arrival():
    table = _table()
    assert table.observe_abandoned("t-1", _HASH, now=1.0)
    assert table.take_unavailable_requests() == set()

    table.wait_for_event("t-1", "req-1", _HASH, deadline=100.0)

    assert table.take_unavailable_requests() == {"req-1"}


def test_abandoned_report_after_ready_is_ignored():
    """The tensor already landed, so the load can still proceed."""
    table = _table()
    spec = ECMooncakeLoadSpec(
        mm_hash=_HASH, nbytes=8, shape=(4,), dtype="float16", transfer_id="t-1"
    )
    table.observe_ready(spec, deadline=100.0)

    assert not table.observe_abandoned("t-1", _HASH, now=1.0)
    assert table.get("t-1").state is SchedulerTransferState.AVAILABLE


def test_control_server_publishes_abandoned_reservations():
    manager = _manager()
    writer, _ = _writer_and_follower(manager)
    port = get_open_port()
    server = ConsumerControlServer(
        "127.0.0.1",
        port,
        reserve=lambda payload: {},
        status=lambda transfer_id: None,
        complete=manager.complete,
        cancel=manager.cancel,
        reap=lambda: 0,
        drain_abandoned=manager.drain_abandoned,
    )
    server.start()
    context = zmq.Context()
    try:
        events = context.socket(zmq.PULL)
        events.setsockopt(zmq.RCVTIMEO, 5000)
        events.connect(f"tcp://127.0.0.1:{server.event_port}")
        control = context.socket(zmq.REQ)
        control.setsockopt(zmq.RCVTIMEO, 5000)
        control.connect(f"tcp://127.0.0.1:{port}")

        control.send_json(
            make_cancel_request("t-writer", writer.reservation_id, abandon=True)
        )
        assert control.recv_json() == {"ok": True, "result": {"cancelled": True}}

        received = {events.recv_json()["transfer_id"] for _ in range(2)}
        assert received == {"t-writer", "t-follower"}
    finally:
        server.close()
        context.destroy(linger=0)
