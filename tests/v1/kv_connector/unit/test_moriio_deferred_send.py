# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer deferred-send ACK vs timeout: blocks must not be force-freed."""

import threading
import time
from types import MethodType, SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.v1.outputs import KVConnectorOutput

_update = MoRIIOConnectorScheduler.update_connector_output
_unmap = MoRIIOConnectorScheduler.unmap_request_id
_map = MoRIIOConnectorScheduler.map_request_id


def _producer_sched() -> SimpleNamespace:
    sched = SimpleNamespace(
        is_producer=True,
        _deferred_send_deadlines={},
        _pending_sent_acks={},
        _defer_timeout=60.0,
        _stale_deferred_sends=set(),
        _stale_deferred_log_at=0.0,
        transfer_id_to_request_id={},
        request_id_to_transfer_id={},
    )
    sched.unmap_request_id = MethodType(_unmap, sched)
    sched.map_request_id = MethodType(_map, sched)
    return sched


def test_expired_deferred_send_is_not_finished_sending():
    sched = _producer_sched()
    sched.map_request_id("rid-a", "tid-a")
    sched._deferred_send_deadlines["rid-a"] = (time.monotonic() - 1.0, "tid-a")
    output = KVConnectorOutput(finished_sending=None)

    _update(sched, output)

    assert output.finished_sending is None
    assert "rid-a" in sched._deferred_send_deadlines
    assert sched.request_id_to_transfer_id == {"rid-a": "tid-a"}
    assert "rid-a" in sched._stale_deferred_sends
    assert sched.transfer_id_to_request_id == {"tid-a": "rid-a"}


def test_lost_acks_retain_all_deferred_entries_and_mappings():
    """A lost ACK path must keep every deferred entry and transfer-id
    mapping so a late ACK can still free. Force-freeing or dropping
    maps would return blocks to the pool while a READ may still
    reference them."""
    sched = _producer_sched()
    n = 1000
    expired_at = time.monotonic() - 1.0
    for i in range(n):
        rid, tid = f"rid-{i}", f"tid-{i}"
        sched.map_request_id(rid, tid)
        sched._deferred_send_deadlines[rid] = (expired_at, tid)

    output = KVConnectorOutput(finished_sending=None)
    _update(sched, output)

    assert output.finished_sending is None
    assert len(sched._deferred_send_deadlines) == n
    assert len(sched.request_id_to_transfer_id) == n
    assert len(sched.transfer_id_to_request_id) == n
    assert len(sched._stale_deferred_sends) == n

    ack_output = KVConnectorOutput(finished_sending={"rid-7"})
    _update(sched, ack_output)

    assert ack_output.finished_sending == {"rid-7"}
    assert "rid-7" not in sched._deferred_send_deadlines
    assert "rid-7" not in sched.request_id_to_transfer_id
    assert "tid-7" not in sched.transfer_id_to_request_id
    assert len(sched._deferred_send_deadlines) == n - 1
    assert len(sched.request_id_to_transfer_id) == n - 1
    assert len(sched.transfer_id_to_request_id) == n - 1


def test_ack_still_surfaces_deferred_send_for_free():
    sched = _producer_sched()
    sched.map_request_id("rid-a", "tid-a")
    sched._deferred_send_deadlines["rid-a"] = (time.monotonic() + 60.0, "tid-a")
    output = KVConnectorOutput(finished_sending={"rid-a"})

    _update(sched, output)

    assert output.finished_sending == {"rid-a"}
    assert "rid-a" not in sched._deferred_send_deadlines
    assert sched.request_id_to_transfer_id == {}
    assert sched.transfer_id_to_request_id == {}


def test_late_ack_after_timeout_still_frees():
    sched = _producer_sched()
    sched.map_request_id("rid-a", "tid-a")
    sched._deferred_send_deadlines["rid-a"] = (time.monotonic() - 1.0, "tid-a")
    stale_output = KVConnectorOutput(finished_sending=None)
    _update(sched, stale_output)
    assert stale_output.finished_sending is None
    assert "rid-a" in sched._stale_deferred_sends

    ack_output = KVConnectorOutput(finished_sending={"rid-a"})
    _update(sched, ack_output)

    assert ack_output.finished_sending == {"rid-a"}
    assert "rid-a" not in sched._deferred_send_deadlines
    assert "rid-a" not in sched._stale_deferred_sends
    assert sched.request_id_to_transfer_id == {}


def test_consumer_is_noop_for_deferred_send_reap():
    sched = SimpleNamespace(
        is_producer=False,
        _deferred_send_deadlines={"rid-a": (time.monotonic() - 1.0, "tid-a")},
    )
    output = KVConnectorOutput(finished_sending=None)
    _update(sched, output)
    assert output.finished_sending is None
    assert "rid-a" in sched._deferred_send_deadlines


class _PendingStatus:
    def Succeeded(self):
        return False

    def Failed(self):
        return False


class _FakeNotifyWrapper:
    def __init__(self):
        self.lock = threading.Lock()
        self.sent: list[tuple] = []

    def send_notify(
        self,
        transfer_id,
        host,
        port,
        message_type=None,
        message_fields=None,
    ):
        self.sent.append((transfer_id, host, port, message_type, message_fields))

    def shutdown(self):
        pass


def test_timed_out_read_notifies_producer_to_release():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 4
    worker.moriio_wrapper = _FakeNotifyWrapper()
    worker._recving_transfers = {"req": {"layer0": _PendingStatus()}}
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", "7000", "tx-timeout")
    }
    worker._recving_transfers_start = {"req": time.monotonic() - 200.0}

    assert worker._pop_done_transfers() == set()
    assert worker.moriio_wrapper.sent == [
        (
            "tx-timeout",
            "127.0.0.1",
            "7000",
            "release",
            {"consumer_tp_size": 4},
        )
    ]
    assert worker._recving_transfers == {}
    assert worker._recving_transfers_callback_addr == {}
    assert worker._recving_transfers_start == {}
