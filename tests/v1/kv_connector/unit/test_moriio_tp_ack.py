# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket
import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    HandshakeError,
    MoRIIOMode,
    MoRIIOTransferAck,
    TransferError,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorWorker,
    get_moriio_expected_ack_count,
    get_moriio_remote_tp_rank,
    resolve_moriio_transfer_ack,
    validate_moriio_heterogeneous_tp_kv_heads,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)


def test_notify_listener_bind_failure_reaches_caller():
    wrapper = MoRIIOWrapper(moriio_engine=object())

    with socket.socket() as occupied_port:
        occupied_port.bind(("", 0))
        occupied_port.listen()
        wrapper.notify_port = occupied_port.getsockname()[1]

        with pytest.raises(HandshakeError, match="Address already in use"):
            wrapper.async_wait_reqid()

    assert wrapper.notify_thread is None


def test_remote_tp_rank_same_tp_maps_to_self():
    assert [get_moriio_remote_tp_rank(rank, 4, 4) for rank in range(4)] == [
        0,
        1,
        2,
        3,
    ]


def test_remote_tp_rank_p4_d8_floor_maps_decode_to_prefill():
    assert [get_moriio_remote_tp_rank(rank, 8, 4) for rank in range(8)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_remote_tp_rank_p8_d4_maps_to_first_prefill_rank_per_pair():
    assert [get_moriio_remote_tp_rank(rank, 4, 8) for rank in range(4)] == [
        0,
        2,
        4,
        6,
    ]


@pytest.mark.parametrize(
    ("local_tp_rank", "local_tp_size", "remote_tp_size"),
    [
        (0, 6, 4),
        (0, 4, 6),
    ],
)
def test_remote_tp_rank_invalid_non_multiple_tp_raises(
    local_tp_rank: int, local_tp_size: int, remote_tp_size: int
):
    with pytest.raises(ValueError, match="multiple"):
        get_moriio_remote_tp_rank(local_tp_rank, local_tp_size, remote_tp_size)


@pytest.mark.parametrize(
    ("local_tp_size", "remote_tp_size", "total_num_kv_heads"),
    [
        (4, 4, 8),
        (8, 4, 4),
        (4, 8, 4),
    ],
)
def test_heterogeneous_tp_head_guard_allows_supported_layouts(
    local_tp_size: int, remote_tp_size: int, total_num_kv_heads: int
):
    validate_moriio_heterogeneous_tp_kv_heads(
        local_tp_size,
        remote_tp_size,
        total_num_kv_heads,
        is_mla=False,
    )


def test_heterogeneous_tp_head_guard_allows_mla_layouts():
    validate_moriio_heterogeneous_tp_kv_heads(
        local_tp_size=2,
        remote_tp_size=4,
        total_num_kv_heads=4,
        is_mla=True,
    )


@pytest.mark.parametrize(
    ("local_tp_size", "remote_tp_size", "total_num_kv_heads"),
    [
        (4, 2, 4),
        (2, 4, 4),
    ],
)
def test_heterogeneous_tp_head_guard_rejects_split_kv_heads(
    local_tp_size: int, remote_tp_size: int, total_num_kv_heads: int
):
    with pytest.raises(NotImplementedError, match="replicated KV heads"):
        validate_moriio_heterogeneous_tp_kv_heads(
            local_tp_size,
            remote_tp_size,
            total_num_kv_heads,
            is_mla=False,
        )


def test_expected_ack_count_for_homogeneous_or_smaller_consumer_tp_is_one():
    assert get_moriio_expected_ack_count(4, 4) == 1
    assert get_moriio_expected_ack_count(8, 4) == 1


def test_expected_ack_count_for_decode_fan_in():
    assert get_moriio_expected_ack_count(4, 8) == 2


def test_expected_ack_count_rejects_non_multiple_fan_in():
    with pytest.raises(ValueError, match="multiple"):
        get_moriio_expected_ack_count(4, 6)


def test_plain_string_ack_is_backward_compatible_single_ack():
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            "tx-plain",
            producer_tp_size=4,
            live_transfer_ids={"tx-plain"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-plain"
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-plain"}


def test_structured_release_ack_waits_for_all_expected_acks():
    ack = MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-fanin"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {"tx-fanin": 1}
    assert completed_transfer_ids == set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-fanin"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-fanin"
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-fanin"}


def test_duplicate_ack_after_completion_does_not_resolve_twice():
    ack = MoRIIOTransferAck("tx-dup", consumer_tp_size=8)
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-dup"
    )
    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-dup"}


def test_ack_for_non_live_transfer_is_ignored():
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            MoRIIOTransferAck("tx-stale", consumer_tp_size=8),
            producer_tp_size=4,
            live_transfer_ids={"tx-live"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {}
    assert completed_transfer_ids == set()


def test_worker_get_finished_counts_structured_release_fan_in():
    class FakeWrapper:
        def __init__(self):
            self.batches = [
                [MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)],
                [MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)],
            ]

        def pop_finished_req_ids(self):
            return self.batches.pop(0)

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.world_size = 4
    worker.moriio_wrapper = FakeWrapper()
    worker.transfer_id_to_request_id = {"tx-fanin": "req-fanin"}
    worker._consumer_notification_counts = {}
    worker._completed_consumer_notifications = set()
    worker._pending_unmapped_acks = []

    assert worker.get_finished() == (set(), set())
    assert worker._consumer_notification_counts == {"tx-fanin": 1}

    assert worker.get_finished() == ({"req-fanin"}, set())
    assert worker._consumer_notification_counts == {}
    assert worker._completed_consumer_notifications == {"tx-fanin"}


@pytest.mark.parametrize("exit_kind", ["killed", "eof"])
def test_worker_reports_unexpected_heartbeat_exit(exit_kind):
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_heartbeat import (
        MoRIIOHeartbeat,
    )

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.moriio_wrapper = SimpleNamespace(
        pop_finished_req_ids=lambda: [], shutdown=lambda: None
    )
    worker._pending_unmapped_acks = []
    worker._heartbeat = MoRIIOHeartbeat("tcp://127.0.0.1:1", {"type": "P"}, 0.02)
    process = worker._heartbeat._process
    try:
        assert worker.get_finished() == (set(), set())
        if exit_kind == "killed":
            process.kill()
        else:
            assert process.stdin is not None
            process.stdin.close()
        returncode = process.wait(timeout=5)
        with pytest.raises(RuntimeError, match=f"heartbeat.*{returncode}"):
            worker.get_finished()
    finally:
        worker.shutdown()


def test_read_completion_sends_structured_release_with_consumer_tp_size():
    class DoneStatus:
        def Succeeded(self):
            return True

        def Failed(self):
            return False

    class FakeWrapper:
        def __init__(self):
            self.lock = threading.Lock()
            self.sent = []

        # Exercise the real batch verdict rather than restating it here.
        poll_transfer_batch = MoRIIOWrapper.poll_transfer_batch

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

    wrapper = FakeWrapper()
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 8
    worker.moriio_config = SimpleNamespace(recv_abort_timeout=600.0)
    worker.moriio_wrapper = wrapper
    # A layer maps to the list of reads posted for it (a KDA layer posts two).
    worker._recving_transfers = defaultdict(
        dict, {"req": {"layer0": [DoneStatus(), DoneStatus()]}}
    )
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", "7000", "tx-release")
    }
    # Transfer-timeout reaping state consulted by _pop_done_transfers.
    worker._recving_transfers_start = {}
    # Load-error bookkeeping cleared alongside a completed transfer.
    worker._recving_local_blocks = {}
    worker._invalid_block_ids = set()

    assert worker._pop_done_transfers() == {"tx-release"}
    assert wrapper.sent == [
        (
            "tx-release",
            "127.0.0.1",
            "7000",
            "release",
            {"consumer_tp_size": 8},
        )
    ]
    assert worker._recving_transfers == {}
    assert worker._recving_transfers_callback_addr == {}


def test_aborted_read_without_local_blocks_only_releases_remote_blocks():
    class FakeWrapper:
        def __init__(self):
            self.sent = []

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

    wrapper = FakeWrapper()
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.mode = MoRIIOMode.READ
    worker.world_size = 8
    worker.moriio_wrapper = wrapper

    worker._read_blocks(
        local_block_ids=[],
        remote_block_ids=[[10, 11], [90]],
        dst_engine_id="prefill",
        request_id="req-aborted",
        transfer_id="tx-aborted",
        remote_host="127.0.0.1",
        remote_notify_port=7000,
        remote_tp_size=8,
        remote_dp_rank=0,
        chosen_tp=2,
    )

    assert wrapper.sent == [
        (
            "tx-aborted",
            "127.0.0.1",
            "7002",
            "release",
            {"consumer_tp_size": 8},
        )
    ]


def test_read_completion_waits_for_every_posted_read():
    class Status:
        def __init__(self, done: bool):
            self.done = done

        def Succeeded(self):
            return self.done

        def Failed(self):
            return False

    class FakeWrapper:
        def __init__(self):
            self.lock = threading.Lock()
            self.sent = []

        poll_transfer_batch = MoRIIOWrapper.poll_transfer_batch

        def send_notify(self, *args, **kwargs):
            self.sent.append((args, kwargs))

        def shutdown(self):
            pass

    wrapper = FakeWrapper()
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 8
    worker.moriio_config = SimpleNamespace(recv_abort_timeout=600.0)
    worker.moriio_wrapper = wrapper
    worker._recving_transfers = defaultdict(
        dict, {"req": {"kda_layer": [Status(done=True), Status(done=False)]}}
    )
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", "7000", "tx-pending")
    }
    worker._recving_transfers_start = {"req": float("inf")}
    worker._recving_local_blocks = {"req": [1, 2]}
    worker._invalid_block_ids = set()

    assert worker._pop_done_transfers() == set()
    assert "req" in worker._recving_transfers
    assert wrapper.sent == []


@pytest.mark.parametrize(
    ("has_mamba", "expected_invalid"),
    [(False, {1, 2}), (True, set())],
)
def test_failed_read_reports_blocks_only_without_hma(has_mamba, expected_invalid):
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = has_mamba
    worker._recving_local_blocks = {"req": [1, 2]}
    worker._invalid_block_ids = set()

    worker._record_failed_recv("req")

    assert worker.get_block_ids_with_load_errors() == expected_invalid


def test_hybrid_step_barrier_fails_closed():
    class FailingWrapper:
        def waiting_for_transfer_complete(self, _statuses, fail_fast=True):
            raise TransferError("failed")

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = True
    worker._reads_issued_this_step = [_ReadStatus("failed")]
    worker.moriio_wrapper = FailingWrapper()

    with pytest.raises(TransferError, match="failed"):
        worker._await_reads_issued_this_step()


def test_step_barrier_awaits_every_read_before_forward():
    class RecordingWrapper:
        def __init__(self):
            self.awaited = []

        def waiting_for_transfer_complete(self, statuses, fail_fast=True):
            self.awaited.append(list(statuses))

        def shutdown(self):
            pass

    attn, indexer, mtp_attn = object(), object(), object()
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = False
    worker._reads_issued_this_step = [attn, indexer, mtp_attn]
    worker.moriio_wrapper = RecordingWrapper()

    worker._await_reads_issued_this_step()
    worker._await_reads_issued_this_step()

    assert worker.moriio_wrapper.awaited == [[attn, indexer, mtp_attn]]
    assert worker._reads_issued_this_step == []


def test_wait_for_layer_load_is_noop():
    class _Boom:
        def __getattr__(self, name):
            raise AssertionError("wait_for_layer_load must not touch the worker")

    connector = MoRIIOConnector.__new__(MoRIIOConnector)
    connector.connector_worker = _Boom()

    assert connector.wait_for_layer_load("layer") is None


def test_step_barrier_without_mamba_does_not_raise_on_failure():
    # Without HMA, failures go to get_block_ids_with_load_errors.
    class FailingWrapper:
        def waiting_for_transfer_complete(self, _statuses, fail_fast=True):
            raise TransferError("failed")

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = False
    worker._reads_issued_this_step = [_ReadStatus("failed")]
    worker.moriio_wrapper = FailingWrapper()

    worker._await_reads_issued_this_step()
    assert worker._reads_issued_this_step == []


class _ReadStatus:
    def __init__(self, state: str):
        self.state = state

    def Succeeded(self):
        return self.state == "success"

    def Failed(self):
        return self.state == "failed"

    def Message(self):
        return "SQ full"

    def Code(self):
        return -1


class _PolledStatus(_ReadStatus):
    """Lands on its succeed_after-th Succeeded() poll; never if None."""

    def __init__(self, succeed_after: int | None = None):
        super().__init__("in_progress")
        self.succeed_after = succeed_after
        self.polls = 0

    def Succeeded(self):
        self.polls += 1
        if self.polls == self.succeed_after:
            self.state = "success"
        return super().Succeeded()


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = 0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps += 1
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(moriio_engine, "time", clock)
    return clock


def _worker_with_failed_and_pending_read(has_mamba, pending_b, monkeypatch):
    failed_a = _ReadStatus("failed")
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = has_mamba
    worker.world_size = 1
    worker.moriio_config = SimpleNamespace(recv_abort_timeout=600.0)
    worker.moriio_wrapper = MoRIIOWrapper(moriio_engine=object(), transfer_timeout=5.0)
    monkeypatch.setattr(worker.moriio_wrapper, "send_notify", lambda *a, **kw: None)
    worker._reads_issued_this_step = [failed_a, pending_b]
    worker._recving_transfers = {
        "a": {"layer0": [failed_a]},
        "b": {"layer0": [pending_b]},
    }
    worker._recving_transfers_callback_addr = {
        "a": ("127.0.0.1", "7000", "tx-a"),
        "b": ("127.0.0.1", "7000", "tx-b"),
    }
    worker._recving_transfers_start = {}
    worker._recving_local_blocks = {"a": [1, 2], "b": [3, 4]}
    worker._invalid_block_ids = set()
    return worker


def test_step_barrier_failure_still_awaits_other_requests(monkeypatch, fake_clock):
    late_b = _PolledStatus(succeed_after=5)
    worker = _worker_with_failed_and_pending_read(False, late_b, monkeypatch)

    worker._await_reads_issued_this_step()

    assert late_b.state == "success"
    assert late_b.polls >= 5
    assert fake_clock.sleeps >= 4
    assert worker._pop_done_transfers() == {"tx-b"}
    assert worker.get_block_ids_with_load_errors() == {1, 2}


def test_hybrid_step_barrier_failure_does_not_wait(monkeypatch, fake_clock):
    pending_b = _PolledStatus()
    worker = _worker_with_failed_and_pending_read(True, pending_b, monkeypatch)

    with pytest.raises(TransferError, match="SQ full"):
        worker._await_reads_issued_this_step()

    assert pending_b.state == "in_progress"
    assert fake_clock.sleeps == 0


@pytest.mark.parametrize("has_mamba", [False, True])
def test_step_barrier_timeout_fails_closed_while_reads_in_flight(has_mamba, fake_clock):
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = has_mamba
    worker.moriio_wrapper = MoRIIOWrapper(moriio_engine=object(), transfer_timeout=0.05)
    worker._reads_issued_this_step = [_PolledStatus(), _PolledStatus()]

    with pytest.raises(TransferError, match="timed out"):
        worker._await_reads_issued_this_step()

    assert fake_clock.now > 0.05


@pytest.mark.skipif(
    moriio_engine.StatusCode is None, reason="mori batched wait unavailable"
)
@pytest.mark.parametrize("has_mamba", [False, True])
def test_step_barrier_native_timeout_fails_closed_while_reads_in_flight(has_mamba):
    engine = SimpleNamespace(
        wait_all=Mock(return_value=moriio_engine.StatusCode.IN_PROGRESS)
    )
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = has_mamba
    worker.moriio_wrapper = MoRIIOWrapper(moriio_engine=engine, transfer_timeout=0.05)
    worker._reads_issued_this_step = [_PolledStatus(), _PolledStatus()]

    with pytest.raises(TransferError, match="timed out"):
        worker._await_reads_issued_this_step()

    engine.wait_all.assert_called_once()


def test_step_barrier_noop_when_no_reads_posted():
    class RecordingWrapper:
        def __init__(self):
            self.called = False

        def waiting_for_transfer_complete(self, _statuses):
            self.called = True

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = False
    worker._reads_issued_this_step = []
    worker.moriio_wrapper = RecordingWrapper()

    worker._await_reads_issued_this_step()

    assert worker.moriio_wrapper.called is False


def test_write_mode_read_blocks_posts_no_read():
    class RecordingWrapper:
        def __init__(self):
            self.sent = []

        def send_notify(self, *args, **kwargs):
            self.sent.append((args, kwargs))

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.mode = MoRIIOMode.WRITE
    worker._reads_issued_this_step = []
    worker.moriio_wrapper = RecordingWrapper()

    worker._read_blocks(
        local_block_ids=[1, 2],
        remote_block_ids=[[10, 11], [90]],
        dst_engine_id="prefill",
        request_id="req-write",
        transfer_id="tx-write",
        remote_host="127.0.0.1",
        remote_notify_port=7000,
        remote_tp_size=8,
    )

    assert worker._reads_issued_this_step == []
    assert worker.moriio_wrapper.sent == []


def test_producer_start_load_kv_posts_no_read():
    class RecordingWrapper:
        def __init__(self):
            self.waited = False
            self.async_waited = False

        def async_wait_reqid(self):
            self.async_waited = True

        def waiting_for_transfer_complete(self, _statuses):
            self.waited = True

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = True
    worker._reads_issued_this_step = []
    worker._consumer_notification_counts = {}
    worker._completed_consumer_notifications = set()
    worker.moriio_wrapper = RecordingWrapper()

    worker.start_load_kv(SimpleNamespace(transfer_id_to_request_id={}))

    assert worker._reads_issued_this_step == []
    assert worker.moriio_wrapper.async_waited is True
    assert worker.moriio_wrapper.waited is False


def test_requested_cudagraph_mode_is_never_overridden():
    # The configured cudagraph mode is always honored: the barrier fires when
    # the operator sets cudagraph_mode=PIECEWISE, and READ mode with full
    # graphs only warns instead of silently forcing PIECEWISE.
    assert (
        MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": True}) is False
    )
    assert (
        MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": False}) is False
    )
