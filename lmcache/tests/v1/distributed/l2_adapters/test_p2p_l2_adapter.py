# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the P2P L2 adapter (mocked MQ client + transfer channel)."""

# Standard
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters import p2p_l2_adapter as p2p_mod
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (
    P2PL2Adapter,
    P2PL2AdapterConfig,
)
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelReadResult,
)
from lmcache.v1.multiprocess.protocol import RequestType

_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


def _key(i: int) -> ObjectKey:
    return ObjectKey(chunk_hash=f"hash{i}".encode(), model_name="m", kv_rank=1)


class _FakeFuture:
    """Minimal MessagingFuture stand-in returning a value or raising."""

    def __init__(self, value=None, exc=None):
        self._value = value
        self._exc = exc

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._value

    def query(self) -> bool:
        return self._exc is None


@contextmanager
def _adapter(lookup_timeout_s: float = 10.0, load_timeout_s: float = 10.0):
    mq = MagicMock()
    tc_ctx = MagicMock()
    tc_client = MagicMock()
    tc_ctx.get_transfer_channel_client.return_value = tc_client
    tc_ctx.get_transfer_channel_address.side_effect = lambda pairs: [
        TransferChannelAddress(offset=o, size=s) for o, s in pairs
    ]
    notifier = MagicMock()

    with (
        patch.object(p2p_mod, "MessageQueueClient", return_value=mq),
        patch.object(p2p_mod, "get_transfer_channel_context", return_value=tc_ctx),
        patch.object(p2p_mod, "PeriodicEventNotifier") as mock_pen,
    ):
        mock_pen.get.return_value = notifier
        config = P2PL2AdapterConfig(
            peer_mq_server_url="tcp://peer:5555",
            peer_transfer_channel_server_url="peer:7600",
            lookup_timeout_s=lookup_timeout_s,
            load_timeout_s=load_timeout_s,
        )
        adapter = P2PL2Adapter(config)
        try:
            yield adapter, mq, tc_ctx, tc_client, notifier
        finally:
            adapter.close()


def _lookup_side_effect(remote_task_id, addresses):
    """submit_request dispatcher keyed on request type."""

    def _dispatch(request_type, payloads, response_cls=None):
        if request_type == RequestType.P2P_LOOKUP_AND_LOCK:
            return _FakeFuture(value=remote_task_id)
        if request_type == RequestType.P2P_QUERY_LOOKUP_RESULTS:
            return _FakeFuture(value=addresses)
        return _FakeFuture(value=None)

    return _dispatch


# ---------------------------------------------------------------------------
# Config + factory
# ---------------------------------------------------------------------------


def test_config_from_dict_roundtrip():
    config = P2PL2AdapterConfig.from_dict(
        {
            "type": "p2p",
            "peer_mq_server_url": "tcp://peer:5555",
            "peer_transfer_channel_server_url": "peer:7600",
            "lookup_timeout_s": 4,
            "load_timeout_s": 6,
        }
    )
    assert config.peer_mq_server_url == "tcp://peer:5555"
    assert config.peer_transfer_channel_server_url == "peer:7600"
    assert config.lookup_timeout_s == 4.0
    assert config.load_timeout_s == 6.0


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "p2p", "peer_transfer_channel_server_url": "peer:7600"},
        {"type": "p2p", "peer_mq_server_url": "tcp://peer:5555"},
        {
            "type": "p2p",
            "peer_mq_server_url": "tcp://peer:5555",
            "peer_transfer_channel_server_url": "peer:7600",
            "lookup_timeout_s": -1,
        },
    ],
)
def test_config_from_dict_rejects_invalid(bad):
    with pytest.raises(ValueError):
        P2PL2AdapterConfig.from_dict(bad)


def test_factory_creates_adapter():
    # First Party
    from lmcache.v1.distributed.l2_adapters import create_l2_adapter

    with (
        patch.object(p2p_mod, "MessageQueueClient", return_value=MagicMock()),
        patch.object(p2p_mod, "get_transfer_channel_context", return_value=MagicMock()),
        patch.object(p2p_mod, "PeriodicEventNotifier") as mock_pen,
    ):
        mock_pen.get.return_value = MagicMock()
        config = P2PL2AdapterConfig("tcp://peer:5555", "peer:7600")
        adapter = create_l2_adapter(config)
        assert isinstance(adapter, P2PL2Adapter)
        adapter.close()


# ---------------------------------------------------------------------------
# Event fds + notifier registration
# ---------------------------------------------------------------------------


def test_event_fds_are_distinct_and_registered():
    with _adapter() as (adapter, _mq, _tc_ctx, _tc, notifier):
        fds = {
            adapter.get_store_event_fd(),
            adapter.get_lookup_and_lock_event_fd(),
            adapter.get_load_event_fd(),
        }
        assert len(fds) == 3
        registered = {c.args[0] for c in notifier.register_fd.call_args_list}
        assert adapter.get_lookup_and_lock_event_fd() in registered
        assert adapter.get_load_event_fd() in registered
        # The store fd is a dummy and is never pulsed.
        assert adapter.get_store_event_fd() not in registered


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_lookup_forwards_layout_desc_and_returns_addresses():
    keys = [_key(0), _key(1), _key(2)]
    addresses = [
        TransferChannelAddress(offset=100, size=10),
        TransferChannelAddress(offset=200, size=20),
        TransferChannelAddress(offset=-1, size=0),  # not found on peer
    ]
    with _adapter() as (adapter, mq, _tc_ctx, _tc, _notifier):
        mq.submit_request.side_effect = _lookup_side_effect(42, addresses)

        task_id = adapter.submit_lookup_and_lock_task(keys, _LAYOUT)

        # The real layout_desc is forwarded into the lookup payload.
        lookup_call = mq.submit_request.call_args_list[0]
        assert lookup_call.args[0] == RequestType.P2P_LOOKUP_AND_LOCK
        assert lookup_call.args[1] == [keys, _LAYOUT]
        assert lookup_call.args[1][1] is _LAYOUT

        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True
        assert bitmap.test(1) is True
        assert bitmap.test(2) is False

        # Valid addresses are stashed by key for the load phase.
        assert adapter._remote_addresses[keys[0]] == addresses[0]
        assert adapter._remote_addresses[keys[1]] == addresses[1]
        assert keys[2] not in adapter._remote_addresses


def test_lookup_query_not_ready_then_ready():
    keys = [_key(0)]
    addresses = [TransferChannelAddress(offset=100, size=10)]
    with _adapter() as (adapter, mq, _tc_ctx, _tc, _notifier):
        responses = iter([_FakeFuture(value=None), _FakeFuture(value=addresses)])

        def _dispatch(request_type, payloads, response_cls=None):
            if request_type == RequestType.P2P_LOOKUP_AND_LOCK:
                return _FakeFuture(value=7)
            if request_type == RequestType.P2P_QUERY_LOOKUP_RESULTS:
                return next(responses)
            return _FakeFuture(value=None)

        mq.submit_request.side_effect = _dispatch
        task_id = adapter.submit_lookup_and_lock_task(keys, _LAYOUT)

        # First pulse: peer not ready yet.
        assert adapter.query_lookup_and_lock_result(task_id) is None
        # Second pulse: ready.
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None and bitmap.test(0) is True


def test_lookup_submit_timeout_yields_miss():
    keys = [_key(0), _key(1)]
    with _adapter() as (adapter, mq, _tc_ctx, _tc, _notifier):

        def _dispatch(request_type, payloads, response_cls=None):
            if request_type == RequestType.P2P_LOOKUP_AND_LOCK:
                return _FakeFuture(exc=TimeoutError())
            return _FakeFuture(value=None)

        mq.submit_request.side_effect = _dispatch
        task_id = adapter.submit_lookup_and_lock_task(keys, _LAYOUT)

        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0
        # No P2P_QUERY_LOOKUP_RESULTS was issued for a failed lookup.
        assert all(
            c.args[0] != RequestType.P2P_QUERY_LOOKUP_RESULTS
            for c in mq.submit_request.call_args_list
        )


def test_lookup_deadline_expired_yields_miss():
    keys = [_key(0)]
    with _adapter(lookup_timeout_s=0.01) as (adapter, mq, _tc_ctx, _tc, _notifier):
        # Lookup id resolves, but the query is never ready before the deadline.
        def _dispatch(request_type, payloads, response_cls=None):
            if request_type == RequestType.P2P_LOOKUP_AND_LOCK:
                return _FakeFuture(value=7)
            return _FakeFuture(value=None)

        mq.submit_request.side_effect = _dispatch
        task_id = adapter.submit_lookup_and_lock_task(keys, _LAYOUT)
        time.sleep(0.02)
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0


def test_query_unknown_task_returns_none():
    with _adapter() as (adapter, _mq, _tc_ctx, _tc, _notifier):
        assert adapter.query_lookup_and_lock_result(999) is None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def test_load_reads_into_local_objects():
    keys = [_key(0), _key(1)]
    with _adapter() as (adapter, _mq, _tc_ctx, tc_client, _notifier):
        adapter._remote_addresses[keys[0]] = TransferChannelAddress(offset=100, size=10)
        adapter._remote_addresses[keys[1]] = TransferChannelAddress(offset=200, size=20)
        tc_client.submit_read.return_value = 55
        tc_client.query_read_status.return_value = TransferChannelReadResult(
            finished=True, succeeded_mask=[True, False]
        )

        objects = [
            MagicMock(shm_offset=1000, shm_byte_length=10),
            MagicMock(shm_offset=2000, shm_byte_length=20),
        ]
        task_id = adapter.submit_load_task(keys, objects)

        local, remote = tc_client.submit_read.call_args.args
        assert local == [
            TransferChannelAddress(offset=1000, size=10),
            TransferChannelAddress(offset=2000, size=20),
        ]
        assert remote == [
            adapter._remote_addresses[keys[0]],
            adapter._remote_addresses[keys[1]],
        ]

        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True
        assert bitmap.test(1) is False


def test_load_not_finished_returns_none():
    keys = [_key(0)]
    with _adapter() as (adapter, _mq, _tc_ctx, tc_client, _notifier):
        adapter._remote_addresses[keys[0]] = TransferChannelAddress(offset=100, size=10)
        tc_client.submit_read.return_value = 1
        tc_client.query_read_status.return_value = TransferChannelReadResult(
            finished=False
        )
        task_id = adapter.submit_load_task(
            keys, [MagicMock(shm_offset=0, shm_byte_length=10)]
        )
        assert adapter.query_load_result(task_id) is None


def test_load_deadline_expired_yields_failure():
    keys = [_key(0)]
    with _adapter(load_timeout_s=0.01) as (adapter, _mq, _tc_ctx, tc_client, _notifier):
        adapter._remote_addresses[keys[0]] = TransferChannelAddress(offset=100, size=10)
        tc_client.submit_read.return_value = 1
        task_id = adapter.submit_load_task(
            keys, [MagicMock(shm_offset=0, shm_byte_length=10)]
        )
        time.sleep(0.02)
        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0


# ---------------------------------------------------------------------------
# Unlock + store no-ops + lifecycle
# ---------------------------------------------------------------------------


def test_unlock_sends_rpc_and_clears_addresses():
    keys = [_key(0), _key(1)]
    with _adapter() as (adapter, mq, _tc_ctx, _tc, _notifier):
        adapter._remote_addresses[keys[0]] = TransferChannelAddress(offset=1, size=1)
        adapter.submit_unlock(keys)
        unlock_call = mq.submit_request.call_args
        assert unlock_call.args[0] == RequestType.P2P_UNLOCK_OBJECTS
        assert unlock_call.args[1] == [keys]
        assert keys[0] not in adapter._remote_addresses


def test_unlock_empty_is_noop():
    with _adapter() as (adapter, mq, _tc_ctx, _tc, _notifier):
        adapter.submit_unlock([])
        mq.submit_request.assert_not_called()


def test_store_completes_immediately_without_leaking():
    with _adapter() as (adapter, _mq, _tc_ctx, _tc, _notifier):
        task_id = adapter.submit_store_task([_key(0)], [MagicMock()])
        completed = adapter.pop_completed_store_tasks()
        assert set(completed) == {task_id}
        assert completed[task_id].is_successful() is True
        assert completed[task_id].bytes_transferred() == 0
        # Draining clears the completed set.
        assert adapter.pop_completed_store_tasks() == {}


def test_load_missing_address_is_failure_not_raise():
    keys = [_key(0)]
    with _adapter() as (adapter, _mq, _tc_ctx, tc_client, _notifier):
        # No lookup ran, so no remote address is stashed for this key.
        task_id = adapter.submit_load_task(
            keys, [MagicMock(shm_offset=0, shm_byte_length=10)]
        )
        tc_client.submit_read.assert_not_called()
        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0


def test_load_invalid_address_is_failure_not_raise():
    keys = [_key(0)]
    with _adapter() as (adapter, _mq, _tc_ctx, tc_client, _notifier):
        # An invalid stashed address must be rejected, not read.
        adapter._remote_addresses[keys[0]] = TransferChannelAddress(offset=-1, size=0)
        task_id = adapter.submit_load_task(
            keys, [MagicMock(shm_offset=0, shm_byte_length=10)]
        )
        tc_client.submit_read.assert_not_called()
        bitmap = adapter.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.popcount() == 0


def test_report_status():
    with _adapter() as (adapter, _mq, _tc_ctx, _tc, _notifier):
        status = adapter.report_status()
        assert status["is_healthy"] is True
        assert status["peer_mq_server_url"] == "tcp://peer:5555"
        assert status["peer_transfer_channel_server_url"] == "peer:7600"
        assert status["in_flight_lookups"] == 0
        assert status["in_flight_loads"] == 0


def test_close_unregisters_fds_and_closes_client():
    with _adapter() as (adapter, mq, tc_ctx, _tc, notifier):
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()
        adapter.close()
        unregistered = {c.args[0] for c in notifier.unregister_fd.call_args_list}
        assert lookup_fd in unregistered
        assert load_fd in unregistered
        mq.close.assert_called()
        # The transfer-channel client is dropped from the context on close.
        tc_ctx.remove_transfer_channel_client.assert_called_once_with("peer:7600")


def test_close_is_idempotent():
    with _adapter() as (adapter, mq, tc_ctx, _tc, _notifier):
        adapter.close()
        adapter.close()
        # The second close is a no-op: no duplicate teardown of shared resources.
        mq.close.assert_called_once()
        tc_ctx.remove_transfer_channel_client.assert_called_once_with("peer:7600")
