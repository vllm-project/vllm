# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for control-layer ABCs, ZmqClientConnection, and ECSession."""

from unittest.mock import MagicMock, patch

import pytest
import zmq

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.base import (
    ControlConnection,
)


class _ConcreteConn(ControlConnection):
    """Minimal concrete subclass for ABC-instantiation tests."""

    def __init__(self):
        self._alive = True
        self._sent = []
        self._inbox = []

    @property
    def alive(self) -> bool:
        return self._alive

    def send(self, msg: bytes) -> None:
        self._sent.append(msg)

    def recv(self) -> list[bytes]:
        msgs, self._inbox = self._inbox, []
        return msgs

    def mark_dead(self) -> None:
        self._alive = False

    def close(self) -> None:
        self._alive = False


def test_control_connection_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ControlConnection()  # type: ignore[abstract]


def test_concrete_connection_send_recv():
    conn = _ConcreteConn()
    assert conn.alive
    conn.send(b"hello")
    conn._inbox.append(b"world")
    assert conn.recv() == [b"world"]
    assert conn._sent == [b"hello"]


def test_mark_dead_sets_alive_false():
    conn = _ConcreteConn()
    conn.mark_dead()
    assert not conn.alive


def test_close_sets_alive_false():
    conn = _ConcreteConn()
    conn.close()
    assert not conn.alive


# ── ZmqClientConnection ───────────────────────────────────────────────────────


from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq import (  # noqa: E402
    ZmqClientConnection,
    ZmqClientTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import PeerAddr  # noqa: E402


def _conn(messages: list[list[bytes]], monitor=None) -> ZmqClientConnection:
    """Build a ZmqClientConnection whose dealer yields the given frame-lists."""
    dealer = MagicMock()
    dealer.recv_multipart.side_effect = [*messages, zmq.Again()]
    return ZmqClientConnection(dealer=dealer, monitor=monitor)


def test_zmq_client_connection_recv_extracts_payloads_from_envelopes():
    conn = _conn([[b"", b"msg1"], [b"", b"msg2"]])
    assert conn.recv() == [b"msg1", b"msg2"]


def test_zmq_client_connection_recv_skips_frames_without_empty_delimiter():
    # A frame that does not start with b"" is malformed and must be dropped.
    conn = _conn([[b"", b"ok"], [b"no-delimiter", b"bad"]])
    assert conn.recv() == [b"ok"]


def test_zmq_client_connection_recv_breaks_on_non_again_exception():
    # A socket error should stop the drain and return whatever was collected.
    dealer = MagicMock()
    dealer.recv_multipart.side_effect = [[b"", b"first"], OSError("broken")]
    conn = ZmqClientConnection(dealer=dealer, monitor=None)
    assert conn.recv() == [b"first"]


def test_zmq_client_connection_send_prefixes_empty_delimiter():
    dealer = MagicMock()
    conn = ZmqClientConnection(dealer=dealer, monitor=None)
    conn.send(b"payload")
    dealer.send_multipart.assert_called_once_with([b"", b"payload"])


def test_zmq_client_connection_mark_dead_does_not_close_socket():
    # mark_dead signals liveness only; resource teardown is close()'s job.
    dealer = MagicMock()
    conn = ZmqClientConnection(dealer=dealer, monitor=None)
    conn.mark_dead()
    assert not conn.alive
    dealer.close.assert_not_called()


def test_zmq_client_connection_close_without_monitor():
    dealer = MagicMock()
    conn = ZmqClientConnection(dealer=dealer, monitor=None)
    conn.close()
    dealer.close.assert_called_once_with(linger=0)
    assert not conn.alive


def test_zmq_client_connection_close_with_monitor_disables_then_closes_both():
    # The disable_monitor → monitor.close → dealer.close sequence must be
    # respected: closing the dealer before disabling the monitor can cause
    # a zmq assertion in some versions.
    dealer = MagicMock()
    monitor = MagicMock()
    conn = ZmqClientConnection(dealer=dealer, monitor=monitor)
    conn.close()
    dealer.disable_monitor.assert_called_once()
    monitor.close.assert_called_once_with(linger=0)
    dealer.close.assert_called_once_with(linger=0)
    assert not conn.alive


def test_zmq_client_connection_close_is_idempotent():
    dealer = MagicMock()
    conn = ZmqClientConnection(dealer=dealer, monitor=None)
    conn.close()
    conn.close()  # must not raise
    assert not conn.alive


# ── ZmqClientTransport ────────────────────────────────────────────────────────


def _make_transport() -> tuple[ZmqClientTransport, MagicMock]:
    """Return a ZmqClientTransport with a mocked ZMQ context."""
    ctx = MagicMock()
    ctx.socket.return_value = MagicMock()
    with (
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.zmq.Context",
            return_value=ctx,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_socket",
            return_value=MagicMock(),
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_path",
            return_value="tcp://host:0",
        ),
    ):
        transport = ZmqClientTransport()
    transport._ctx = ctx
    return transport, ctx


def _inject_conn(transport: ZmqClientTransport, addr: PeerAddr) -> ZmqClientConnection:
    """Inject a ZmqClientConnection directly into the transport pool."""
    conn = ZmqClientConnection(dealer=MagicMock(), monitor=MagicMock())
    transport._connections[addr] = conn
    return conn


@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_socket"
)
@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_path"
)
@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.zmq.Context"
)
def test_zmq_client_transport_connect_caches_connection(
    mock_ctx_cls, mock_path, mock_sock
):
    mock_ctx_cls.return_value = MagicMock()
    mock_ctx_cls.return_value.socket.return_value = MagicMock()
    mock_sock.return_value = MagicMock()
    transport = ZmqClientTransport()
    addr: PeerAddr = ("host", 1234)
    assert transport.connect(addr) is transport.connect(addr)


@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_socket"
)
@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_path"
)
@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.zmq.Context"
)
def test_zmq_client_transport_connect_creates_independent_conns_per_addr(
    mock_ctx_cls, mock_path, mock_sock
):
    mock_ctx_cls.return_value = MagicMock()
    mock_ctx_cls.return_value.socket.return_value = MagicMock()
    mock_sock.return_value = MagicMock()
    transport = ZmqClientTransport()
    conn_a = transport.connect(("host-a", 1111))
    conn_b = transport.connect(("host-b", 2222))
    assert conn_a is not conn_b


@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.recv_monitor_message"
)
def test_zmq_client_transport_poll_dead_detects_disconnected_peer(mock_recv_mon):
    transport, _ = _make_transport()
    addr: PeerAddr = ("host", 1234)
    conn = _inject_conn(transport, addr)
    mock_recv_mon.side_effect = [{"event": zmq.EVENT_DISCONNECTED}, zmq.Again()]

    dead = transport.poll_dead()

    assert addr in dead
    assert not conn.alive


@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.recv_monitor_message"
)
def test_zmq_client_transport_poll_dead_skips_conn_without_monitor(mock_recv_mon):
    transport, _ = _make_transport()
    addr: PeerAddr = ("host", 9999)
    conn = ZmqClientConnection(dealer=MagicMock(), monitor=None)
    transport._connections[addr] = conn

    dead = transport.poll_dead()

    assert dead == []
    mock_recv_mon.assert_not_called()


@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.recv_monitor_message"
)
def test_zmq_client_transport_poll_dead_ignores_non_disconnect_events(mock_recv_mon):
    transport, _ = _make_transport()
    addr: PeerAddr = ("host", 1234)
    conn = _inject_conn(transport, addr)
    # A CONNECTED event should not trigger a dead notification.
    mock_recv_mon.side_effect = [{"event": zmq.EVENT_CONNECTED}, zmq.Again()]

    dead = transport.poll_dead()

    assert dead == []
    assert conn.alive


def test_zmq_client_transport_remove_closes_and_evicts_conn():
    transport, _ = _make_transport()
    addr: PeerAddr = ("host", 5678)
    conn = _inject_conn(transport, addr)

    removed = transport.remove(addr)

    assert removed is conn
    assert addr not in transport._connections
    conn.dealer.close.assert_called()


def test_zmq_client_transport_remove_returns_none_for_unknown_addr():
    transport, _ = _make_transport()
    assert transport.remove(("ghost", 0)) is None


@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_socket"
)
@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.make_zmq_path"
)
@patch(
    "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq.zmq.Context"
)
def test_zmq_client_transport_close_closes_all_connections(
    mock_ctx_cls, mock_path, mock_sock
):
    mock_ctx = MagicMock()
    mock_ctx_cls.return_value = mock_ctx
    mock_ctx.socket.return_value = MagicMock()
    dealer_a, dealer_b = MagicMock(), MagicMock()
    mock_sock.side_effect = [dealer_a, dealer_b]
    transport = ZmqClientTransport()
    transport.connect(("a", 1))
    transport.connect(("b", 2))
    transport.close()
    dealer_a.close.assert_called()
    dealer_b.close.assert_called()
    assert transport._connections == {}
