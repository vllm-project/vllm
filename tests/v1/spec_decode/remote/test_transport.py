# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline stream transport: endpoint parsing, framing, deadlines, and
integer frame helpers."""

import errno
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from unittest.mock import Mock, call

import msgspec
import pytest

from vllm.v1.spec_decode.remote import transport
from vllm.v1.spec_decode.remote.transport import (
    FRAME_CONTROL,
    FRAME_DATA,
    MAX_FRAME_BYTES,
    ConnectionClosed,
    DataFrame,
    FramedConnection,
    TransportError,
    TransportTimeout,
    connect,
    decode_data_frame,
    frame_to_ints,
    ints_to_frame,
    is_local_endpoint,
    listen,
    parse_endpoint,
    validate_frame,
)


@pytest.fixture
def pair():
    listener = listen("tcp://127.0.0.1:0")
    client = connect(listener.endpoint, timeout=5.0)
    server = listener.accept(timeout=5.0)
    yield client, server
    client.close()
    server.close()
    listener.close()


@pytest.fixture
def close_error_pair():
    class CloseErrorSocket(socket.socket):
        def close(self):
            super().close()
            raise OSError(errno.EIO, "injected socket cleanup failure")

    left, peer = socket.socketpair()
    conn = FramedConnection(CloseErrorSocket(fileno=left.detach()))
    try:
        yield conn, peer
    finally:
        if not conn.closed:
            with pytest.raises(OSError, match="injected socket cleanup failure"):
                conn.close()
        peer.close()


@pytest.mark.parametrize("duplex", [False, True])
def test_send_cleanup_error_preserves_disconnect(close_error_pair, duplex):
    conn, peer = close_error_pair
    peer.close()
    with pytest.raises(ConnectionClosed, match="send failed") as failure:
        conn.send_control(b"request", drain_incoming=duplex)
    assert isinstance(failure.value.__cause__, OSError)
    assert failure.value.__cause__.errno != errno.EIO
    assert conn.closed


@pytest.mark.parametrize("duplex", [False, True])
def test_send_cleanup_error_preserves_timeout(close_error_pair, duplex):
    conn, _ = close_error_pair
    conn._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    with pytest.raises(TransportTimeout, match="timed out sending"):
        conn.send_control(
            bytes(1 << 20),
            deadline=time.monotonic() + 0.1,
            drain_incoming=duplex,
        )
    assert conn.closed


def test_receive_cleanup_error_preserves_frame_rejection(close_error_pair):
    conn, peer = close_error_pair
    peer.sendall(struct.pack("!BI", 0xFF, 0))
    with pytest.raises(TransportError, match="unknown frame kind"):
        conn.recv(timeout=1)
    assert conn.closed


def test_receive_cleanup_error_preserves_partial_frame_timeout(close_error_pair):
    conn, peer = close_error_pair
    conn.set_receive_limits(partial_frame_timeout=0.01)
    peer.sendall(struct.pack("!BI", FRAME_CONTROL, 1))
    with pytest.raises(TransportTimeout, match="started frame exceeded"):
        conn.recv(timeout=1)
    assert conn.closed


def test_parse_tcp_endpoint():
    assert parse_endpoint("tcp://127.0.0.1:5555") == (
        socket.AF_INET,
        ("127.0.0.1", 5555),
    )


def test_parse_ipv6_tcp_endpoint():
    assert parse_endpoint("tcp://[::1]:5555") == (socket.AF_INET6, ("::1", 5555))


def test_ipv6_listener_round_trip():
    try:
        listener = listen("tcp://[::1]:0")
    except TransportError:
        pytest.skip("IPv6 loopback unavailable")
    try:
        assert listener.endpoint.startswith("tcp://[::1]:")
        client = connect(listener.endpoint, timeout=5.0)
        server = listener.accept(timeout=5.0)
        client.send_control(b"v6")
        assert server.recv(timeout=5.0).body == b"v6"
        client.close()
        server.close()
    finally:
        listener.close()


@pytest.mark.parametrize(
    ("endpoint", "local"),
    [
        ("tcp://127.0.0.1:1", True),
        ("tcp://[::1]:1", True),
        ("tcp://localhost:1", True),
        ("tcp://0.0.0.0:1", False),
        ("tcp://10.0.0.5:1", False),
        ("tcp://draft-host:1", False),
        ("unix:///run/vllm/draft.sock", True),
    ],
)
def test_is_local_endpoint(endpoint, local):
    if endpoint.startswith("unix") and not hasattr(socket, "AF_UNIX"):
        pytest.skip("unix sockets unavailable")
    assert is_local_endpoint(endpoint) is local


def test_parse_unix_endpoint():
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("unix sockets unavailable")
    assert parse_endpoint("unix:///run/vllm/draft.sock") == (
        socket.AF_UNIX,
        "/run/vllm/draft.sock",
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:1",
        "tcp://127.0.0.1",
        "tcp://127.0.0.1:99999",
        "tcp://:5555",
        "unix://",
        "draft",
    ],
)
def test_parse_rejects_bad_endpoints(endpoint):
    with pytest.raises(ValueError):
        parse_endpoint(endpoint)


def test_listen_resolves_ephemeral_port():
    listener = listen("tcp://127.0.0.1:0")
    try:
        assert listener.endpoint.startswith("tcp://127.0.0.1:")
        assert not listener.endpoint.endswith(":0")
    finally:
        listener.close()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="unix sockets unavailable")
def test_unix_listener_refuses_live_socket(tmp_path):
    path = tmp_path / "draft.sock"
    listener = listen(f"unix://{path}")
    try:
        with pytest.raises(TransportError, match="cannot listen"):
            listen(listener.endpoint)
        client = connect(listener.endpoint, timeout=1.0)
        server = listener.accept(timeout=1.0)
        try:
            client.send_control(b"still listening")
            assert server.recv(timeout=1.0).body == b"still listening"
        finally:
            client.close()
            server.close()
    finally:
        listener.close()
    assert not path.exists()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="unix sockets unavailable")
def test_unix_listener_preserves_existing_file(tmp_path):
    path = tmp_path / "draft.sock"
    path.write_text("keep")
    with pytest.raises(TransportError, match="cannot listen"):
        listen(f"unix://{path}")
    assert path.read_text() == "keep"


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="unix sockets unavailable")
@pytest.mark.parametrize(
    "replace_live", [True, False], ids=["live-path", "closed-path"]
)
def test_unix_close_preserves_replacement_listener(tmp_path, replace_live):
    path = tmp_path / "draft.sock"
    listener = listen(f"unix://{path}")
    try:
        if replace_live:
            path.unlink()
        else:
            listener.close()
        replacement = listen(listener.endpoint)
        try:
            listener.close()
            client = connect(replacement.endpoint, timeout=1.0)
            server = replacement.accept(timeout=1.0)
            client.close()
            server.close()
        finally:
            replacement.close()
    finally:
        listener.close()
    assert not path.exists()


def test_int_frame_roundtrip():
    frame = ints_to_frame(3, [1, -2, 3, 4], dtype="int64", shape=(2, 2))
    assert frame.slot == 3
    assert frame.shape == (2, 2)
    assert frame_to_ints(frame) == [1, -2, 3, 4]


def test_int_frame_rejects_shape_and_dtype_mismatch():
    with pytest.raises(ValueError):
        ints_to_frame(0, [1, 2, 3], shape=(2,))
    with pytest.raises(TransportError, match="not an integer"):
        ints_to_frame(0, [1], dtype="bfloat16")
    truncated = DataFrame(slot=0, dtype="int32", shape=(2,), payload=b"\x00" * 7)
    with pytest.raises(TransportError, match="payload bytes"):
        frame_to_ints(truncated)


@pytest.mark.parametrize(
    ("dtype", "itemsize"),
    [
        ("int32", 4),
        ("int64", 8),
        ("float16", 2),
        ("bfloat16", 2),
        ("float32", 4),
        ("float64", 8),
    ],
)
def test_frame_validation_checks_payload_size(dtype, itemsize):
    validate_frame(DataFrame(0, dtype, (2, 3), bytes(6 * itemsize)))
    for length in (6 * itemsize - 1, 6 * itemsize + 1):
        with pytest.raises(TransportError, match="payload bytes"):
            validate_frame(DataFrame(0, dtype, (2, 3), bytes(length)))


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (DataFrame(0, "unknown", (1,), b"\x00"), "unknown dtype"),
        (DataFrame(0, "int32", (-1, -1), bytes(4)), "negative shape"),
        (DataFrame(0, "float32", (0, 3), bytes(4)), "payload bytes"),
    ],
)
def test_frame_validation_rejects_invalid_metadata(frame, match):
    with pytest.raises(TransportError, match=match):
        validate_frame(frame)


def test_control_and_data_frames_roundtrip(pair):
    client, server = pair
    client.send_control(b"hello")
    client.send_data(ints_to_frame(7, [5, 6], dtype="int32"))
    client.send_control(b"")

    frame = server.recv(timeout=5.0)
    assert (frame.kind, frame.body) == (FRAME_CONTROL, b"hello")
    frame = server.recv(timeout=5.0)
    assert frame.kind == FRAME_DATA
    data = decode_data_frame(frame.body)
    assert (data.slot, frame_to_ints(data)) == (7, [5, 6])
    frame = server.recv(timeout=5.0)
    assert (frame.kind, frame.body) == (FRAME_CONTROL, b"")


@pytest.mark.parametrize("numel", [0, 63, 64, 16383, 16384])
def test_data_frame_byte_budget_includes_metadata_before_encoding(
    monkeypatch, pair, numel
):
    frame = DataFrame((1 << 64) - 1, "int32", (numel,), bytes(4 * numel))
    encoded = msgspec.msgpack.encode(frame)
    client, server = pair
    monkeypatch.setattr(transport, "MAX_FRAME_BYTES", len(encoded))
    client.send_data(frame)
    assert server.recv(timeout=5.0).body == encoded

    original_encoder = transport._frame_encoder

    def encode_metadata_only(value):
        assert not value.payload, "oversized payload was encoded"
        return original_encoder.encode(value)

    monkeypatch.setattr(transport, "MAX_FRAME_BYTES", len(encoded) - 1)
    monkeypatch.setattr(transport, "_frame_encoder", Mock(encode=encode_metadata_only))
    with pytest.raises(TransportError, match="exceeds"):
        client.send_data(frame)
    assert not client.closed


def test_tensor_frame_byte_budget_rejects_before_tensor_copy(monkeypatch):
    import torch

    from vllm.v1.spec_decode.remote.session import tensor_to_frame

    # A meta tensor cannot be copied to CPU, and requires no large allocation.
    tensor = torch.empty((512, 3), dtype=torch.bfloat16, device="meta")
    monkeypatch.setattr(transport, "MAX_FRAME_BYTES", 1024)
    with pytest.raises(TransportError, match="exceeds"):
        tensor_to_frame(0, tensor)


@pytest.mark.parametrize("drain_incoming", [False, True])
@pytest.mark.timeout(5)
def test_send_deadline_closes_connection_when_peer_does_not_read(drain_incoming):
    sock, peer = socket.socketpair()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    connection = FramedConnection(sock)
    try:
        started = time.monotonic()
        with pytest.raises(TransportTimeout, match="sending a frame"):
            connection.send_data(
                DataFrame(0, "int32", (1 << 18,), bytes(1 << 20)),
                deadline=started + 0.05,
                drain_incoming=drain_incoming,
            )
        assert time.monotonic() - started < 2.0
        assert connection.closed
        with pytest.raises(ConnectionClosed):
            connection.send_control(b"cannot reuse partial write")
    finally:
        connection.close()
        peer.close()


def test_send_frames_share_absolute_deadline(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(transport, "time", Mock(monotonic=lambda: now[0]))
    sock, peer_sock = socket.socketpair()
    sock.settimeout(1)
    peer = FramedConnection(peer_sock)

    class MeteredSocket:
        """Charge virtual time per byte, independent of send coalescing."""

        def settimeout(self, timeout):
            self.remaining = timeout

        def sendall(self, data):
            available = len(data) if self.remaining is None else self.remaining
            sent = min(len(data), int(available))
            if sent:
                sock.sendall(data[:sent])
            now[0] += min(len(data), available)
            if sent != len(data):
                raise TimeoutError

        def shutdown(self, how):
            sock.shutdown(how)

        def close(self):
            sock.close()

    connection = FramedConnection(MeteredSocket())
    try:
        deadline = 25.0
        connection.send_control(b"first", deadline=deadline)
        with pytest.raises(TransportTimeout):
            connection.send_data(ints_to_frame(0, [1]), deadline=deadline)
        assert now[0] == deadline
        assert connection.closed
        assert peer.recv(timeout=1).body == b"first"
        with pytest.raises(ConnectionClosed):
            peer.recv(timeout=1)
    finally:
        connection.close()
        peer.close()


@pytest.mark.parametrize("drain_incoming", [False, True])
def test_expired_deadline_before_frame_keeps_stream_usable(pair, drain_incoming):
    # Nothing reached the wire, so frame boundaries are intact and the
    # connection must not be torn down.
    client, server = pair
    with pytest.raises(TransportTimeout, match="before sending"):
        client.send_control(
            b"too late", deadline=time.monotonic() - 1.0, drain_incoming=drain_incoming
        )
    assert not client.closed
    client.send_control(b"next")
    assert server.recv(timeout=1.0).body == b"next"


@pytest.mark.timeout(10)
def test_duplex_send_buffers_large_reply_before_peer_reads_request():
    sockets = socket.socketpair()
    for sock in sockets:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    client, server = (FramedConnection(sock) for sock in sockets)
    request, reply = b"q" * (1 << 20), b"a" * (1 << 20)

    def exchange():
        server.send_control(reply, deadline=time.monotonic() + 5)
        return server.recv(timeout=5).body

    with ThreadPoolExecutor(max_workers=1) as pool:
        received = pool.submit(exchange)
        try:
            client.send_control(
                request, deadline=time.monotonic() + 5, drain_incoming=True
            )
            assert client.recv(timeout=5).body == reply
            assert received.result(timeout=5) == request
        finally:
            client.close()
            server.close()


@pytest.mark.parametrize("close_at", ["register", "select"])
@pytest.mark.timeout(5)
def test_duplex_send_close_races_raise_connection_closed(monkeypatch, close_at):
    sock, peer = socket.socketpair()
    connection = FramedConnection(sock)
    selector = transport.selectors.DefaultSelector()
    original = getattr(selector, close_at)

    def close_then_call(*args, **kwargs):
        connection.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(selector, close_at, close_then_call)
    monkeypatch.setattr(transport.selectors, "DefaultSelector", lambda: selector)
    try:
        with pytest.raises(ConnectionClosed):
            connection.send_control(b"request", drain_incoming=True)
    finally:
        connection.close()
        peer.close()


def test_send_without_deadline_resets_previous_receive_timeout(pair):
    client, server = pair
    with pytest.raises(TransportTimeout):
        client.recv(timeout=0.01)
    client.send_control(b"no send deadline")
    assert client._sock.gettimeout() is None
    assert server.recv(timeout=1.0).body == b"no send deadline"


def test_send_failure_closes_connection():
    sock = Mock(spec=socket.socket)
    sock.sendall.side_effect = BrokenPipeError("peer disconnected")
    connection = FramedConnection(sock)
    with pytest.raises(ConnectionClosed, match="send failed"):
        connection.send_control(b"cannot deliver")
    assert connection.closed
    sock.close.assert_called_once()


@pytest.mark.parametrize("duplex", [False, True], ids=["recv", "duplex-send"])
def test_receive_allocation_failure_fences_consumed_frame(monkeypatch, duplex):
    error = MemoryError("injected receive allocation")

    class FailedBuffer(bytearray):
        def extend(self, data):
            raise error

    sock = Mock(spec=socket.socket)
    sock.recv.return_value = struct.pack("!BI", FRAME_CONTROL, 2)
    sock.send.side_effect = len
    with monkeypatch.context() as patch:
        patch.setattr("builtins.bytearray", FailedBuffer)
        connection = FramedConnection(sock)
    if duplex:
        selector = Mock()
        selector.select.side_effect = [
            [(None, transport.selectors.EVENT_WRITE)],
            [(None, transport.selectors.EVENT_READ)],
        ]
        monkeypatch.setattr(
            transport.selectors, "DefaultSelector", lambda: nullcontext(selector)
        )
    with pytest.raises(MemoryError) as caught:
        if duplex:
            connection.send_control(b"request", drain_incoming=True)
        else:
            connection.recv(timeout=0.1)
    assert caught.value is error
    sock.recv.assert_called_once_with(5)
    if duplex:
        assert bytes(sock.send.call_args.args[0]) == struct.pack(
            "!BI", FRAME_CONTROL, 7
        )
    assert connection.closed
    sock.close.assert_called_once()
    with pytest.raises(ConnectionClosed):
        connection.send_control(b"cannot reuse a consumed header")
    with pytest.raises(ConnectionClosed):
        connection.recv(timeout=0.1)


@pytest.mark.parametrize("close_error", [None, OSError("injected close failure")])
def test_send_allocation_failure_after_header_preserves_error(close_error):
    error = MemoryError("injected body send allocation")
    sock = Mock(spec=socket.socket)
    sock.sendall.side_effect = [None, error]
    sock.close.side_effect = close_error
    connection = FramedConnection(sock)
    with pytest.raises(MemoryError) as caught:
        connection.send_control(b"request")
    assert caught.value is error
    assert sock.sendall.call_args_list == [
        call(struct.pack("!BI", FRAME_CONTROL, 7)),
        call(b"request"),
    ]
    assert connection.closed
    sock.close.assert_called_once()


@pytest.mark.parametrize("stage", ["metadata", "payload", "header"])
def test_send_serialization_allocation_failure_preserves_stream(monkeypatch, stage):
    error = MemoryError("injected serialization allocation")
    sock = Mock(spec=socket.socket)
    connection = FramedConnection(sock)
    frame = ints_to_frame(0, [1])
    with monkeypatch.context() as patch:
        if stage == "header":
            patch.setattr(transport, "_HEADER", Mock(pack=Mock(side_effect=error)))
        else:
            results = [error] if stage == "metadata" else [b"metadata", error]
            patch.setattr(
                transport, "_frame_encoder", Mock(encode=Mock(side_effect=results))
            )
        with pytest.raises(MemoryError) as caught:
            connection.send_data(frame)
    assert caught.value is error
    sock.sendall.assert_not_called()
    assert not connection.closed
    connection.send_control(b"next")
    assert sock.sendall.call_args_list == [
        call(struct.pack("!BI", FRAME_CONTROL, 4)),
        call(b"next"),
    ]
    connection.close()


def test_partial_frame_survives_timeout():
    listener = listen("tcp://127.0.0.1:0")
    _, address = parse_endpoint(listener.endpoint)
    raw = socket.create_connection(address, timeout=5.0)
    server = listener.accept(timeout=5.0)
    try:
        raw.sendall(struct.pack("!BI", FRAME_CONTROL, 4) + b"ab")
        with pytest.raises(TransportTimeout):
            server.recv(timeout=0.05)
        raw.sendall(b"cd")
        frame = server.recv(timeout=5.0)
        assert frame.body == b"abcd"
    finally:
        raw.close()
        server.close()
        listener.close()


@pytest.mark.parametrize(
    ("header", "match"),
    [
        (struct.pack("!BI", 0x7F, 0), "unknown frame kind"),
        (struct.pack("!BI", FRAME_CONTROL, MAX_FRAME_BYTES + 1), "exceeds"),
    ],
    ids=["unknown-kind", "oversized"],
)
def test_bad_frame_headers_rejected(header, match):
    listener = listen("tcp://127.0.0.1:0")
    _, address = parse_endpoint(listener.endpoint)
    raw = socket.create_connection(address, timeout=5.0)
    server = listener.accept(timeout=5.0)
    try:
        raw.sendall(header)
        with pytest.raises(TransportError, match=match):
            server.recv(timeout=5.0)
        # A rejected header poisons the stream; no further body is accepted.
        assert server.closed
        with pytest.raises(ConnectionClosed):
            server.recv(timeout=5.0)
    finally:
        raw.close()
        server.close()
        listener.close()


def test_peer_close_raises_connection_closed(pair):
    client, server = pair
    client.close()
    with pytest.raises(ConnectionClosed):
        server.recv(timeout=5.0)
    with pytest.raises(ConnectionClosed):
        client.send_control(b"late")


def test_close_unblocks_pending_recv(pair):
    _, server = pair
    outcome: list[BaseException | None] = []

    def reader():
        try:
            server.recv()
        except BaseException as e:  # noqa: BLE001 - recorded for the assert
            outcome.append(e)
        else:
            outcome.append(None)

    thread = threading.Thread(target=reader)
    thread.start()
    server.close()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert isinstance(outcome[0], ConnectionClosed)


@pytest.mark.parametrize("kind", [FRAME_CONTROL, FRAME_DATA])
def test_receive_rejects_oversized_header_before_body(kind):
    raw, sock = socket.socketpair()
    conn = FramedConnection(sock)
    conn.set_receive_limits(
        max_control_bytes=32, max_data_bytes=32, max_buffered_bytes=128
    )
    try:
        raw.sendall(struct.pack("!BI", kind, 33))
        with pytest.raises(TransportError, match="exceeds 32"):
            conn.recv(timeout=0.5)
        assert conn.closed
        assert len(conn._buffer) == 5
    finally:
        raw.close()
        conn.close()


@pytest.mark.parametrize("overflow", ["frame", "aggregate"])
def test_duplex_receive_rejects_overflow_before_reading_body(overflow):
    raw, sock = socket.socketpair()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    conn = FramedConnection(sock)
    conn.set_receive_limits(
        max_control_bytes=32, max_data_bytes=32, max_buffered_bytes=64
    )
    wire = (
        struct.pack("!BI", FRAME_DATA, 33)
        if overflow == "frame"
        else struct.pack("!BI", FRAME_CONTROL, 30)
        + b"x" * 30
        + struct.pack("!BI", FRAME_CONTROL, 30)
    )
    try:
        raw.sendall(wire)
        with pytest.raises(TransportError, match="exceeds|budget"):
            conn.send_control(
                b"x" * (1 << 20), drain_incoming=True, deadline=time.monotonic() + 1
            )
        assert conn.closed
        assert len(conn._buffer) <= 40
    finally:
        raw.close()
        conn.close()


@pytest.mark.parametrize("partial", [b"\x01", struct.pack("!BI", FRAME_DATA, 8) + b"x"])
def test_started_frame_timeout_closes_connection_but_idle_timeout_does_not(partial):
    raw, sock = socket.socketpair()
    conn = FramedConnection(sock)
    conn.set_receive_limits(partial_frame_timeout=0.03)
    try:
        with pytest.raises(TransportTimeout):
            conn.recv(timeout=0.04)
        assert not conn.closed
        raw.sendall(partial)
        with pytest.raises(TransportTimeout, match="started frame"):
            conn.recv()
        assert conn.closed
    finally:
        raw.close()
        conn.close()


def test_per_call_limit_rejects_body_before_receiving():
    raw, sock = socket.socketpair()
    conn = FramedConnection(sock)
    try:
        raw.sendall(struct.pack("!BI", FRAME_DATA, 128))
        with pytest.raises(TransportError, match="exceeds 16"):
            conn.recv(timeout=0.5, max_frame_bytes=16)
        assert conn.closed
        assert len(conn._buffer) == 5
    finally:
        raw.close()
        conn.close()


def test_receive_limit_revalidation_keeps_partial_frame_age():
    raw, sock = socket.socketpair()
    conn = FramedConnection(sock)
    try:
        # Duplex progress can leave complete replies ahead of an incomplete
        # frame. Revalidating those replies must not reset the incomplete age.
        conn._append_received(struct.pack("!BI", FRAME_CONTROL, 1))
        conn._append_received(b"x")
        conn._append_received(struct.pack("!BI", FRAME_DATA, 8))
        conn._append_received(b"x")
        started = conn._partial_started
        assert started is not None
        conn.set_receive_limits(partial_frame_timeout=0.03)
        assert conn._partial_started == started
        assert conn.recv(timeout=0.5).body == b"x"
        with pytest.raises(TransportTimeout, match="started frame"):
            conn.recv()
        assert conn.closed
    finally:
        conn.close()
        raw.close()
