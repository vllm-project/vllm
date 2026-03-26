# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
import threading

import pytest
import zmq

from vllm.utils.network_utils import (
    bind_zmq_socket_and_get_address,
    get_open_port,
    get_open_ports_list,
    get_tcp_uri,
    is_wildcard_addr,
    join_host_port,
    make_zmq_path,
    make_zmq_socket,
    split_host_port,
    split_zmq_path,
)


def test_get_open_port(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", "5678")
        # make sure we can get multiple ports, even if the env var is set
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
            s1.bind(("localhost", get_open_port()))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                s2.bind(("localhost", get_open_port()))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s3:
                    s3.bind(("localhost", get_open_port()))


def test_get_open_ports_list_with_vllm_port(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", "5678")
        ports = get_open_ports_list(5)
        assert len(ports) == 5
        assert len(set(ports)) == 5, "ports must be unique"

        # verify every port is actually bindable
        sockets = []
        try:
            for p in ports:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("localhost", p))
                sockets.append(s)
        finally:
            for s in sockets:
                s.close()


@pytest.mark.parametrize(
    "path,expected",
    [
        ("ipc://some_path", ("ipc", "some_path", "")),
        ("tcp://127.0.0.1:5555", ("tcp", "127.0.0.1", "5555")),
        ("tcp://[::1]:5555", ("tcp", "::1", "5555")),  # IPv6 address
        ("inproc://some_identifier", ("inproc", "some_identifier", "")),
    ],
)
def test_split_zmq_path(path, expected):
    assert split_zmq_path(path) == expected


@pytest.mark.parametrize(
    "invalid_path",
    [
        "invalid_path",  # Missing scheme
        "tcp://127.0.0.1",  # Missing port
        "tcp://[::1]",  # Missing port for IPv6
        "tcp://:5555",  # Missing host
    ],
)
def test_split_zmq_path_invalid(invalid_path):
    with pytest.raises(ValueError):
        split_zmq_path(invalid_path)


def test_make_zmq_socket_ipv6():
    # Check if IPv6 is supported by trying to create an IPv6 socket
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.close()
    except OSError:
        pytest.skip("IPv6 is not supported on this system")

    ctx = zmq.Context()
    ipv6_path = "tcp://[::]:5555"  # IPv6 loopback address
    socket_type = zmq.REP  # Example socket type

    # Create the socket
    zsock: zmq.Socket = make_zmq_socket(ctx, ipv6_path, socket_type)

    # Verify that the IPV6 option is set
    assert zsock.getsockopt(zmq.IPV6) == 1, (
        "IPV6 option should be enabled for IPv6 addresses"
    )

    # Clean up
    zsock.close()
    ctx.term()


def test_make_zmq_path():
    assert make_zmq_path("tcp", "127.0.0.1", "5555") == "tcp://127.0.0.1:5555"
    assert make_zmq_path("tcp", "::1", "5555") == "tcp://[::1]:5555"


def test_get_tcp_uri():
    assert get_tcp_uri("127.0.0.1", 5555) == "tcp://127.0.0.1:5555"
    assert get_tcp_uri("::1", 5555) == "tcp://[::1]:5555"


def test_split_host_port():
    # valid ipv4
    assert split_host_port("127.0.0.1:5555") == ("127.0.0.1", 5555)
    # invalid ipv4
    with pytest.raises(ValueError):
        # multi colon
        assert split_host_port("127.0.0.1::5555")
    with pytest.raises(ValueError):
        # tailing colon
        assert split_host_port("127.0.0.1:5555:")
    with pytest.raises(ValueError):
        # no colon
        assert split_host_port("127.0.0.15555")
    with pytest.raises(ValueError):
        # none int port
        assert split_host_port("127.0.0.1:5555a")

    # valid ipv6
    assert split_host_port("[::1]:5555") == ("::1", 5555)
    # invalid ipv6
    with pytest.raises(ValueError):
        # multi colon
        assert split_host_port("[::1]::5555")
    with pytest.raises(IndexError):
        # no colon
        assert split_host_port("[::1]5555")
    with pytest.raises(ValueError):
        # none int port
        assert split_host_port("[::1]:5555a")


def test_join_host_port():
    assert join_host_port("127.0.0.1", 5555) == "127.0.0.1:5555"
    assert join_host_port("::1", 5555) == "[::1]:5555"


# ---------------------------------------------------------------------------
# Tests for late-binding helpers (added alongside #28498 fix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "addr,expected",
    [
        ("tcp://*:0", True),
        ("tcp://127.0.0.1:0", True),
        ("tcp://192.168.1.5:0", True),
        ("tcp://127.0.0.1:8080", False),
        ("tcp://127.0.0.1:1", False),
        ("ipc:///tmp/socket", False),
        ("inproc://my-socket", False),
    ],
)
def test_is_wildcard_addr(addr, expected):
    assert is_wildcard_addr(addr) == expected


def test_make_zmq_socket_return_address_false_is_socket():
    """Default return_address=False should return a socket, not a tuple."""
    ctx = zmq.Context()
    try:
        result = make_zmq_socket(ctx, "tcp://127.0.0.1:0", zmq.PULL,
                                 bind=True, return_address=False)
        assert not isinstance(result, tuple), (
            "return_address=False must return a socket, not a tuple"
        )
        result.close()
    finally:
        ctx.term()


def test_make_zmq_socket_return_address_true_is_tuple():
    """return_address=True should return (socket, address) tuple."""
    ctx = zmq.Context()
    try:
        result = make_zmq_socket(ctx, "tcp://127.0.0.1:0", zmq.PULL,
                                 bind=True, return_address=True)
        assert isinstance(result, tuple) and len(result) == 2, (
            "return_address=True must return a (socket, address) tuple"
        )
        sock, addr = result
        assert isinstance(addr, str)
        sock.close()
    finally:
        ctx.term()


def test_make_zmq_socket_wildcard_resolves_port():
    """Binding to port 0 should yield a real OS-assigned port in the address."""
    ctx = zmq.Context()
    try:
        sock, addr = make_zmq_socket(ctx, "tcp://127.0.0.1:0", zmq.PULL,
                                     bind=True, return_address=True)
        assert addr.startswith("tcp://"), f"Expected tcp:// address, got {addr!r}"
        _, _, port_str = split_zmq_path(addr)
        assigned_port = int(port_str)
        assert assigned_port > 0, (
            f"OS should assign a non-zero port, got {assigned_port}"
        )
        assert not is_wildcard_addr(addr), (
            "Returned address must have a concrete port, not a wildcard"
        )
        sock.close()
    finally:
        ctx.term()


def test_make_zmq_socket_wildcard_star_host_resolves():
    """Wildcard host 'tcp://*:0' should bind and resolve to a real address."""
    ctx = zmq.Context()
    try:
        sock, addr = make_zmq_socket(ctx, "tcp://*:0", zmq.PULL,
                                     bind=True, return_address=True)
        _, _, port_str = split_zmq_path(addr)
        assert int(port_str) > 0
        assert not is_wildcard_addr(addr)
        sock.close()
    finally:
        ctx.term()


def test_make_zmq_socket_non_wildcard_returns_input_addr():
    """For a fixed port, return_address should echo back the original path."""
    ctx = zmq.Context()
    # Find a free port first so we don't collide
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    fixed_addr = f"tcp://127.0.0.1:{port}"

    try:
        sock, addr = make_zmq_socket(ctx, fixed_addr, zmq.PULL,
                                     bind=True, return_address=True)
        assert addr == fixed_addr, (
            f"Fixed-port address should be echoed back unchanged, "
            f"expected {fixed_addr!r}, got {addr!r}"
        )
        sock.close()
    finally:
        ctx.term()


def test_bind_zmq_socket_and_get_address_returns_usable_addr():
    """bind_zmq_socket_and_get_address must return an address that another
    socket can actually connect and send a message to."""
    ctx = zmq.Context()
    try:
        pull_sock, addr = bind_zmq_socket_and_get_address(
            ctx, "tcp://127.0.0.1:0", zmq.PULL
        )
        assert addr.startswith("tcp://")
        _, _, port_str = split_zmq_path(addr)
        assert int(port_str) > 0

        # Verify end-to-end connectivity
        push_sock = ctx.socket(zmq.PUSH)
        push_sock.connect(addr)
        push_sock.send(b"hello")
        pull_sock.setsockopt(zmq.RCVTIMEO, 2000)  # 2 s timeout
        msg = pull_sock.recv()
        assert msg == b"hello", f"Expected b'hello', got {msg!r}"

        push_sock.close()
        pull_sock.close()
    finally:
        ctx.term()


def test_wildcard_binds_produce_unique_ports():
    """Multiple wildcard binds must each get a distinct OS port — the core
    guarantee that eliminates the race condition from issue #28498."""
    n = 10
    ctx = zmq.Context()
    sockets = []
    try:
        addrs = []
        for _ in range(n):
            sock, addr = make_zmq_socket(
                ctx, "tcp://127.0.0.1:0", zmq.PULL,
                bind=True, return_address=True
            )
            sockets.append(sock)
            addrs.append(addr)

        ports = [int(split_zmq_path(a)[2]) for a in addrs]
        assert len(set(ports)) == n, (
            f"Expected {n} unique ports, got duplicates: {ports}"
        )
    finally:
        for s in sockets:
            s.close()
        ctx.term()


def test_wildcard_binds_unique_ports_concurrent():
    """Concurrent wildcard binds from multiple threads must not collide.

    This is a regression test for the race condition described in #28498:
    previously, two threads could independently call _get_open_port(), both
    observe the same free port, and then one would fail with EADDRINUSE.
    With late binding the OS guarantees unique assignment.
    """
    n = 20
    results: list = [None] * n
    errors: list = [None] * n

    def bind_in_thread(idx: int) -> None:
        ctx = zmq.Context()
        try:
            sock, addr = make_zmq_socket(
                ctx, "tcp://127.0.0.1:0", zmq.PULL,
                bind=True, return_address=True
            )
            results[idx] = addr
            sock.close()
        except Exception as exc:
            errors[idx] = exc
        finally:
            ctx.term()

    threads = [threading.Thread(target=bind_in_thread, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [e for e in errors if e is not None]
    assert not failed, f"Some threads failed to bind: {failed}"

    ports = [int(split_zmq_path(a)[2]) for a in results if a]
    assert len(set(ports)) == n, (
        f"Expected {n} unique ports across threads, got duplicates: {ports}"
    )
