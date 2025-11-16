# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
from concurrent.futures import ThreadPoolExecutor

import pytest
import zmq

from vllm.utils.network_utils import (
    get_and_hold_open_port,
    get_open_port,
    get_tcp_uri,
    join_host_port,
    make_zmq_path,
    make_zmq_socket,
    split_host_port,
    split_zmq_path,
)
from vllm.v1.utils import get_engine_client_zmq_addr_with_socket


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


# Tests for socket holding functionality (issue #28498)


def test_get_and_hold_open_port_basic():
    """Test that get_and_hold_open_port returns a valid socket and port."""
    sock, port = get_and_hold_open_port()
    try:
        # Verify socket is a valid socket object
        assert isinstance(sock, socket.socket)
        # Verify port is a positive integer
        assert isinstance(port, int)
        assert port > 0
        # Verify socket is bound to the port
        assert sock.getsockname()[1] == port
    finally:
        sock.close()


def test_get_and_hold_open_port_prevents_reuse():
    """Test that the held socket prevents other processes from using the port.

    This is the core test for the race condition fix from issue #28498.
    """
    sock1, port1 = get_and_hold_open_port()
    try:
        # Try to bind another socket to the same port - should fail
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(OSError) as exc_info:
            s2.bind(("", port1))
        s2.close()
        assert exc_info.value.errno in (48, 98)  # EADDRINUSE: 48 on macOS, 98 on Linux
    finally:
        sock1.close()

    # After closing sock1, the port should be available again
    # (though OS may have delay in releasing)
    # Note: We don't test rebinding immediately due to TIME_WAIT state


def test_get_and_hold_open_port_with_dp_master_port(monkeypatch: pytest.MonkeyPatch):
    """Test that get_and_hold_open_port avoids DP master port range."""
    with monkeypatch.context() as m:
        # Set DP master port to a high value to avoid conflicts
        m.setenv("VLLM_DP_MASTER_PORT", "50000")

        # Get multiple ports
        socks_and_ports = [get_and_hold_open_port() for _ in range(5)]
        try:
            # Verify none of the ports are in the reserved range [50000, 50010)
            for sock, port in socks_and_ports:
                assert port not in range(50000, 50010), (
                    f"Port {port} should not be in DP master port "
                    f"reserved range [50000, 50010)"
                )
        finally:
            for sock, _ in socks_and_ports:
                sock.close()


def test_get_and_hold_open_port_ipv4():
    """Test that get_and_hold_open_port returns IPv4 socket with SO_REUSEADDR."""
    sock, port = get_and_hold_open_port()
    try:
        # Verify socket family is AF_INET (IPv4)
        assert sock.family == socket.AF_INET
        # Verify SO_REUSEADDR is set
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 1
    finally:
        sock.close()


def test_get_and_hold_open_port_multiple_sockets():
    """Test that we can hold multiple ports simultaneously."""
    num_sockets = 5
    socks_and_ports = [get_and_hold_open_port() for _ in range(num_sockets)]
    try:
        # Verify all ports are unique
        ports = [port for _, port in socks_and_ports]
        assert len(set(ports)) == num_sockets, "All ports should be unique"

        # Verify all sockets are alive and bound
        for sock, port in socks_and_ports:
            assert sock.getsockname()[1] == port
    finally:
        for sock, _ in socks_and_ports:
            sock.close()


def test_get_and_hold_open_port_concurrent():
    """Test that get_and_hold_open_port works correctly under concurrent access.

    This test validates the fix for the race condition reported in issue #28498.
    """
    num_threads = 10
    results = []

    def allocate_port():
        sock, port = get_and_hold_open_port()
        return (sock, port)

    # Allocate ports concurrently from multiple threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(allocate_port) for _ in range(num_threads)]
        results = [f.result() for f in futures]

    try:
        # Verify all ports are unique
        ports = [port for _, port in results]
        assert len(set(ports)) == num_threads, (
            "All ports should be unique even under concurrent allocation"
        )

        # Verify no EADDRINUSE errors occurred (implicitly checked by futures)
    finally:
        for sock, _ in results:
            sock.close()


def test_socket_holding_prevents_zmq_conflict():
    """Test that a held socket prevents ZMQ from binding to the same port."""
    # Get a held socket
    sock, port = get_and_hold_open_port()
    try:
        # Try to create a ZMQ socket on the same port - should fail
        ctx = zmq.Context()
        zmq_sock = ctx.socket(zmq.REP)
        with pytest.raises(zmq.error.ZMQError) as exc_info:
            zmq_sock.bind(f"tcp://127.0.0.1:{port}")
        # ZMQError wraps EADDRINUSE
        assert "Address already in use" in str(exc_info.value).lower()
        zmq_sock.close()
        ctx.term()
    finally:
        sock.close()


def test_get_engine_client_zmq_addr_with_socket_tcp():
    """Test get_engine_client_zmq_addr_with_socket for TCP addresses."""
    address, held_socket = get_engine_client_zmq_addr_with_socket(
        local_only=False, host="127.0.0.1", port=0
    )
    try:
        # Verify address format
        assert address.startswith("tcp://127.0.0.1:")
        # Verify socket is returned for auto-assigned port
        assert held_socket is not None
        assert isinstance(held_socket, socket.socket)
        # Extract port from address and verify it matches socket
        port = int(address.split(":")[-1])
        assert held_socket.getsockname()[1] == port
    finally:
        if held_socket:
            held_socket.close()


def test_get_engine_client_zmq_addr_with_socket_ipc():
    """Test get_engine_client_zmq_addr_with_socket for IPC addresses."""
    address, held_socket = get_engine_client_zmq_addr_with_socket(
        local_only=True, host="127.0.0.1"
    )
    # Verify address format for IPC
    assert address.startswith("ipc://")
    # Verify no socket is returned for IPC (no port allocation needed)
    assert held_socket is None


def test_get_engine_client_zmq_addr_with_socket_explicit_port():
    """Test get_engine_client_zmq_addr_with_socket with explicit port."""
    explicit_port = 12345
    address, held_socket = get_engine_client_zmq_addr_with_socket(
        local_only=False, host="127.0.0.1", port=explicit_port
    )
    # Verify address uses the explicit port
    assert address == f"tcp://127.0.0.1:{explicit_port}"
    # Verify no socket is returned for explicit port (no holding needed)
    assert held_socket is None
