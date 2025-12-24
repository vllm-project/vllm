# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket

import pytest
import zmq

from vllm.utils.network_utils import (
    get_open_port,
    get_reserved_port,
    get_reserved_ports_list,
    get_tcp_uri,
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


def test_reserved_port_basic():
    """Test that ReservedPort holds a port and releases it correctly."""
    reserved = get_reserved_port()
    assert isinstance(reserved.port, int)
    assert 1 <= reserved.port <= 65535
    assert reserved._socket is not None

    # Port should NOT be bindable while reserved
    with pytest.raises(OSError), socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", reserved.port))

    released_port = reserved.release()
    assert released_port == reserved.port
    assert reserved._socket is None


def test_reserved_port_context_manager():
    """Test that ReservedPort works as a context manager."""
    with get_reserved_port() as reserved:
        port = reserved.port
        assert isinstance(port, int)
        assert reserved._socket is not None
    assert reserved._socket is None


def test_reserved_port_prevents_race_condition():
    """Test for GitHub issue #28498."""
    reserved = get_reserved_port()
    port = reserved.port

    # Port should fail to bind while reserved
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_sock:
            test_sock.bind(("", port))
            pytest.fail("Port was not properly reserved")
    except OSError:
        pass

    reserved.release()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_sock:
        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_sock.bind(("", port))


def test_get_reserved_ports_list():
    """Test getting a list of reserved ports."""
    reservations = get_reserved_ports_list(3)
    assert len(reservations) == 3
    ports = [r.port for r in reservations]
    assert len(set(ports)) == 3
    for r in reservations:
        assert r._socket is not None
        r.release()


def test_reserved_port_double_release():
    """Test that releasing a port twice is safe."""
    reserved = get_reserved_port()
    port1 = reserved.release()
    port2 = reserved.release()
    assert port1 == port2
