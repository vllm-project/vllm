# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
from types import SimpleNamespace

import pytest
import zmq

from vllm.utils import network_utils
from vllm.utils.network_utils import (
    get_open_port,
    get_open_ports_list,
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


def test_get_open_port_skips_dp_reserved_range(monkeypatch: pytest.MonkeyPatch):
    # VLLM_PORT makes the port scan deterministic, so a candidate landing in
    # the data parallel reserved range used to loop forever.
    dp_master_port = 5678
    reserved_port_range = range(dp_master_port, dp_master_port + 10)
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", str(dp_master_port))
        m.setenv("VLLM_DP_MASTER_PORT", str(dp_master_port))
        port = get_open_port()
        assert port not in reserved_port_range
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))


def test_get_open_port_dp_without_vllm_port(monkeypatch: pytest.MonkeyPatch):
    dp_master_port = 5678
    reserved_port_range = range(dp_master_port, dp_master_port + 10)
    with monkeypatch.context() as m:
        m.delenv("VLLM_PORT", raising=False)
        m.setenv("VLLM_DP_MASTER_PORT", str(dp_master_port))
        port = get_open_port()
        assert port not in reserved_port_range
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))


def test_get_open_port_dp_vllm_port_above_range(monkeypatch: pytest.MonkeyPatch):
    # The scan only ever counts upward, so a VLLM_PORT above the reserved
    # range must be honoured as-is rather than pushed higher.
    dp_master_port = 5678
    reserved_port_range = range(dp_master_port, dp_master_port + 10)
    start_port = reserved_port_range.stop + 12
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", str(start_port))
        m.setenv("VLLM_DP_MASTER_PORT", str(dp_master_port))
        port = get_open_port()
        assert port >= start_port
        assert port not in reserved_port_range
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))


def test_get_open_port_without_dp_master_port(monkeypatch: pytest.MonkeyPatch):
    start_port = 5678
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", str(start_port))
        m.delenv("VLLM_DP_MASTER_PORT", raising=False)
        port = get_open_port()
        assert port >= start_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))


def test_get_open_port_dp_reserved_range_exhausted(monkeypatch: pytest.MonkeyPatch):
    dp_master_port = 5678
    reserved_port_range = range(dp_master_port, dp_master_port + 10)

    class OnlyReservedPortsFree:
        """Socket stub where every port outside the reserved range is taken."""

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def bind(self, address):
            if address[1] not in reserved_port_range:
                raise OSError("address already in use")

    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", str(dp_master_port))
        m.setenv("VLLM_DP_MASTER_PORT", str(dp_master_port))
        m.setattr(
            network_utils,
            "socket",
            SimpleNamespace(
                socket=OnlyReservedPortsFree,
                AF_INET=socket.AF_INET,
                AF_INET6=socket.AF_INET6,
                SOCK_STREAM=socket.SOCK_STREAM,
            ),
        )
        with pytest.raises(
            RuntimeError,
            match=f"starting from port {reserved_port_range.stop}",
        ):
            get_open_port()


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
