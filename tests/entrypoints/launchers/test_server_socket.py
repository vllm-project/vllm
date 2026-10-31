# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket

import pytest

from vllm.entrypoints.launchers.launcher import create_server_socket


def _has_ipv6_loopback() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(
    not socket.has_dualstack_ipv6(), reason="host has no dual-stack sockets"
)
def test_unspecified_host_accepts_ipv4_and_ipv6():
    """With no --host, the server is reachable over both IPv4 and IPv6."""
    hosts = ["127.0.0.1"] + (["::1"] if _has_ipv6_loopback() else [])
    with create_server_socket(("", 0), reuse_port=False) as server:
        server.listen()
        port = server.getsockname()[1]
        for host in hosts:
            with socket.create_connection((host, port), timeout=5):
                conn, _ = server.accept()
                conn.close()


@pytest.mark.parametrize(
    "host,family", [("127.0.0.1", socket.AF_INET), ("::1", socket.AF_INET6)]
)
def test_explicit_host_keeps_its_family(host: str, family: socket.AddressFamily):
    if family == socket.AF_INET6 and not _has_ipv6_loopback():
        pytest.skip("host has no IPv6 loopback")
    with create_server_socket((host, 0), reuse_port=False) as server:
        assert server.family == family
        assert server.getsockname()[0] == host
