# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    _enable_ipv6_if_needed,
    _make_zmq_tcp_path,
    _normalize_zmq_address,
    _split_zmq_address,
)


def test_p2p_zmq_address_helpers_support_ipv6():
    assert _normalize_zmq_address("127.0.0.1:5555") == "127.0.0.1:5555"
    assert _make_zmq_tcp_path("127.0.0.1:5555") == "tcp://127.0.0.1:5555"

    assert _normalize_zmq_address("::1:5555") == "[::1]:5555"
    assert _normalize_zmq_address("[::1]:5555") == "[::1]:5555"
    assert _make_zmq_tcp_path("::1:5555") == "tcp://[::1]:5555"
    assert _make_zmq_tcp_path("[::1]:5555") == "tcp://[::1]:5555"


def test_p2p_zmq_socket_enables_ipv6():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    try:
        _enable_ipv6_if_needed(sock, "[::1]:5555")
        assert sock.getsockopt(zmq.IPV6) == 1
    finally:
        sock.close()
        ctx.term()


def test_p2p_zmq_socket_enables_ipv6_for_ipv6_hostname(monkeypatch):
    import socket

    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET6, None, None, None, None)]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    try:
        _enable_ipv6_if_needed(sock, "example.test:5555")
        assert sock.getsockopt(zmq.IPV6) == 1
    finally:
        sock.close()
        ctx.term()


@pytest.mark.parametrize(
    "address",
    [
        "",
        "localhost",
        "localhost:",
        ":5555",
        "[::1]",
        "localhost:not-a-port",
    ],
)
def test_p2p_zmq_address_rejects_missing_or_invalid_ports(address):
    with pytest.raises(ValueError, match="Invalid P2P ZMQ address|cannot be empty"):
        _split_zmq_address(address)


def test_p2p_request_id_parsing_supports_ipv6():
    assert P2pNcclConnector.parse_request_id(
        "request___prefill_addr_::1:5555___", is_prefill=False
    ) == ("::1", 5555)
    assert P2pNcclConnector.parse_request_id(
        "request___prefill_addr_[::1]:5555___", is_prefill=False
    ) == ("::1", 5555)
    assert P2pNcclConnector.parse_request_id(
        "request___decode_addr_::1:5555", is_prefill=True
    ) == ("::1", 5555)
