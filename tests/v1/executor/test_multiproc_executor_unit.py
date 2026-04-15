# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for helper logic in the multiprocess executor."""

import socket

from vllm.utils.network_utils import create_distributed_init_endpoint


def test_create_distributed_init_endpoint_without_reserved_socket(monkeypatch):
    monkeypatch.setattr(
        "vllm.utils.network_utils.get_open_port",
        lambda: 23456,
    )

    endpoint = create_distributed_init_endpoint(
        "127.0.0.1",
        reserve_listen_socket=False,
    )

    assert endpoint.init_method == "tcp://127.0.0.1:23456"
    assert endpoint.listen_socket is None


def test_create_distributed_init_endpoint_with_reserved_socket():
    endpoint = create_distributed_init_endpoint(
        "127.0.0.1",
        reserve_listen_socket=True,
    )

    try:
        assert endpoint.listen_socket is not None
        assert endpoint.listen_socket.family == socket.AF_INET
        _, port = endpoint.listen_socket.getsockname()
        assert endpoint.init_method == f"tcp://127.0.0.1:{port}"
    finally:
        if endpoint.listen_socket is not None:
            endpoint.listen_socket.close()
