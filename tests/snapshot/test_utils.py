# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket
from unittest.mock import MagicMock, call, patch

import pytest

from vllm.snapshot.utils import get_local_ip


def _socket(ip: str | None = None, error: Exception | None = None):
    context = MagicMock()
    sock = context.__enter__.return_value
    sock.connect.side_effect = error
    if ip is not None:
        sock.getsockname.return_value = (ip, 0)
    return context


def test_get_local_ip_probes_ipv4_first():
    ipv4_socket = _socket("10.0.0.2")

    with patch(
        "vllm.snapshot.utils.socket.socket", return_value=ipv4_socket
    ) as factory:
        assert get_local_ip() == "10.0.0.2"

    factory.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)


def test_get_local_ip_falls_back_to_ipv6():
    ipv4_socket = _socket(error=OSError())
    ipv6_socket = _socket("2001:db8::2")

    with patch(
        "vllm.snapshot.utils.socket.socket",
        side_effect=[ipv4_socket, ipv6_socket],
    ) as factory:
        assert get_local_ip() == "2001:db8::2"

    assert factory.call_args_list == [
        call(socket.AF_INET, socket.SOCK_DGRAM),
        call(socket.AF_INET6, socket.SOCK_DGRAM),
    ]


def test_get_local_ip_raises_when_probe_fails():
    with (
        patch(
            "vllm.snapshot.utils.socket.socket",
            side_effect=[OSError(), OSError()],
        ),
        pytest.raises(RuntimeError, match="current local IP"),
    ):
        get_local_ip()
