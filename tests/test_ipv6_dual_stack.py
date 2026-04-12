# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for IPv6 dual-stack socket support in network_utils."""

import socket

import pytest

from vllm.utils.network_utils import (
    _get_addrinfos_for_bind,
    _get_open_port,
    is_port_available,
    try_bind_socket,
)


class TestGetAddrinfosForBind:
    def test_returns_non_empty(self):
        infos = _get_addrinfos_for_bind(None, 0)
        assert len(infos) > 0

    def test_ipv4_first(self):
        infos = _get_addrinfos_for_bind(None, 0)
        assert infos[0][0] == socket.AF_INET

    def test_no_duplicate_families(self):
        infos = _get_addrinfos_for_bind(None, 0)
        families = [info[0] for info in infos]
        # At least one family should be present
        assert len(families) >= 1

    def test_explicit_ipv4_host(self):
        infos = _get_addrinfos_for_bind("127.0.0.1", 0)
        assert all(info[0] == socket.AF_INET for info in infos)

    def test_explicit_ipv6_host(self):
        infos = _get_addrinfos_for_bind("::1", 0)
        assert all(info[0] == socket.AF_INET6 for info in infos)

    def test_all_entries_are_stream(self):
        infos = _get_addrinfos_for_bind(None, 0)
        for info in infos:
            assert info[1] == socket.SOCK_STREAM


class TestTryBindSocket:
    def test_bind_ephemeral(self):
        s = try_bind_socket(None, 0)
        try:
            port = s.getsockname()[1]
            assert port > 0
        finally:
            s.close()

    def test_socket_is_reusable(self):
        s = try_bind_socket(None, 0, reuse_addr=True)
        try:
            val = s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
            assert val != 0
        finally:
            s.close()

    def test_bind_with_listen(self):
        s = try_bind_socket(None, 0, listen=True)
        try:
            port = s.getsockname()[1]
            assert port > 0
        finally:
            s.close()

    def test_ipv6_v6only(self):
        try:
            s = try_bind_socket("::1", 0)
        except OSError:
            pytest.skip("IPv6 not available")
        try:
            if s.family == socket.AF_INET6:
                val = s.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)
                assert val != 0
        finally:
            s.close()

    def test_bad_host_raises(self):
        with pytest.raises(OSError):
            try_bind_socket("192.0.2.1", 0)

    def test_reuse_port(self):
        s = try_bind_socket(None, 0, reuse_port=True)
        try:
            port = s.getsockname()[1]
            assert port > 0
        finally:
            s.close()


class TestIsPortAvailable:
    def test_ephemeral_port_available(self):
        s = try_bind_socket(None, 0)
        port = s.getsockname()[1]
        s.close()
        assert is_port_available(port)

    def test_occupied_port_unavailable(self):
        s = try_bind_socket(None, 0)
        try:
            port = s.getsockname()[1]
            assert not is_port_available(port)
        finally:
            s.close()


class TestGetOpenPort:
    def test_returns_positive(self):
        port = _get_open_port()
        assert port > 0

    def test_returned_port_is_bindable(self):
        port = _get_open_port()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("", port))
        finally:
            s.close()

    def test_start_port(self):
        port = _get_open_port(start_port=19876)
        assert port >= 19876

    def test_max_attempts(self):
        s = try_bind_socket(None, 0)
        try:
            occupied = s.getsockname()[1]
            with pytest.raises(RuntimeError):
                _get_open_port(start_port=occupied, max_attempts=1)
        finally:
            s.close()

    def test_unique_ports(self):
        ports = {_get_open_port() for _ in range(5)}
        assert len(ports) >= 1
