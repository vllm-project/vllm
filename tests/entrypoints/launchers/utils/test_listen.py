# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import errno
import os
import socket
from unittest import mock

import pytest

from vllm.entrypoints.launchers.utils.listen import (
    _describe_port_holder,
    cleanup_listen_socket,
    setup_listen_address,
)

_MODULE = "vllm.entrypoints.launchers.utils.listen"


@pytest.fixture
def make_args():
    """Factory for a minimal ``argparse.Namespace`` resembling serve args."""

    def _make(**overrides) -> argparse.Namespace:
        defaults = {
            "uds": None,
            "host": None,
            "port": 0,
            "ssl_keyfile": None,
            "ssl_certfile": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    return _make


@pytest.fixture(scope="module")
def ipv6_loopback_available() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.fixture
def blocking_listener():
    """Bind+listen on an ephemeral port and yield the (host, port)."""
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        yield blocker.getsockname()[:2]
    finally:
        blocker.close()


# ─────────────────────────────────────────────────────────────────────────────
# setup_listen_address
# ─────────────────────────────────────────────────────────────────────────────


class TestSetupListenAddress:
    def test_uds(self, make_args, tmp_path):
        uds_path = str(tmp_path / "vllm.sock")
        args = make_args(uds=uds_path)

        listen_address, sock = setup_listen_address(args, reuse_port=False)
        try:
            assert listen_address == f"unix:{uds_path}"
            assert sock.family == socket.AF_UNIX
            assert os.path.exists(uds_path)
        finally:
            cleanup_listen_socket(sock, uds_path)

        assert not os.path.exists(uds_path)

    def test_tcp_ephemeral_port(self, make_args):
        args = make_args(host="127.0.0.1", port=0)

        listen_address, sock = setup_listen_address(args, reuse_port=False)
        try:
            bound_port = sock.getsockname()[1]
            assert bound_port != 0
            assert listen_address == f"http://127.0.0.1:{bound_port}"
        finally:
            sock.close()

    def test_tcp_default_host(self, make_args):
        args = make_args(host=None, port=0)

        listen_address, sock = setup_listen_address(args, reuse_port=False)
        try:
            bound_port = sock.getsockname()[1]
            # `None` / "" host must be displayed as 0.0.0.0, not `:port`.
            assert listen_address == f"http://0.0.0.0:{bound_port}"
        finally:
            sock.close()

    def test_tcp_hostname_preserved(self, make_args):
        # `--host localhost` should be preserved in the reported address,
        # not rewritten to `127.0.0.1` by getsockname().
        args = make_args(host="localhost", port=0)

        listen_address, sock = setup_listen_address(args, reuse_port=False)
        try:
            bound_port = sock.getsockname()[1]
            assert listen_address == f"http://localhost:{bound_port}"
        finally:
            sock.close()

    def test_ipv6(self, make_args, ipv6_loopback_available):
        if not ipv6_loopback_available:
            pytest.skip("IPv6 loopback not available")

        args = make_args(host="::1", port=0)
        listen_address, sock = setup_listen_address(args, reuse_port=False)
        try:
            bound_port = sock.getsockname()[1]
            # IPv6 literal must be bracketed in the URL.
            assert listen_address == f"http://[::1]:{bound_port}"
        finally:
            sock.close()

    def test_ssl(self, make_args):
        args = make_args(
            host="127.0.0.1",
            port=0,
            ssl_keyfile="/tmp/key.pem",
            ssl_certfile="/tmp/cert.pem",
        )

        listen_address, sock = setup_listen_address(args, reuse_port=False)
        try:
            bound_port = sock.getsockname()[1]
            assert listen_address == f"https://127.0.0.1:{bound_port}"
        finally:
            sock.close()

    def test_partial_ssl_not_https(self, make_args):
        # Only one of keyfile/certfile: must NOT report https.
        args = make_args(
            host="127.0.0.1",
            port=0,
            ssl_keyfile="/tmp/key.pem",
            ssl_certfile=None,
        )

        listen_address, sock = setup_listen_address(args, reuse_port=False)
        try:
            assert listen_address.startswith("http://")
        finally:
            sock.close()

    def test_missing_ssl_attrs(self):
        # `args` built by embedded callers may not carry ssl_keyfile/certfile.
        args = argparse.Namespace(uds=None, host="127.0.0.1", port=0)
        assert not hasattr(args, "ssl_keyfile")
        assert not hasattr(args, "ssl_certfile")

        listen_address, sock = setup_listen_address(args, reuse_port=False)
        try:
            assert listen_address.startswith("http://")
        finally:
            sock.close()

    def test_port_in_use_real(self, make_args, blocking_listener):
        host, port = blocking_listener
        args = make_args(host=host, port=port)

        with mock.patch(f"{_MODULE}.find_process_using_port", return_value=None):
            with pytest.raises(OSError, match="already in use") as exc_info:
                setup_listen_address(args, reuse_port=False)

        msg = str(exc_info.value)
        assert f"Port {port}" in msg
        assert "unknown process" in msg

    def test_port_in_use_mocked(self, make_args):
        args = make_args(host="127.0.0.1", port=8000)
        err = OSError(errno.EADDRINUSE, "Address already in use")

        with (
            mock.patch(f"{_MODULE}.create_server_socket", side_effect=err),
            mock.patch(
                f"{_MODULE}.find_process_using_port", return_value=None
            ) as find_mock,
        ):
            with pytest.raises(OSError, match="already in use") as exc_info:
                setup_listen_address(args, reuse_port=False)

        assert exc_info.value.__cause__ is err
        assert "Port 8000" in str(exc_info.value)
        find_mock.assert_called_once_with(8000)

    def test_permission_denied(self, make_args):
        args = make_args(host="127.0.0.1", port=80)
        err = OSError(errno.EACCES, "Permission denied")

        with mock.patch(f"{_MODULE}.create_server_socket", side_effect=err):
            with pytest.raises(OSError, match="Permission denied binding") as exc_info:
                setup_listen_address(args, reuse_port=False)

        msg = str(exc_info.value)
        assert "127.0.0.1:80" in msg
        assert "elevated privileges" in msg
        assert exc_info.value.__cause__ is err

    def test_other_oserror_propagates(self, make_args):
        # Non-EADDRINUSE / non-EACCES must be re-raised unchanged.
        args = make_args(host="127.0.0.1", port=8000)
        err = OSError(errno.ENOMEM, "Out of memory")

        with mock.patch(f"{_MODULE}.create_server_socket", side_effect=err):
            with pytest.raises(OSError) as exc_info:
                setup_listen_address(args, reuse_port=False)

        assert exc_info.value is err

    def test_reuse_port_forwarded(self, make_args):
        args = make_args(host="127.0.0.1", port=0)
        fake_sock = mock.MagicMock(spec=socket.socket)
        fake_sock.family = socket.AF_INET
        fake_sock.getsockname.return_value = ("127.0.0.1", 12345)

        with mock.patch(
            f"{_MODULE}.create_server_socket", return_value=fake_sock
        ) as create_mock:
            listen_address, sock = setup_listen_address(args, reuse_port=True)

        create_mock.assert_called_once_with(("127.0.0.1", 0), reuse_port=True)
        assert sock is fake_sock
        assert listen_address == "http://127.0.0.1:12345"


# ─────────────────────────────────────────────────────────────────────────────
# cleanup_listen_socket
# ─────────────────────────────────────────────────────────────────────────────


class TestCleanupListenSocket:
    def test_closes_tcp_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        assert sock.fileno() != -1

        cleanup_listen_socket(sock)

        assert sock.fileno() == -1

    def test_unlinks_uds_with_explicit_path(self, tmp_path):
        uds_path = str(tmp_path / "explicit.sock")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(uds_path)
        assert os.path.exists(uds_path)

        cleanup_listen_socket(sock, uds_path)

        assert not os.path.exists(uds_path)
        assert sock.fileno() == -1

    def test_infers_uds_path_from_sock(self, tmp_path):
        uds_path = str(tmp_path / "infer.sock")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(uds_path)
        assert os.path.exists(uds_path)

        cleanup_listen_socket(sock)  # no uds_path passed

        assert not os.path.exists(uds_path)
        assert sock.fileno() == -1

    def test_idempotent(self, tmp_path):
        uds_path = str(tmp_path / "idem.sock")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(uds_path)

        cleanup_listen_socket(sock, uds_path)
        # Second call must not raise.
        cleanup_listen_socket(sock, uds_path)

    def test_swallows_close_error(self):
        sock = mock.MagicMock(spec=socket.socket)
        sock.family = socket.AF_INET
        sock.close.side_effect = OSError("boom")

        cleanup_listen_socket(sock)  # must not raise

        sock.close.assert_called_once()

    def test_swallows_missing_uds_unlink(self, tmp_path):
        uds_path = str(tmp_path / "never-created.sock")
        sock = mock.MagicMock(spec=socket.socket)
        sock.family = socket.AF_UNIX

        cleanup_listen_socket(sock, uds_path)  # must not raise

        sock.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# _describe_port_holder
# ─────────────────────────────────────────────────────────────────────────────


class TestDescribePortHolder:
    def test_no_process(self):
        with mock.patch(f"{_MODULE}.find_process_using_port", return_value=None):
            assert _describe_port_holder(8000) == "unknown process"

    def test_query_failure(self):
        with mock.patch(
            f"{_MODULE}.find_process_using_port",
            side_effect=RuntimeError("psutil gone"),
        ):
            assert _describe_port_holder(8000) == "unknown process (query failed)"

    def test_normal(self):
        proc = mock.MagicMock()
        proc.pid = 1234
        proc.name.return_value = "uvicorn"
        proc.cmdline.return_value = ["uvicorn", "app:main"]

        with mock.patch(f"{_MODULE}.find_process_using_port", return_value=proc):
            result = _describe_port_holder(8000)

        assert "pid=1234" in result
        assert "name='uvicorn'" in result
        assert "cmd='uvicorn app:main'" in result

    def test_process_attrs_fail(self):
        proc = mock.MagicMock()
        proc.pid = 1234
        proc.name.side_effect = RuntimeError("access denied")
        proc.cmdline.side_effect = RuntimeError("access denied")

        with mock.patch(f"{_MODULE}.find_process_using_port", return_value=proc):
            result = _describe_port_holder(8000)

        assert "pid=1234" in result
        assert "name='<unavailable>'" in result
        assert "cmd='<unavailable>'" in result
