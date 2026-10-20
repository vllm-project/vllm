# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: SIM117
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
    setup_listen_address_from_args,
)

_MODULE = "vllm.entrypoints.launchers.utils.listen"


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


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
# Scenario: serving over a Unix domain socket
# ─────────────────────────────────────────────────────────────────────────────


class TestUnixSocketScenario:
    """Everything related to ``--uds``: bind, stale-file detection, errors."""

    def test_binds_and_returns_unix_address(self, tmp_path):
        uds_path = str(tmp_path / "vllm.sock")

        listen_address, sock = setup_listen_address(uds=uds_path, reuse_port=False)
        try:
            assert listen_address == f"unix:{uds_path}"
            assert sock.family == socket.AF_UNIX
            assert os.path.exists(uds_path)
        finally:
            cleanup_listen_socket(sock, uds_path)

        assert not os.path.exists(uds_path)

    def test_stale_socket_file_gives_actionable_error(self, tmp_path):
        """A crashed prior run leaves a path-bound file; error must say so.

        Port-oriented hints ("free the port") would be confusing for UDS,
        so verify none of them leak through.
        """
        uds_path = tmp_path / "stale.sock"
        uds_path.write_bytes(b"")  # bound to any regular file → EADDRINUSE

        with pytest.raises(OSError, match="already exists") as exc_info:
            setup_listen_address(uds=str(uds_path), reuse_port=False)

        msg = str(exc_info.value)
        assert str(uds_path) in msg
        assert "stale socket file" in msg
        # Must NOT reuse TCP-oriented wording.
        assert "--port" not in msg
        assert "SO_REUSEPORT" not in msg
        assert "--api-server-count" not in msg

    def test_other_oserror_propagates(self):
        # Non-EADDRINUSE errors on the UDS path must be re-raised unchanged.
        err = OSError(errno.EACCES, "Permission denied")

        with mock.patch(f"{_MODULE}.create_server_unix_socket", side_effect=err):
            with pytest.raises(OSError) as exc_info:
                setup_listen_address(uds="/some/path", reuse_port=False)

        assert exc_info.value is err


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: serving over TCP (address construction only)
# ─────────────────────────────────────────────────────────────────────────────


class TestTcpBindScenario:
    """TCP bind + ``listen_address`` construction, no conflict involved."""

    def test_ephemeral_port_reports_real_port(self):
        # ``--port 0`` must report the kernel-assigned port, not ``:0``.
        listen_address, sock = setup_listen_address(
            host="127.0.0.1", port=0, reuse_port=False
        )
        try:
            bound_port = sock.getsockname()[1]
            assert bound_port != 0
            assert listen_address == f"http://127.0.0.1:{bound_port}"
        finally:
            sock.close()

    @pytest.mark.parametrize("host", [None, ""])
    def test_default_host_is_0000(self, host):
        # `None` and "" host must be displayed as 0.0.0.0, not `:port`.
        listen_address, sock = setup_listen_address(host=host, port=0, reuse_port=False)
        try:
            bound_port = sock.getsockname()[1]
            assert listen_address == f"http://0.0.0.0:{bound_port}"
        finally:
            sock.close()

    def test_hostname_preserved(self):
        # ``--host localhost`` must be preserved, not rewritten to
        # ``127.0.0.1`` by getsockname().
        listen_address, sock = setup_listen_address(
            host="localhost", port=0, reuse_port=False
        )
        try:
            bound_port = sock.getsockname()[1]
            assert listen_address == f"http://localhost:{bound_port}"
        finally:
            sock.close()

    def test_ipv6_address_is_bracketed(self, ipv6_loopback_available):
        if not ipv6_loopback_available:
            pytest.skip("IPv6 loopback not available")

        listen_address, sock = setup_listen_address(
            host="::1", port=0, reuse_port=False
        )
        try:
            bound_port = sock.getsockname()[1]
            assert listen_address == f"http://[::1]:{bound_port}"
        finally:
            sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: serving over HTTPS
# ─────────────────────────────────────────────────────────────────────────────


class TestSslScenario:
    """TLS affects the reported scheme (``https://`` vs ``http://``)."""

    def test_both_key_and_cert_implies_https(self):
        listen_address, sock = setup_listen_address(
            host="127.0.0.1",
            port=0,
            reuse_port=False,
            ssl_keyfile="/tmp/key.pem",
            ssl_certfile="/tmp/cert.pem",
        )
        try:
            bound_port = sock.getsockname()[1]
            assert listen_address == f"https://127.0.0.1:{bound_port}"
        finally:
            sock.close()

    def test_partial_ssl_files_stays_http(self):
        # Only one of keyfile/certfile: must NOT report https.
        listen_address, sock = setup_listen_address(
            host="127.0.0.1",
            port=0,
            reuse_port=False,
            ssl_keyfile="/tmp/key.pem",
            ssl_certfile=None,
        )
        try:
            assert listen_address.startswith("http://")
        finally:
            sock.close()

    def test_default_ssl_params_fall_back_to_http(self):
        # ``ssl_keyfile`` / ``ssl_certfile`` default to ``None``; omitting them
        # must not raise and must produce ``http://``.
        listen_address, sock = setup_listen_address(
            host="127.0.0.1", port=0, reuse_port=False
        )
        try:
            assert listen_address.startswith("http://")
        finally:
            sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: --api-server-count > 1 (SO_REUSEPORT)
# ─────────────────────────────────────────────────────────────────────────────


class TestReusePortScenario:
    """Sharing one TCP port across multiple API server processes."""

    def test_two_sockets_share_same_port(self):
        """With reuse_port=True, two sockets can bind the same port.

        This is what enables ``api_server_count > 1`` in the launcher.
        """
        _, sock1 = setup_listen_address(host="127.0.0.1", port=0, reuse_port=True)
        try:
            port = sock1.getsockname()[1]
            _, sock2 = setup_listen_address(
                host="127.0.0.1", port=port, reuse_port=True
            )
            try:
                assert sock2.getsockname()[1] == port
            finally:
                cleanup_listen_socket(sock2)
        finally:
            cleanup_listen_socket(sock1)

    def test_reuse_port_forwarded_to_socket(self):
        fake_sock = mock.MagicMock(spec=socket.socket)
        fake_sock.family = socket.AF_INET
        fake_sock.getsockname.return_value = ("127.0.0.1", 12345)

        with mock.patch(
            f"{_MODULE}.create_server_socket", return_value=fake_sock
        ) as create_mock:
            listen_address, sock = setup_listen_address(
                host="127.0.0.1", port=0, reuse_port=True
            )

        create_mock.assert_called_once_with(("127.0.0.1", 0), reuse_port=True)
        assert sock is fake_sock
        assert listen_address == "http://127.0.0.1:12345"

    def test_unsupported_platform_gives_actionable_error(self):
        # Older kernels / unsupported platforms: SO_REUSEPORT setsockopt
        # fails with ENOPROTOOPT. The hint must point at --api-server-count.
        err = OSError(errno.ENOPROTOOPT, "Protocol not available")

        with mock.patch(f"{_MODULE}.create_server_socket", side_effect=err):
            with pytest.raises(OSError, match="SO_REUSEPORT") as exc_info:
                setup_listen_address(host="127.0.0.1", port=8000, reuse_port=True)

        msg = str(exc_info.value)
        assert "not supported" in msg
        assert "--api-server-count" in msg
        assert exc_info.value.__cause__ is err

    def test_unsupported_platform_without_reuse_port_propagates(self):
        # ENOPROTOOPT without reuse_port is unrelated to SO_REUSEPORT; must
        # propagate unchanged.
        err = OSError(errno.ENOPROTOOPT, "Protocol not available")

        with mock.patch(f"{_MODULE}.create_server_socket", side_effect=err):
            with pytest.raises(OSError) as exc_info:
                setup_listen_address(host="127.0.0.1", port=8000, reuse_port=False)

        assert exc_info.value is err


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: port already in use (EADDRINUSE)
# ─────────────────────────────────────────────────────────────────────────────


class TestPortConflictScenario:
    """The user asked for a port that is already held by another process."""

    def test_real_kernel_conflict(self, blocking_listener):
        host, port = blocking_listener

        with mock.patch(f"{_MODULE}.find_process_using_port", return_value=None):
            with pytest.raises(OSError, match="already in use") as exc_info:
                setup_listen_address(host=host, port=port, reuse_port=False)

        msg = str(exc_info.value)
        assert f"Port {port}" in msg
        assert "unknown process" in msg

    def test_message_preserves_original_error_and_queries_port(self):
        err = OSError(errno.EADDRINUSE, "Address already in use")

        with (
            mock.patch(f"{_MODULE}.create_server_socket", side_effect=err),
            mock.patch(
                f"{_MODULE}.find_process_using_port", return_value=None
            ) as find_mock,
            pytest.raises(OSError, match="already in use") as exc_info,
        ):
            setup_listen_address(host="127.0.0.1", port=8000, reuse_port=False)

        assert exc_info.value.__cause__ is err
        assert "Port 8000" in str(exc_info.value)
        find_mock.assert_called_once_with(8000)

    def test_holder_description_is_embedded(self):
        # Previous tests mock the holder away; this one checks the actual
        # holder description ends up in the raised message.
        err = OSError(errno.EADDRINUSE, "Address already in use")

        proc = mock.MagicMock()
        proc.pid = 4321
        proc.name.return_value = "nginx"
        proc.cmdline.return_value = ["nginx", "-g", "daemon off;"]

        with (
            mock.patch(f"{_MODULE}.create_server_socket", side_effect=err),
            mock.patch(f"{_MODULE}.find_process_using_port", return_value=proc),
            pytest.raises(OSError) as exc_info,
        ):
            setup_listen_address(host="127.0.0.1", port=8000, reuse_port=False)

        msg = str(exc_info.value)
        assert "pid=4321" in msg
        assert "name='nginx'" in msg
        assert "cmd='nginx -g daemon off;'" in msg

    def test_survives_holder_query_failure(self):
        # macOS SIP denies psutil.net_connections(); the EADDRINUSE message
        # must stay actionable even when the holder lookup itself fails.
        err = OSError(errno.EADDRINUSE, "Address already in use")

        with (
            mock.patch(f"{_MODULE}.create_server_socket", side_effect=err),
            mock.patch(
                f"{_MODULE}.find_process_using_port",
                side_effect=PermissionError("macOS SIP"),
            ),
            pytest.raises(OSError, match="already in use") as exc_info,
        ):
            setup_listen_address(host="127.0.0.1", port=8000, reuse_port=False)

        msg = str(exc_info.value)
        assert "unknown process (query failed)" in msg
        assert "Port 8000" in msg

    def test_hint_suggests_reuse_port_when_not_enabled(self):
        err = OSError(errno.EADDRINUSE, "Address already in use")

        with (
            mock.patch(f"{_MODULE}.create_server_socket", side_effect=err),
            mock.patch(f"{_MODULE}.find_process_using_port", return_value=None),
        ):
            with pytest.raises(OSError) as exc_info:
                setup_listen_address(host="127.0.0.1", port=8000, reuse_port=False)

        msg = str(exc_info.value)
        assert "SO_REUSEPORT" in msg
        assert "--api-server-count > 1" in msg

    def test_hint_does_not_suggest_reuse_port_when_already_enabled(self):
        """When reuse_port=True, suggesting "enable SO_REUSEPORT" misleads.

        The holder is the one missing the flag (or is a non-vLLM process),
        so the user cannot act on that hint.
        """
        err = OSError(errno.EADDRINUSE, "Address already in use")

        with (
            mock.patch(f"{_MODULE}.create_server_socket", side_effect=err),
            mock.patch(f"{_MODULE}.find_process_using_port", return_value=None),
        ):
            with pytest.raises(OSError) as exc_info:
                setup_listen_address(host="127.0.0.1", port=8000, reuse_port=True)

        msg = str(exc_info.value)
        assert "SO_REUSEPORT" in msg  # mention is fine
        # Must NOT tell the user to enable SO_REUSEPORT — they already did.
        assert "enable" not in msg.lower()
        assert "--api-server-count" not in msg


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: insufficient privileges (EACCES)
# ─────────────────────────────────────────────────────────────────────────────


class TestPermissionScenario:
    """Binding a privileged port (< 1024) as a non-root user."""

    def test_error_message_points_at_privileges(self):
        err = OSError(errno.EACCES, "Permission denied")

        with mock.patch(f"{_MODULE}.create_server_socket", side_effect=err):
            with pytest.raises(OSError, match="Permission denied binding") as exc_info:
                setup_listen_address(host="127.0.0.1", port=80, reuse_port=False)

        msg = str(exc_info.value)
        assert "127.0.0.1:80" in msg
        assert "elevated privileges" in msg
        assert exc_info.value.__cause__ is err

    @pytest.mark.skipif(
        os.geteuid() == 0,
        reason="requires non-root (root can bind ports <1024)",
    )
    def test_real_kernel_denial(self):
        # On a non-root Linux/macOS CI this exercises the real EACCES path.
        with pytest.raises(OSError, match="Permission denied binding"):
            setup_listen_address(host="127.0.0.1", port=80, reuse_port=False)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: unexpected OS error (must not be masked)
# ─────────────────────────────────────────────────────────────────────────────


class TestUnexpectedErrorScenario:
    def test_unknown_oserror_propagates_unchanged(self):
        err = OSError(errno.ENOMEM, "Out of memory")

        with mock.patch(f"{_MODULE}.create_server_socket", side_effect=err):
            with pytest.raises(OSError) as exc_info:
                setup_listen_address(host="127.0.0.1", port=8000, reuse_port=False)

        assert exc_info.value is err


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: argparse adapter (setup_listen_address_from_args)
# ─────────────────────────────────────────────────────────────────────────────


class TestSetupListenAddressFromArgs:
    """``setup_listen_address_from_args`` is a thin adapter.

    It has exactly one responsibility: unpack ``args`` fields and forward
    them unchanged to ``setup_listen_address``. All bind behavior is
    already covered by the classes above, so these tests mock the inner
    function and only verify the field mapping.
    """

    @pytest.fixture
    def full_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            host="127.0.0.1",
            port=8000,
            uds=None,
            ssl_keyfile="/tmp/key.pem",
            ssl_certfile="/tmp/cert.pem",
        )

    def test_forwards_all_args_fields(self, full_args):
        sentinel = ("http://127.0.0.1:8000", mock.MagicMock(spec=socket.socket))

        with mock.patch(
            f"{_MODULE}.setup_listen_address", return_value=sentinel
        ) as inner:
            result = setup_listen_address_from_args(full_args, reuse_port=True)

        inner.assert_called_once_with(
            host="127.0.0.1",
            port=8000,
            uds=None,
            reuse_port=True,
            ssl_keyfile="/tmp/key.pem",
            ssl_certfile="/tmp/cert.pem",
        )
        # Return value is passed through unchanged.
        assert result == sentinel

    def test_uds_scenario_forwards_uds_and_none_host_port(self):
        args = argparse.Namespace(
            host=None,
            port=None,
            uds="/tmp/vllm.sock",
            ssl_keyfile=None,
            ssl_certfile=None,
        )
        sentinel = ("unix:/tmp/vllm.sock", mock.MagicMock(spec=socket.socket))

        with mock.patch(
            f"{_MODULE}.setup_listen_address", return_value=sentinel
        ) as inner:
            result = setup_listen_address_from_args(args, reuse_port=False)

        inner.assert_called_once_with(
            host=None,
            port=None,
            uds="/tmp/vllm.sock",
            reuse_port=False,
            ssl_keyfile=None,
            ssl_certfile=None,
        )
        assert result == sentinel

    def test_missing_ssl_attrs_forwarded_as_none(self):
        """Embedded callers may build ``args`` without the SSL fields.

        If the adapter uses ``getattr(args, ..., None)``, this must not
        raise and must forward ``None``. If it accesses ``args.ssl_keyfile``
        directly, this test fails — in that case, either drop this test or
        switch the adapter to ``getattr``.
        """
        args = argparse.Namespace(host="127.0.0.1", port=8000, uds=None)
        assert not hasattr(args, "ssl_keyfile")
        assert not hasattr(args, "ssl_certfile")

        with mock.patch(
            f"{_MODULE}.setup_listen_address",
            return_value=("http://127.0.0.1:8000", mock.MagicMock()),
        ) as inner:
            setup_listen_address_from_args(args, reuse_port=False)

        kwargs = inner.call_args.kwargs
        assert kwargs["ssl_keyfile"] is None
        assert kwargs["ssl_certfile"] is None

    def test_reuse_port_is_sourced_from_caller_not_args(self, full_args):
        """Guard against someone sourcing ``reuse_port`` from ``args``.

        The contract is: ``reuse_port`` comes from the caller (it depends on
        ``api_server_count``, which is a launcher-level decision), not from
        the Namespace. If this ever changes, this test catches it.
        """
        assert not hasattr(full_args, "reuse_port")

        with mock.patch(
            f"{_MODULE}.setup_listen_address",
            return_value=("http://127.0.0.1:8000", mock.MagicMock()),
        ) as inner:
            setup_listen_address_from_args(full_args, reuse_port=True)

        assert inner.call_args.kwargs["reuse_port"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: socket cleanup (independent of how the socket was bound)
# ─────────────────────────────────────────────────────────────────────────────


class TestSocketCleanup:
    """``cleanup_listen_socket`` must be safe and thorough in every case."""

    def test_closes_tcp_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        assert sock.fileno() != -1

        cleanup_listen_socket(sock)

        assert sock.fileno() == -1

    def test_swallows_close_error(self):
        sock = mock.MagicMock(spec=socket.socket)
        sock.family = socket.AF_INET
        sock.close.side_effect = OSError("boom")

        cleanup_listen_socket(sock)  # must not raise

        sock.close.assert_called_once()

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
        cleanup_listen_socket(sock, uds_path)  # must not raise

    def test_swallows_missing_uds_unlink(self, tmp_path):
        uds_path = str(tmp_path / "never-created.sock")
        sock = mock.MagicMock(spec=socket.socket)
        sock.family = socket.AF_UNIX

        cleanup_listen_socket(sock, uds_path)  # must not raise

        sock.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: describing who holds the conflicting port
# ─────────────────────────────────────────────────────────────────────────────


class TestPortHolderDescription:
    """``_describe_port_holder`` is best-effort; must never raise."""

    def test_no_process_found(self):
        with mock.patch(f"{_MODULE}.find_process_using_port", return_value=None):
            assert _describe_port_holder(8000) == "unknown process"

    def test_query_failure(self):
        # E.g. macOS SIP denies psutil.net_connections().
        with mock.patch(
            f"{_MODULE}.find_process_using_port",
            side_effect=RuntimeError("psutil gone"),
        ):
            assert _describe_port_holder(8000) == "unknown process (query failed)"

    def test_full_description(self):
        proc = mock.MagicMock()
        proc.pid = 1234
        proc.name.return_value = "uvicorn"
        proc.cmdline.return_value = ["uvicorn", "app:main"]

        with mock.patch(f"{_MODULE}.find_process_using_port", return_value=proc):
            result = _describe_port_holder(8000)

        assert "pid=1234" in result
        assert "name='uvicorn'" in result
        assert "cmd='uvicorn app:main'" in result

    def test_process_attr_access_fails(self):
        # Permission-restricted process: pid is available, name/cmdline are not.
        proc = mock.MagicMock()
        proc.pid = 1234
        proc.name.side_effect = RuntimeError("access denied")
        proc.cmdline.side_effect = RuntimeError("access denied")

        with mock.patch(f"{_MODULE}.find_process_using_port", return_value=proc):
            result = _describe_port_holder(8000)

        assert "pid=1234" in result
        assert "name='<unavailable>'" in result
        assert "cmd='<unavailable>'" in result
