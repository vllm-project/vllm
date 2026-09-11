# tests/entrypoints/openai/test_listen.py
import errno
import socket
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

import vllm.entrypoints.launchers.utils.listen as listen


def make_args(**kwargs) -> Namespace:
    defaults = dict(
        uds=None,
        host="127.0.0.1",
        port=8000,
        ssl_keyfile=None,
        ssl_certfile=None,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def _fake_socket():
    return MagicMock(spec=socket.socket)


# ---------------- _describe_port_holder ----------------


def test_describe_no_process(monkeypatch):
    monkeypatch.setattr(listen, "find_process_using_port", lambda port: None)
    assert listen._describe_port_holder(8000) == "unknown process"


def test_describe_with_process(monkeypatch):
    proc = MagicMock()
    proc.pid = 42
    proc.name.return_value = "python"
    proc.cmdline.return_value = ["python", "-m", "vllm"]
    monkeypatch.setattr(listen, "find_process_using_port", lambda port: proc)

    desc = listen._describe_port_holder(8000)
    assert "pid=42" in desc
    assert "name='python'" in desc
    assert "cmd='python -m vllm'" in desc


def test_describe_handles_exception(monkeypatch):
    proc = MagicMock()
    proc.pid = 42
    proc.name.side_effect = Exception("nope")
    monkeypatch.setattr(listen, "find_process_using_port", lambda port: proc)

    assert "<unavailable>" in listen._describe_port_holder(8000)


# ---------------- setup_listen_address ----------------


class TestSetupListenAddress:
    def test_uds(self, monkeypatch):
        fake = _fake_socket()
        monkeypatch.setattr(listen, "create_server_unix_socket", lambda path: fake)

        addr, sock = listen.setup_listen_address(
            make_args(uds="/tmp/vllm.sock"), reuse_port=False
        )
        assert addr == "unix:/tmp/vllm.sock"
        assert sock is fake

    def test_tcp(self, monkeypatch):
        fake = _fake_socket()
        calls = []

        def fake_create(sock_addr, reuse_port):
            calls.append((sock_addr, reuse_port))
            return fake

        monkeypatch.setattr(listen, "create_server_socket", fake_create)

        addr, sock = listen.setup_listen_address(
            make_args(host="127.0.0.1", port=8000), reuse_port=True
        )
        assert addr == "http://127.0.0.1:8000"
        assert sock is fake
        assert calls == [(("127.0.0.1", 8000), True)]

    def test_https(self, monkeypatch):
        monkeypatch.setattr(
            listen,
            "create_server_socket",
            lambda addr, reuse_port: _fake_socket(),
        )
        addr, _ = listen.setup_listen_address(
            make_args(ssl_keyfile="k", ssl_certfile="c"), reuse_port=False
        )
        assert addr == "https://127.0.0.1:8000"

    def test_ipv6_host(self, monkeypatch):
        monkeypatch.setattr(
            listen,
            "create_server_socket",
            lambda addr, reuse_port: _fake_socket(),
        )
        addr, _ = listen.setup_listen_address(make_args(host="::1"), reuse_port=False)
        assert addr == "http://[::1]:8000"

    def test_empty_host_shows_0_0_0_0(self, monkeypatch):
        monkeypatch.setattr(
            listen,
            "create_server_socket",
            lambda addr, reuse_port: _fake_socket(),
        )
        addr, _ = listen.setup_listen_address(make_args(host=""), reuse_port=False)
        assert addr == "http://0.0.0.0:8000"

    def test_port_in_use_message(self, monkeypatch):
        def raise_eaddr(sock_addr, reuse_port):
            raise OSError(errno.EADDRINUSE, "in use")

        monkeypatch.setattr(listen, "create_server_socket", raise_eaddr)
        monkeypatch.setattr(listen, "_describe_port_holder", lambda p: "pid=123")

        with pytest.raises(OSError, match="Port 8000 is already in use"):
            listen.setup_listen_address(make_args(port=8000), reuse_port=False)

    def test_permission_denied_message(self, monkeypatch):
        def raise_eacces(sock_addr, reuse_port):
            raise OSError(errno.EACCES, "denied")

        monkeypatch.setattr(listen, "create_server_socket", raise_eacces)

        with pytest.raises(OSError, match="Permission denied"):
            listen.setup_listen_address(make_args(port=80), reuse_port=False)

    def test_other_oserror_reraised(self, monkeypatch):
        def raise_other(sock_addr, reuse_port):
            raise OSError(errno.ENETDOWN, "net down")

        monkeypatch.setattr(listen, "create_server_socket", raise_other)

        with pytest.raises(OSError) as ei:
            listen.setup_listen_address(make_args(), reuse_port=False)
        assert ei.value.errno == errno.ENETDOWN


# ---------------- cleanup_listen_socket ----------------


class TestCleanupListenSocket:
    def test_close_only(self):
        sock = MagicMock()
        listen.cleanup_listen_socket(sock)
        sock.close.assert_called_once()

    def test_close_and_unlink(self, tmp_path):
        path = tmp_path / "vllm.sock"
        path.write_text("")
        sock = MagicMock()

        listen.cleanup_listen_socket(sock, str(path))

        sock.close.assert_called_once()
        assert not path.exists()

    def test_missing_uds_is_ok(self, tmp_path):
        sock = MagicMock()
        listen.cleanup_listen_socket(sock, str(tmp_path / "missing.sock"))
        sock.close.assert_called_once()

    def test_close_error_is_suppressed(self):
        sock = MagicMock()
        sock.close.side_effect = OSError("boom")
        listen.cleanup_listen_socket(sock)
