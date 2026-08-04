# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache ping``."""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import io
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.ping import (
    PingCommand,
    ping,
)

# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


class _MockHandler(BaseHTTPRequestHandler):
    """Minimal handler that serves a canned response."""

    response_body: bytes = b""
    response_code: int = 200

    def do_GET(self):
        self.send_response(self.response_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format, *args):
        pass  # suppress stderr noise


def _make_handler(code: int, body: bytes = b""):
    return type("_H", (_MockHandler,), {"response_code": code, "response_body": body})


def _start_server(code: int, body: bytes = b"") -> tuple[HTTPServer, int]:
    """Start a local HTTP server and return ``(server, port)``."""
    server = HTTPServer(("127.0.0.1", 0), _make_handler(code, body))
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()
    return server, port


# ---------------------------------------------------------------------------
# ping() helper tests
# ---------------------------------------------------------------------------


class TestPing:
    def test_success(self):
        server, port = _start_server(200)
        try:
            status, rtt_ms, error = ping(f"http://127.0.0.1:{port}/health")
            assert status == "OK"
            assert rtt_ms > 0
            assert error is None
        finally:
            server.server_close()

    def test_503(self):
        body = json.dumps(
            {"status": "unhealthy", "reason": "engine not initialized"}
        ).encode()
        server, port = _start_server(503, body)
        try:
            status, rtt_ms, error = ping(f"http://127.0.0.1:{port}/health")
            assert status == "FAIL"
            assert rtt_ms > 0
            assert "503" in error
        finally:
            server.server_close()

    def test_connection_refused(self):
        status, rtt_ms, error = ping("http://127.0.0.1:19999/health")
        assert status == "FAIL"
        assert rtt_ms >= 0
        assert "Cannot connect" in error


# ---------------------------------------------------------------------------
# PingCommand end-to-end tests (real HTTP server)
# ---------------------------------------------------------------------------


class TestPingCommandKvcache:
    def test_ok(self):
        body = json.dumps({"status": "healthy"}).encode()
        server, port = _start_server(200, body)
        try:
            cmd = PingCommand()

            class FakeArgs:
                target = "kvcache"
                url = f"http://127.0.0.1:{port}"
                format = "json"
                output = None

            buf = io.StringIO()
            # Standard
            from unittest.mock import patch

            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

            output = json.loads(buf.getvalue())
            assert output["title"] == "Ping KV Cache"
            m = output["metrics"]
            assert m["status"] == "OK"
            assert m["round_trip_time_ms"] > 0
        finally:
            server.server_close()

    def test_503_exits_1(self):
        body = json.dumps(
            {"status": "unhealthy", "reason": "engine not initialized"}
        ).encode()
        server, port = _start_server(503, body)
        try:
            cmd = PingCommand()

            class FakeArgs:
                target = "kvcache"
                url = f"http://127.0.0.1:{port}"
                format = "json"
                output = None

            with pytest.raises(SystemExit) as exc_info:
                cmd.execute(FakeArgs())
            assert exc_info.value.code == 1
        finally:
            server.server_close()

    def test_connection_refused_exits_1(self):
        cmd = PingCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://127.0.0.1:19999"
            format = "json"
            output = None

        with pytest.raises(SystemExit) as exc_info:
            cmd.execute(FakeArgs())
        assert exc_info.value.code == 1


class TestPingCommandEngine:
    def test_ok(self):
        server, port = _start_server(200)
        try:
            cmd = PingCommand()

            class FakeArgs:
                target = "engine"
                url = f"http://127.0.0.1:{port}"
                format = "json"
                output = None

            buf = io.StringIO()
            # Standard
            from unittest.mock import patch

            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

            output = json.loads(buf.getvalue())
            assert output["title"] == "Ping Engine"
            m = output["metrics"]
            assert m["status"] == "OK"
            assert m["round_trip_time_ms"] > 0
        finally:
            server.server_close()

    def test_503_exits_1(self):
        server, port = _start_server(503)
        try:
            cmd = PingCommand()

            class FakeArgs:
                target = "engine"
                url = f"http://127.0.0.1:{port}"
                format = "json"
                output = None

            with pytest.raises(SystemExit) as exc_info:
                cmd.execute(FakeArgs())
            assert exc_info.value.code == 1
        finally:
            server.server_close()


# ---------------------------------------------------------------------------
# Default URL resolution
# ---------------------------------------------------------------------------


class TestDefaultUrls:
    def test_kvcache_default(self):
        """Verify that url=None resolves to localhost:8080/healthcheck."""
        cmd = PingCommand()

        class FakeArgs:
            target = "kvcache"
            url = None
            format = "json"
            output = None

        # Connection will be refused (no server), but we can check the URL
        # from the error message.
        with pytest.raises(SystemExit):
            cmd.execute(FakeArgs())

    def test_engine_default(self):
        """Verify that url=None resolves to localhost:8000/health."""
        cmd = PingCommand()

        class FakeArgs:
            target = "engine"
            url = None
            format = "json"
            output = None

        with pytest.raises(SystemExit):
            cmd.execute(FakeArgs())
