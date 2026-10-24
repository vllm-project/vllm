# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Readiness deadline tests using a real managed HTTP server."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.benchmarks.sweep import server as server_module
from vllm.benchmarks.sweep.server import ServerProcess

pytestmark = pytest.mark.skip_global_cleanup

_HTTP_SERVER = """
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(float(sys.argv[2]))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass

with HTTPServer(('127.0.0.1', 0), Handler) as server:
    Path(sys.argv[1]).write_text(str(server.server_port))
    server.serve_forever()
"""


@pytest.fixture
def http_server(tmp_path: Path):
    servers: list[ServerProcess] = []

    def start(delay: float = 0):
        port_file = tmp_path / f"port-{len(servers)}"
        server = ServerProcess(
            [sys.executable, "-c", _HTTP_SERVER, str(port_file), str(delay)],
            [],
            show_stdout=False,
        )
        server.start()
        servers.append(server)
        deadline = time.monotonic() + 10
        while not port_file.exists():
            assert time.monotonic() < deadline, "HTTP fixture failed to start"
            time.sleep(0.01)
        server.server_cmd.extend(
            ["--host", "127.0.0.1", "--port", port_file.read_text()]
        )
        return server

    yield start

    for server in servers:
        server.stop()
        server._server_process.wait(timeout=5)


def test_ready_http_server_returns(http_server):
    server = http_server()

    server.wait_until_ready(timeout=1)


def test_health_response_after_deadline_times_out(http_server):
    """A health endpoint responding after the wait budget is not a success."""
    server = http_server(delay=2)

    with pytest.raises(TimeoutError, match="within 1 seconds"):
        server.wait_until_ready(timeout=1)


@pytest.mark.parametrize("ready,probe_duration", [(False, 0.6), (True, 1.2)])
def test_deadline_rejects_late_probes(http_server, monkeypatch, ready, probe_duration):
    """A deadline excludes both another retry and a late successful probe."""
    server = http_server()
    clock = 0.0
    probes = 0

    def probe(*args, **kwargs):
        nonlocal clock, probes
        probes += 1
        clock += probe_duration
        return ready

    def sleep(seconds):
        nonlocal clock
        clock += seconds

    monkeypatch.setattr(server, "is_server_ready", probe)
    monkeypatch.setattr(
        server_module, "time", SimpleNamespace(monotonic=lambda: clock, sleep=sleep)
    )

    with pytest.raises(TimeoutError):
        server.wait_until_ready(timeout=1)

    assert probes == 1
