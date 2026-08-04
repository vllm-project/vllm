# SPDX-License-Identifier: Apache-2.0
"""End-to-end test: the MP cache server emits usage telemetry at startup.

Spawns a real ``run_cache_server`` in a child process with the stats-server
URL pointed at a local HTTP sink, and asserts the startup report arrives
over the wire.
"""

# Standard
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import threading
import time

# Third Party
import pytest

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.server import run_cache_server

if not torch_dev.is_available():
    pytest.skip(
        f"requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

SERVER_PORT = 15599
STARTUP_TIMEOUT_SECONDS = 90.0


class _SinkHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.received.append((self.path, body))  # type: ignore[attr-defined]
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


class _UsageSink(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _SinkHandler)
        self.received: list[tuple[str, dict[str, object]]] = []


def _usage_server_runner(port: int) -> None:
    """Child-process entry point running a minimal cache server."""
    mp_config = MPServerConfig(host="localhost", port=port)
    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 30,
                use_lazy=True,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


def test_server_startup_emits_usage_report(monkeypatch, tmp_path):
    sink = _UsageSink()
    sink_thread = threading.Thread(target=sink.serve_forever, daemon=True)
    sink_thread.start()

    # The child inherits this environment at spawn: tracking enabled,
    # stats URL pointed at the sink, HOME isolated so no real state files
    # (machine_id) are touched.
    monkeypatch.setenv("LMCACHE_TRACK_USAGE", "true")
    monkeypatch.setenv(
        "LMCACHE_USAGE_TRACK_URL", f"http://127.0.0.1:{sink.server_address[1]}"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LMCACHE_USAGE_TRACK_INTERVAL", "1")

    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_usage_server_runner, args=(SERVER_PORT,))
    process.start()
    try:
        # With a short flush interval, continuous heartbeats can interleave
        # with (or precede) the one-shot startup messages: wait for the
        # startup message types, not a message count.
        def received_types() -> set[object]:
            return {payload["message_type"] for _, payload in sink.received}

        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline and not (
            {"EnvMessage", "MPServerMessage"} <= received_types()
        ):
            time.sleep(0.1)
        received = list(sink.received)
        payloads = {payload["message_type"]: payload for _, payload in received}
        assert {"EnvMessage", "MPServerMessage"} <= set(payloads), (
            f"startup messages missing, got {received}"
        )
        startup_paths = {
            path
            for path, payload in received
            if payload["message_type"] in ("EnvMessage", "MPServerMessage")
        }
        assert startup_paths == {"/context"}

        mp_payload = payloads["MPServerMessage"]
        assert mp_payload["deployment_mode"] == "mp_server"
        assert mp_payload["chunk_size"] == 256
        assert mp_payload["l1_size_bytes"] == 1 << 30
        assert "instance_id" not in mp_payload
        assert payloads["EnvMessage"]["session_id"] == mp_payload["session_id"]

        # With a 1 s flush interval the continuous reporter must deliver a
        # heartbeat shortly after startup.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not any(
            payload["message_type"] == "ContinuousContextMessage"
            for _, payload in sink.received
        ):
            time.sleep(0.1)
        continuous = [
            (path, payload)
            for path, payload in sink.received
            if payload["message_type"] == "ContinuousContextMessage"
        ]
        assert continuous, "no ContinuousContextMessage received"
        path, payload = continuous[0]
        assert path == "/cache-usage"
        assert payload["deployment_mode"] == "mp_server"
        assert payload["sequence_number"] >= 1
        assert payload["uptime_seconds"] >= 0
        assert payload["session_id"] == mp_payload["session_id"]
    finally:
        process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join()
        sink.shutdown()
        sink.server_close()
