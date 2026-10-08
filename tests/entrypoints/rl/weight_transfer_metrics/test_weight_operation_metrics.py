# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from vllm.entrypoints.rl.online import api_router
from vllm.entrypoints.rl.online import metrics as rl_metrics
from vllm.entrypoints.rl.online.metrics import WeightOperationMetrics
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.v1.metrics.prometheus import get_prometheus_registry

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

DURATION = "vllm:rl_weight_update_operation_duration_seconds_count"
IN_FLIGHT = "vllm:rl_weight_update_operations_in_flight"


def sample(registry, name, operation):
    return registry.get_sample_value(name, {"operation": operation})


class Request:
    def __init__(self, engine):
        self.app = SimpleNamespace(state=SimpleNamespace(engine_client=engine))


class Engine:
    """Holds every engine call open until the test releases it."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            self.started.set()
            await self.release.wait()

        return call


@pytest.fixture
def registry(monkeypatch):
    registry = CollectorRegistry()
    monkeypatch.setattr(rl_metrics, "_metrics", WeightOperationMetrics(registry))
    return registry


@pytest.mark.parametrize("fail", [False, True])
def test_record_releases_in_flight_and_observes_duration(registry, fail):
    metrics = rl_metrics.weight_operation_metrics()
    with (
        pytest.raises(RuntimeError) if fail else nullcontext(),
        metrics.record("update"),
    ):
        assert sample(registry, IN_FLIGHT, "update") == 1
        if fail:
            raise RuntimeError
    assert sample(registry, IN_FLIGHT, "update") == 0
    assert sample(registry, DURATION, "update") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,route,body",
    [
        ("init", api_router.init_weight_transfer_engine, {"init_info": {}}),
        ("start", api_router.start_weight_update, None),
        ("start_draft", api_router.start_draft_weight_update, None),
        ("update", api_router.update_weights, {"update_info": {}}),
        ("finish", api_router.finish_weight_update, None),
    ],
)
async def test_route_records_its_operation(registry, operation, route, body):
    engine = Engine()
    task = asyncio.create_task(route(Request(engine), **(body or {})))
    await engine.started.wait()
    assert sample(registry, IN_FLIGHT, operation) == 1
    engine.release.set()
    await task
    assert sample(registry, IN_FLIGHT, operation) == 0
    assert sample(registry, DURATION, operation) == 1


@pytest.mark.asyncio
async def test_cancelled_route_releases_in_flight(registry):
    engine = Engine()
    task = asyncio.create_task(
        api_router.update_weights(Request(engine), update_info={})
    )
    await engine.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sample(registry, IN_FLIGHT, "update") == 0
    assert sample(registry, DURATION, "update") == 1


HOLD_OPERATION = """
import sys
from vllm.entrypoints.rl.online.metrics import weight_operation_metrics
with weight_operation_metrics().record("update"):
    print("entered", flush=True)
    sys.stdin.readline()
"""


def test_in_flight_is_summed_across_api_server_processes(tmp_path, monkeypatch):
    """Two API-server processes each hold an operation; /metrics sums them."""
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    servers = [
        subprocess.Popen(
            [sys.executable, "-c", HOLD_OPERATION],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        for _ in range(2)
    ]
    try:
        for server in servers:
            assert server.stdout is not None
            assert server.stdout.readline().strip() == "entered"
        assert sample(get_prometheus_registry(), IN_FLIGHT, "update") == 2
    finally:
        for server in servers:
            server.communicate("\n", timeout=60)

    scraped = get_prometheus_registry()
    assert sample(scraped, IN_FLIGHT, "update") == 0
    assert sample(scraped, DURATION, "update") == 2


def test_rejected_request_is_not_recorded(registry):
    app = FastAPI()
    app.include_router(api_router.router)
    init_exception_handler(app)
    app.state.engine_client = Engine()
    app.state.args = SimpleNamespace(log_error_stack=False)
    response = TestClient(app).post("/update_weights", json={})
    assert response.status_code == 400
    assert sample(registry, DURATION, "update") is None


def test_default_recorder_is_created_lazily_on_the_default_registry():
    """The default recorder registers on the process registry only when first used.

    Runs in a fresh interpreter: the registry is process-global, so a route test
    elsewhere in the session may already have registered these collectors, and
    "not registered yet" would then be unobservable.
    """
    probe = """
from prometheus_client import REGISTRY

from vllm.entrypoints.rl.online import metrics as rl_metrics

NAMES = [
    "vllm:rl_weight_update_operation_duration_seconds",
    "vllm:rl_weight_update_operations_in_flight",
]
assert rl_metrics._metrics is None
assert not set(NAMES) & set(REGISTRY._names_to_collectors)

metrics = rl_metrics.weight_operation_metrics()
assert REGISTRY._names_to_collectors[NAMES[0]] is metrics.duration
assert REGISTRY._names_to_collectors[NAMES[1]] is metrics.in_flight
assert rl_metrics.weight_operation_metrics() is metrics
print("lazy registration ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "lazy registration ok" in result.stdout
