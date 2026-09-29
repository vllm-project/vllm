# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY, CollectorRegistry

from vllm.entrypoints.serve.dev.rlhf import api_router
from vllm.entrypoints.serve.dev.rlhf import metrics as rlhf_metrics
from vllm.entrypoints.serve.dev.rlhf.metrics import WeightOperationMetrics

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

DURATION = "vllm:rl_weight_update_operation_duration_seconds_count"
IN_FLIGHT = "vllm:rl_weight_update_operations_in_flight"


def sample(registry, name, operation):
    return registry.get_sample_value(name, {"operation": operation})


class Request:
    def __init__(self, engine, body=None):
        self.app = SimpleNamespace(state=SimpleNamespace(engine_client=engine))
        self._body = body

    async def json(self):
        return self._body


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
    monkeypatch.setattr(rlhf_metrics, "_metrics", WeightOperationMetrics(registry))
    return registry


@pytest.mark.parametrize("fail", [False, True])
def test_record_releases_in_flight_and_observes_duration(registry, fail):
    metrics = rlhf_metrics.weight_operation_metrics()
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
    task = asyncio.create_task(route(Request(engine, body)))
    await engine.started.wait()
    assert sample(registry, IN_FLIGHT, operation) == 1
    engine.release.set()
    await task
    assert sample(registry, IN_FLIGHT, operation) == 0
    assert sample(registry, DURATION, operation) == 1


@pytest.mark.asyncio
async def test_rejected_request_is_not_recorded(registry):
    with pytest.raises(HTTPException):
        await api_router.update_weights(Request(Engine(), {}))
    assert sample(registry, DURATION, "update") is None


def test_default_recorder_is_created_lazily_on_the_default_registry(monkeypatch):
    monkeypatch.setattr(rlhf_metrics, "_metrics", None)
    names = set(REGISTRY._names_to_collectors)
    assert "vllm:rl_weight_update_operation_duration_seconds" not in names
    metrics = rlhf_metrics.weight_operation_metrics()
    try:
        assert (
            REGISTRY._names_to_collectors[
                "vllm:rl_weight_update_operation_duration_seconds"
            ]
            is metrics.duration
        )
    finally:
        REGISTRY.unregister(metrics.duration)
        REGISTRY.unregister(metrics.in_flight)
