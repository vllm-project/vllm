# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from prometheus_client import CollectorRegistry

from vllm.entrypoints.serve.dev.rlhf import api_router
from vllm.entrypoints.serve.dev.rlhf.metrics import WeightOperationMetrics

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
PREFIX = "vllm:rl_weight_update_"


class _Request:
    def __init__(self, engine, body):
        self.app = SimpleNamespace(state=SimpleNamespace(engine_client=engine))
        self._body = body

    async def json(self):
        return self._body


class _BlockingEngine:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def update_weights(self, request):
        self.started.set()
        await self.release.wait()


@pytest.mark.asyncio
async def test_update_weights_route_records_in_flight_and_success(monkeypatch):
    registry = CollectorRegistry()
    metrics = WeightOperationMetrics(registry)
    monkeypatch.setattr(api_router, "_weight_metrics", lambda: metrics)
    engine = _BlockingEngine()
    request = _Request(engine, {"update_info": {"names": ["model.weight"]}})
    labels = {"operation": "update"}

    task = asyncio.create_task(api_router.update_weights(request))
    await engine.started.wait()
    assert registry.get_sample_value(PREFIX + "requests_in_flight", labels) == 1

    engine.release.set()
    response = await task
    assert response.status_code == 200
    assert registry.get_sample_value(PREFIX + "requests_in_flight", labels) == 0
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total", {**labels, "status": "success"}
        )
        == 1
    )
    assert (
        registry.get_sample_value(PREFIX + "request_duration_seconds_count", labels)
        == 1
    )


@pytest.mark.asyncio
async def test_update_weights_route_rejects_before_recording(monkeypatch):
    registry = CollectorRegistry()
    metrics = WeightOperationMetrics(registry)
    monkeypatch.setattr(api_router, "_weight_metrics", lambda: metrics)
    request = _Request(_BlockingEngine(), {})

    with pytest.raises(HTTPException) as exc_info:
        await api_router.update_weights(request)

    assert exc_info.value.status_code == 400
    assert (
        registry.get_sample_value(
            PREFIX + "requests_total",
            {"operation": "update", "status": "success"},
        )
        is None
    )
