# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.serve.dev.sleep import metrics as sleep_metrics
from vllm.entrypoints.serve.dev.sleep.api_router import attach_router
from vllm.entrypoints.serve.dev.sleep.metrics import SleepModeOperationMetrics
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.entrypoints.serve.instrumentator.metrics import (
    attach_router as attach_metrics_router,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.metrics import prometheus as prometheus_metrics

MODEL_NAME = "meta-llama/Llama-3.2-1B"


@pytest.fixture
def sleep_route_app(monkeypatch):
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    app.state.engine_client = AsyncMock()
    metrics = SleepModeOperationMetrics(CollectorRegistry())
    monkeypatch.setattr(
        "vllm.entrypoints.serve.dev.sleep.api_router.sleep_mode_operation_metrics",
        lambda: metrics,
    )

    attach_router(app)
    init_exception_handler(app)
    return app, metrics


@pytest.mark.cpu_test
@pytest.mark.parametrize("level", [0, 1, 2])
def test_sleep_route_response_and_engine_arguments(sleep_route_app, level):
    app, _ = sleep_route_app
    with TestClient(app) as client:
        response = client.post("/sleep", params={"level": level})
    assert response.status_code == 200
    assert response.json() == {"status": "sleeping", "level": level}
    app.state.engine_client.sleep.assert_awaited_once_with(level, "abort")


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("query", "expected_param"),
    [
        ("level=invalid", "query.level"),
        ("level=-1", "query.level"),
        ("level=3", "query.level"),
        ("mode=invalid", "query.mode"),
    ],
)
def test_sleep_route_rejects_invalid_query_before_dispatch(
    sleep_route_app, query, expected_param
):
    app, metrics = sleep_route_app
    with TestClient(app) as client:
        response = client.post(f"/sleep?{query}")
    assert response.status_code == 400
    assert response.json()["error"]["param"] == expected_param
    app.state.engine_client.sleep.assert_not_awaited()
    assert list(metrics.operations.collect()[0].samples) == []
    assert list(metrics.duration.collect()[0].samples) == []
    assert list(metrics.in_flight.collect()[0].samples) == []


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("fully_awake", "tags", "expected"),
    [
        (True, "", {"status": "awake", "tags_woken": None}),
        (False, "?tags=weights", {"status": "sleeping", "tags_woken": ["weights"]}),
    ],
)
def test_wake_route_maps_engine_result_without_state_query(
    sleep_route_app, fully_awake, tags, expected
):
    app, _ = sleep_route_app
    app.state.engine_client.wake_up.return_value = fully_awake
    with TestClient(app) as client:
        response = client.post(f"/wake_up{tags}")
    assert response.status_code == 200
    assert response.json() == expected
    app.state.engine_client.wake_up.assert_awaited_once_with(expected["tags_woken"])
    app.state.engine_client.is_sleeping.assert_not_awaited()


@pytest.mark.cpu_test
@pytest.mark.asyncio
@pytest.mark.parametrize("fully_awake", [True, False])
async def test_async_llm_wake_returns_engine_result(fully_awake):
    llm = SimpleNamespace(
        engine_core=SimpleNamespace(wake_up_async=AsyncMock(return_value=fully_awake)),
        logger_manager=Mock(),
    )
    assert await AsyncLLM.wake_up(llm, ["weights"]) is fully_awake
    llm.engine_core.wake_up_async.assert_awaited_once_with(["weights"])
    assert llm.logger_manager.record_sleep_state.call_count == int(fully_awake)


@pytest.mark.cpu_test
@pytest.mark.parametrize("fully_awake", [True, False])
def test_llm_engine_wake_returns_engine_result(fully_awake):
    llm = SimpleNamespace(
        engine_core=SimpleNamespace(wake_up=Mock(return_value=fully_awake)),
        logger_manager=Mock(),
    )
    assert LLMEngine.wake_up(llm, ["weights"]) is fully_awake
    llm.engine_core.wake_up.assert_called_once_with(["weights"])
    assert llm.logger_manager.record_sleep_state.call_count == int(fully_awake)


@pytest.mark.cpu_test
@pytest.mark.parametrize("fails", [False, True])
def test_release_kv_cache_memory_route(sleep_route_app, fails):
    app, metrics = sleep_route_app
    release = app.state.engine_client.release_kv_cache_memory
    if fails:
        release.side_effect = RuntimeError("requires a completed pause first")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/release_kv_cache_memory")

    assert response.status_code == (500 if fails else 200)
    if not fails:
        assert response.json() == {"status": "kv_cache_released"}
    release.assert_awaited_once_with()
    assert (
        metrics.operations.labels(
            "release_kv_cache_memory", "error" if fails else "success"
        )._value.get()
        == 1
    )
    assert metrics.in_flight.labels("release_kv_cache_memory")._value.get() == 0


@pytest.mark.cpu_test
@pytest.mark.parametrize("operation", ["sleep", "release_kv_cache_memory", "wake"])
def test_sleep_mode_recorder_tracks_success_error_and_in_flight(operation):
    metrics = SleepModeOperationMetrics(CollectorRegistry())
    with metrics.record(operation):
        assert metrics.in_flight.labels(operation)._value.get() == 1
    with pytest.raises(RuntimeError), metrics.record(operation):
        raise RuntimeError("engine failed")
    assert metrics.in_flight.labels(operation)._value.get() == 0
    assert metrics.operations.labels(operation, "success")._value.get() == 1
    assert metrics.operations.labels(operation, "error")._value.get() == 1
    assert metrics.duration.labels(operation)._sum.get() >= 0
    count = next(
        sample.value
        for sample in metrics.duration.collect()[0].samples
        if sample.name.endswith("_count") and sample.labels == {"operation": operation}
    )
    assert count == 2


@pytest.mark.cpu_test
def test_sleep_operation_visible_on_production_metrics_endpoint(monkeypatch):
    registry = CollectorRegistry()
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.setattr(sleep_metrics, "REGISTRY", registry)
    monkeypatch.setattr(sleep_metrics, "_metrics", None)
    monkeypatch.setattr(prometheus_metrics, "REGISTRY", registry)

    app = FastAPI()
    app.state.engine_client = AsyncMock()
    attach_router(app)
    attach_metrics_router(app)

    with TestClient(app) as client:
        assert client.post("/sleep").status_code == 200
        response = client.get("/metrics")

    assert response.status_code == 200
    samples = [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    ]
    assert any(
        sample.name == "vllm:rl_sleep_mode_operations_total"
        and sample.labels == {"operation": "sleep", "status": "success"}
        and sample.value == 1
        for sample in samples
    )
    assert any(
        sample.name == "vllm:rl_sleep_mode_operations_in_flight"
        and sample.labels == {"operation": "sleep"}
        and sample.value == 0
        for sample in samples
    )


def test_sleep_mode():
    # dtype, max-len etc set so that this can run in CI
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "128",
        "--enable-sleep-mode",
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        env_dict={"VLLM_SERVER_DEV_MODE": "1", "CUDA_VISIBLE_DEVICES": "0"},
    ) as remote_server:
        response = requests.post(remote_server.url_for("sleep"), params={"level": "1"})
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True

        # check sleep metrics
        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        awake, weights_offloaded, discard_all = _get_sleep_metrics_from_api(response)
        assert awake == 0
        assert weights_offloaded == 1
        assert discard_all == 0

        response = requests.post(remote_server.url_for("wake_up"))
        assert response.status_code == 200
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is False

        # check sleep metrics
        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        awake, weights_offloaded, discard_all = _get_sleep_metrics_from_api(response)
        assert awake == 1
        assert weights_offloaded == 0
        assert discard_all == 0

        # test wake up with tags
        response = requests.post(remote_server.url_for("sleep"), params={"level": "1"})
        assert response.status_code == 200

        response = requests.post(
            remote_server.url_for("wake_up"), params={"tags": ["weights"]}
        )
        assert response.status_code == 200

        # Partial wake keeps the engine sleeping.
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True

        response = requests.post(
            remote_server.url_for("wake_up"), params={"tags": ["kv_cache"]}
        )
        assert response.status_code == 200

        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is False

        # check sleep metrics
        response = requests.get(remote_server.url_for("metrics"))
        assert response.status_code == 200
        awake, weights_offloaded, discard_all = _get_sleep_metrics_from_api(response)
        assert awake == 1
        assert weights_offloaded == 0
        assert discard_all == 0


def _get_sleep_metrics_from_api(response: requests.Response):
    """Return (awake, weights_offloaded, discard_all)."""
    awake, weights_offloaded, discard_all = None, None, None

    for family in text_string_to_metric_families(response.text):
        if family.name == "vllm:engine_sleep_state":
            for sample in family.samples:
                if sample.name == "vllm:engine_sleep_state":
                    for label_name, label_value in sample.labels.items():
                        if label_value == "awake":
                            awake = sample.value
                        elif label_value == "weights_offloaded":
                            weights_offloaded = sample.value
                        elif label_value == "discard_all":
                            discard_all = sample.value

    assert awake is not None
    assert weights_offloaded is not None
    assert discard_all is not None

    return awake, weights_offloaded, discard_all
