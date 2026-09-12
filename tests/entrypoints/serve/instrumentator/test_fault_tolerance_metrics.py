# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from vllm.entrypoints.serve.instrumentator import metrics


def _engine_client(engines: list[dict], enabled: bool = True):
    client = Mock(errored=False)
    client.vllm_config.parallel_config.enable_fault_tolerance = enabled
    client.get_status = AsyncMock(return_value={"engines": engines})
    return client


def _app(client):
    app = FastAPI()
    app.state.engine_client = client
    metrics.attach_router(app)
    return app


def _samples(client, path="/metrics", name="vllm:engine_healthy"):
    response = client.get(path)
    assert response.status_code == 200
    return [
        sample
        for family in text_string_to_metric_families(response.text)
        if family.name == name
        for sample in family.samples
    ]


@pytest.mark.parametrize("path", ["/metrics", "/metrics/"])
def test_ft_metrics_follow_local_health(monkeypatch, path):
    monkeypatch.setattr(metrics, "get_prometheus_registry", CollectorRegistry)
    engines = [{"id": 2, "status": "healthy"}]
    engine_client = _engine_client(engines)
    with TestClient(_app(engine_client)) as client:
        for status, errored, expected in (
            ("healthy", False, 1),
            ("unhealthy", False, 0),
            ("dead", False, 0),
            ("healthy", True, 0),
            ("healthy", False, 1),
        ):
            engines[0]["status"] = status
            engine_client.errored = errored
            samples = _samples(client, path)
            assert len(samples) == 1
            assert samples[0].labels == {"engine": "2"}
            assert samples[0].value == expected
        engines.clear()
        assert _samples(client, path) == []


@pytest.mark.parametrize("client", [None, _engine_client([], enabled=False)])
def test_ft_metrics_absent_without_ft(monkeypatch, client):
    monkeypatch.setattr(metrics, "get_prometheus_registry", CollectorRegistry)
    with TestClient(_app(client)) as http_client:
        assert _samples(http_client) == []
    if client is not None:
        client.get_status.assert_not_called()


@pytest.mark.parametrize("path", ["/metrics", "/metrics/"])
def test_ft_metrics_are_local_with_shared_multiprocess_metrics(
    monkeypatch, tmp_path, path
):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from prometheus_client import Gauge; "
            "Gauge('existing_metric', 'Existing metric').set(7)",
        ],
        check=True,
    )
    engines = [{"id": 0, "status": "healthy"}]
    app0 = _app(_engine_client(engines))
    app1 = _app(_engine_client([{"id": 1, "status": "healthy"}]))
    with TestClient(app0) as client0, TestClient(app1) as client1:
        assert _samples(client0, path)[0].labels == {"engine": "0"}
        engines[0]["status"] = "unhealthy"
        assert _samples(client0, path)[0].value == 0
        samples = _samples(client1, path)
        assert len(samples) == 1
        assert samples[0].labels == {"engine": "1"}
        assert samples[0].value == 1
        existing = _samples(client1, path, "existing_metric")
        assert len(existing) == 1
        assert existing[0].value == 7
        assert _samples(client1, path + "?name[]=vllm:engine_healthy") == samples
        assert _samples(client1, path + "?name[]=existing_metric") == []

    with TestClient(_app(_engine_client([{"id": 1, "status": "dead"}]))) as replacement:
        assert _samples(replacement, path)[0].value == 0
