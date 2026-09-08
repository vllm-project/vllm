# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from vllm.entrypoints.serve.instrumentator import metrics


def _engine_client(health: dict[int, bool], enabled: bool = True):
    client = Mock()
    client.vllm_config.parallel_config.enable_fault_tolerance = enabled
    client.get_engine_health.return_value = health
    return client


def _app(client):
    app = FastAPI()
    app.state.engine_client = client
    metrics.attach_router(app)
    return app


def _samples(client, path="/metrics"):
    response = client.get(path)
    assert response.status_code == 200
    return [
        sample
        for family in text_string_to_metric_families(response.text)
        if family.name == "vllm:engine_healthy"
        for sample in family.samples
    ]


@pytest.mark.parametrize("path", ["/metrics", "/metrics/"])
def test_ft_metrics_follow_local_health(monkeypatch, path):
    monkeypatch.setattr(metrics, "get_prometheus_registry", CollectorRegistry)
    health = {2: True}
    engine = _engine_client(health)
    with TestClient(_app(engine)) as client:
        for healthy in (True, False, True, False):
            health[2] = healthy
            samples = _samples(client, path)
            assert len(samples) == 1
            assert samples[0].labels == {"engine": "2"}
            assert samples[0].value == int(healthy)


@pytest.mark.parametrize("client", [None, _engine_client({0: True}, enabled=False)])
def test_ft_metrics_absent_without_ft(monkeypatch, client):
    monkeypatch.setattr(metrics, "get_prometheus_registry", CollectorRegistry)
    with TestClient(_app(client)) as http_client:
        assert _samples(http_client) == []
    if client is not None:
        client.get_engine_health.assert_not_called()


def test_ft_metrics_do_not_retain_missing_status(monkeypatch):
    monkeypatch.setattr(metrics, "get_prometheus_registry", CollectorRegistry)
    health = {0: True}
    with TestClient(_app(_engine_client(health))) as client:
        assert _samples(client)[0].value == 1
        health.clear()
        assert _samples(client) == []


@pytest.mark.parametrize("path", ["/metrics", "/metrics/"])
def test_ft_metrics_are_local_with_shared_multiprocess_metrics(
    monkeypatch, tmp_path, path
):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    # Leave an ordinary metric behind after its writer exits, as the shared
    # registry does in a multi-port DP deployment.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from prometheus_client import Gauge; "
            "Gauge('existing_metric', 'Existing metric').set(7)",
        ],
        env=os.environ.copy(),
        check=True,
    )
    health = {0: True}
    app0 = _app(_engine_client(health))
    app1 = _app(_engine_client({1: True}))
    with TestClient(app0) as client0, TestClient(app1) as client1:
        assert _samples(client0, path)[0].labels == {"engine": "0"}
        health[0] = False
        assert _samples(client0, path)[0].value == 0
        samples = _samples(client1, path)
        assert len(samples) == 1
        assert samples[0].labels == {"engine": "1"}
        assert samples[0].value == 1
        response = client1.get("/metrics")
        existing = [
            sample
            for family in text_string_to_metric_families(response.text)
            if family.name == "existing_metric"
            for sample in family.samples
        ]
        assert len(existing) == 1
        assert existing[0].value == 7

    # A later API instance cannot inherit the healthy value from the old one.
    with TestClient(_app(_engine_client({1: False}))) as replacement:
        assert _samples(replacement, path)[0].value == 0
