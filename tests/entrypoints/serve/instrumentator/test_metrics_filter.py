# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import (
    CONTENT_TYPE_PLAIN_0_0_4,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

from vllm.entrypoints.serve.instrumentator import metrics as metrics_endpoint
from vllm.v1.metrics import prometheus


class _LateCollector:
    def __init__(self):
        self.metrics = []

    def collect(self):
        yield from self.metrics


def _record_multiprocess_http_metric():
    registry = CollectorRegistry()
    Counter(
        "http_requests",
        "Total number of requests by method, status and handler.",
        labelnames=["method", "status", "handler"],
        registry=registry,
    ).labels("GET", "2xx", "/test").inc()


@pytest.fixture
def single_process_metrics_client(monkeypatch) -> Iterator[TestClient]:
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    registry = CollectorRegistry(auto_describe=True)
    monkeypatch.setattr(prometheus, "REGISTRY", registry)

    Gauge(
        "vllm:num_requests_waiting",
        "Number of requests waiting to be processed.",
        registry=registry,
    ).set(2)
    Counter(
        "vllm:tokens",
        "Number of processed tokens.",
        registry=registry,
    ).inc(10)
    Gauge(
        "vllm:unrequested",
        "Metric that should not be returned.",
        registry=registry,
    ).set(3)
    Histogram(
        "vllm:request_latency_seconds",
        "Request latency.",
        buckets=(0.5, 1.0, 2.0),
        registry=registry,
    ).observe(0.75)

    app = FastAPI()
    metrics_endpoint.attach_router(app)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    with TestClient(app) as client:
        yield client


@pytest.fixture
def multiprocess_metrics_client(monkeypatch, tmp_path) -> Iterator[TestClient]:
    collector = _LateCollector()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    monkeypatch.setattr(
        prometheus.multiprocess,
        "MultiProcessCollector",
        lambda registry: registry.register(collector),
    )

    registry = prometheus.get_prometheus_registry()
    monkeypatch.setattr(metrics_endpoint, "get_prometheus_registry", lambda: registry)

    app = FastAPI()
    metrics_endpoint.attach_router(app)
    http_requests = CounterMetricFamily(
        "http_requests",
        "Total number of requests by method, status and handler.",
        labels=["method", "status", "handler"],
    )
    http_requests.add_metric(["GET", "2xx", "/test"], 1)
    collector.metrics = [
        GaugeMetricFamily(
            "vllm:num_requests_waiting",
            "Number of requests waiting to be processed.",
            value=2,
        ),
        CounterMetricFamily(
            "vllm:tokens",
            "Number of processed tokens.",
            value=10,
            created=1,
        ),
        GaugeMetricFamily(
            "vllm:unrequested",
            "Metric that should not be returned.",
            value=3,
        ),
        http_requests,
    ]

    with TestClient(app) as client:
        yield client


def test_single_process_unfiltered_metrics_remain_available(
    single_process_metrics_client,
):
    single_process_metrics_client.get("/test")
    response = single_process_metrics_client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_PLAIN_0_0_4
    assert "content-encoding" not in response.headers
    assert "vllm:num_requests_waiting 2.0" in response.text
    assert "vllm:unrequested 3.0" in response.text
    assert "http_requests_total{" in response.text


def test_single_process_name_parameters_filter_metrics(single_process_metrics_client):
    response = single_process_metrics_client.get(
        "/metrics",
        params=[
            ("name[]", "vllm:num_requests_waiting"),
            ("name[]", "vllm:tokens_total"),
        ],
    )

    assert response.status_code == 200
    assert "vllm:num_requests_waiting 2.0" in response.text
    assert "vllm:tokens_total 10.0" in response.text
    assert "vllm:tokens_created" not in response.text
    assert "vllm:unrequested" not in response.text


@pytest.mark.parametrize("method", ["HEAD", "POST", "PUT", "OPTIONS"])
def test_metrics_preserves_existing_non_get_behavior(
    single_process_metrics_client, method
):
    response = single_process_metrics_client.request(method, "/metrics")

    assert response.status_code == 200


def test_histogram_family_name_does_not_select_samples(
    single_process_metrics_client,
):
    response = single_process_metrics_client.get(
        "/metrics", params=[("name[]", "vllm:request_latency_seconds")]
    )

    assert response.status_code == 200
    assert response.content == b""


def test_histogram_sample_names_can_be_filtered(single_process_metrics_client):
    response = single_process_metrics_client.get(
        "/metrics",
        params=[
            ("name[]", "vllm:request_latency_seconds_bucket"),
            ("name[]", "vllm:request_latency_seconds_sum"),
        ],
    )

    assert response.status_code == 200
    assert "vllm:request_latency_seconds_bucket{" in response.text
    assert "vllm:request_latency_seconds_sum " in response.text
    assert "vllm:request_latency_seconds_count " not in response.text


def test_openmetrics_content_negotiation(single_process_metrics_client):
    response = single_process_metrics_client.get(
        "/metrics",
        headers={"Accept": "application/openmetrics-text; version=1.0.0"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/openmetrics-text; version=1.0.0"
    )
    assert response.text.endswith("# EOF\n")


def test_name_parameter_filters_late_registered_metrics(
    multiprocess_metrics_client,
):
    response = multiprocess_metrics_client.get(
        "/metrics",
        params=[
            ("name[]", "vllm:num_requests_waiting"),
            ("name[]", "vllm:tokens_total"),
        ],
    )

    assert response.status_code == 200
    assert "vllm:num_requests_waiting 2.0" in response.text
    assert "vllm:tokens_total 10.0" in response.text
    assert "vllm:tokens_created" not in response.text
    assert "vllm:unrequested" not in response.text


def test_throwaway_registry_metrics_are_collected_from_mmap(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    process = multiprocessing.get_context("spawn").Process(
        target=_record_multiprocess_http_metric
    )
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Prometheus metric writer process did not exit")
    assert process.exitcode == 0

    registry = CollectorRegistry(support_collectors_without_names=True)
    prometheus.multiprocess.MultiProcessCollector(registry)
    output = generate_latest(
        registry.restricted_registry(["http_requests_total"])
    ).decode()

    assert output.count("# HELP http_requests_total ") == 1
    assert output.count("# TYPE http_requests_total counter") == 1
    assert output.count("http_requests_total{") == 1


def test_unknown_metric_name_returns_empty_response(multiprocess_metrics_client):
    response = multiprocess_metrics_client.get(
        "/metrics", params=[("name[]", "vllm:does_not_exist")]
    )

    assert response.status_code == 200
    assert response.content == b""


def test_multiprocess_http_metric_is_emitted_once(multiprocess_metrics_client):
    response = multiprocess_metrics_client.get(
        "/metrics", params=[("name[]", "http_requests_total")]
    )

    assert response.status_code == 200
    assert response.text.count("# HELP http_requests_total ") == 1
    assert response.text.count("# TYPE http_requests_total counter") == 1
    assert response.text.count("http_requests_total{") == 1
