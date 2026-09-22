# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end wiring of the production recorder without multiprocess setup.

`PROMETHEUS_MULTIPROC_DIR` is unset here on purpose: this is what a default
`vllm serve` process looks like, and the metrics have to be registered and
scrapable on the registry that `/metrics` is served from.

The recorder is the production singleton, so label values are namespaced per test
to keep the assertions independent of test order.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY, generate_latest

from vllm.entrypoints.serve.dev.rlhf import api_router
from vllm.entrypoints.serve.dev.rlhf import metrics as rlhf_metrics

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
COUNTER = "vllm:rl_weight_update_operations_total"

# Dedicated label values so tests do not depend on each other's counters.
OP_OK = "single_process_ok"
OP_ENGINE_ERROR = "single_process_engine_error"
OP_INVALID = "single_process_invalid"

ROUTES = {
    "init": api_router.init_weight_transfer_engine,
    "update": api_router.update_weights,
    "start": api_router.start_weight_update,
}


class _Request:
    def __init__(self, engine, body):
        self.app = SimpleNamespace(state=SimpleNamespace(engine_client=engine))
        self._body = body

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Engine:
    def __init__(self, fail_start: bool = False):
        self.calls: list[str] = []
        self.fail_start = fail_start

    async def update_weights(self, *args, **kwargs):
        self.calls.append("update_weights")

    async def init_weight_transfer_engine(self, *args, **kwargs):
        self.calls.append("init_weight_transfer_engine")

    async def start_weight_update(self, *args, **kwargs):
        self.calls.append("start_weight_update")
        if self.fail_start:
            raise RuntimeError("engine down")


class _RejectingEngine(_Engine):
    async def update_weights(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("engine reached for an invalid payload")

    async def init_weight_transfer_engine(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("engine reached for an invalid payload")


def _counter_value(operation: str, status: str) -> float:
    value = rlhf_metrics.weight_operation_metrics().operations.labels(
        operation=operation, status=status
    )
    return value._value.get()


def _reset_counter(operation: str) -> None:
    """Zero this operation's counter so assertions do not depend on test order."""
    metrics = rlhf_metrics.weight_operation_metrics()
    for status in ("success", "error"):
        metrics.operations.labels(operation=operation, status=status)._value.set(0)


def test_production_singleton_registers_on_the_served_registry(monkeypatch):
    """``registry=None`` makes prometheus_client skip registration entirely."""
    monkeypatch.setattr(rlhf_metrics, "_metrics", None)

    metrics = rlhf_metrics.weight_operation_metrics()
    assert metrics is rlhf_metrics.weight_operation_metrics()

    for name in (
        COUNTER,
        "vllm:rl_weight_update_operation_duration_seconds",
        "vllm:rl_weight_update_operations_in_flight",
    ):
        assert name in REGISTRY._names_to_collectors, name


def test_metrics_are_scrapable_without_multiprocess(monkeypatch):
    """No PROMETHEUS_MULTIPROC_DIR: the route must still be visible in /metrics."""
    assert "PROMETHEUS_MULTIPROC_DIR" not in os.environ

    response = asyncio.run(ROUTES["start"](_Request(_Engine(), {})))

    assert response.status_code == 200
    expected = _counter_value("start", "success")
    assert expected >= 1
    text = generate_latest(REGISTRY).decode()
    assert f'{COUNTER}{{operation="start",status="success"}} {expected}' in text, (
        "the RL metrics are not exported by the default registry"
    )


def test_production_singleton_is_not_recreated(monkeypatch):
    monkeypatch.setattr(rlhf_metrics, "_metrics", None)
    first = rlhf_metrics.weight_operation_metrics()
    assert rlhf_metrics.weight_operation_metrics() is first


@pytest.mark.asyncio
async def test_engine_error_is_counted_through_the_singleton():
    engine = _Engine(fail_start=True)

    with pytest.raises(RuntimeError, match="engine down"):
        await ROUTES["start"](_Request(engine, {}))

    assert _counter_value("start", "error") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,body",
    [
        # missing / null
        ("/init_weight_transfer_engine", {}),
        ("/init_weight_transfer_engine", {"init_info": None}),
        ("/update_weights", {}),
        ("/update_weights", {"update_info": None}),
        # wrong shape: rejected by the dataclasses and by the Rust frontend
        ("/init_weight_transfer_engine", {"init_info": 1}),
        ("/init_weight_transfer_engine", {"init_info": []}),
        ("/init_weight_transfer_engine", {"init_info": [{"rank": 0}]}),
        ("/init_weight_transfer_engine", {"init_info": "x"}),
        ("/update_weights", {"update_info": 1}),
        ("/update_weights", {"update_info": "x"}),
        ("/update_weights", {"update_info": [{}, None]}),
        ("/update_weights", {"update_info": [1]}),
        ("/update_weights", {"update_info": [None]}),
    ],
)
async def test_invalid_payload_records_nothing(path, body):
    operation = "init" if "init" in path else "update"
    route = ROUTES[operation]
    _reset_counter(operation)

    with pytest.raises(HTTPException) as exc_info:
        await route(_Request(_RejectingEngine(), body))

    assert exc_info.value.status_code == 400
    # The recorder must not have been entered, so neither status moved.
    assert _counter_value(operation, "error") == 0
    assert _counter_value(operation, "success") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,body",
    [
        ("/update_weights", {"update_info": {}}),
        ("/update_weights", {"update_info": [{"rank": 0}]}),
        ("/init_weight_transfer_engine", {"init_info": {}}),
        ("/init_weight_transfer_engine", {"init_info": {"rank": 0}}),
    ],
)
async def test_valid_payload_shapes_are_recorded(path, body):
    operation = "init" if "init" in path else "update"
    engine = _Engine()
    _reset_counter(operation)

    response = await ROUTES[operation](_Request(engine, body))

    assert response.status_code == 200
    assert engine.calls, "engine was not called for a valid payload"
    assert _counter_value(operation, "success") == 1


@pytest.mark.asyncio
async def test_malformed_json_is_rejected_without_recording():
    request = _Request(_RejectingEngine(), json.JSONDecodeError("bad", "{}", 0))
    _reset_counter("update")

    with pytest.raises(HTTPException) as exc_info:
        await ROUTES["update"](request)

    assert exc_info.value.status_code == 400
    assert _counter_value("update", "error") == 0
