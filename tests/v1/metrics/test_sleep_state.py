# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
from prometheus_client import CollectorRegistry, Gauge, generate_latest

from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import PrometheusStatLogger, StatLoggerManager
from vllm.v1.metrics.stats import EngineSleepState, SchedulerStats

pytestmark = pytest.mark.cpu_test


class FakeExecutor:
    sleep = Executor.sleep
    wake_up = Executor.wake_up
    discard = Executor.discard
    is_sleeping = Executor.is_sleeping

    def __init__(self):
        self.sleeping_tags = set()
        self.sleep_resource_states = {"weights": "resident", "kv_cache": "resident"}
        self.collective_rpc = Mock()


def test_resource_transitions_and_noop_wake():
    executor = FakeExecutor()
    executor.sleep(1)
    assert executor.sleep_resource_states == {
        "weights": "offloaded",
        "kv_cache": "released",
    }
    executor.wake_up(["weights"])
    assert executor.sleep_resource_states == {
        "weights": "resident",
        "kv_cache": "released",
    }
    executor.wake_up(["invalid"])
    assert executor.sleep_resource_states["kv_cache"] == "released"
    executor.wake_up(["kv_cache"])
    assert not executor.is_sleeping
    executor.discard(("kv_cache",))
    assert executor.sleep_resource_states == {
        "weights": "resident",
        "kv_cache": "released",
    }
    executor.wake_up()
    executor.sleep(2)
    assert executor.sleep_resource_states["weights"] == "discarded"


@pytest.mark.parametrize("operation", ["sleep", "wake", "discard"])
def test_failed_rpc_is_unknown(operation):
    executor = FakeExecutor()
    if operation == "wake":
        executor.sleep(1)
    executor.collective_rpc.side_effect = RuntimeError("worker failed")
    with pytest.raises(RuntimeError, match="worker failed"):
        if operation == "sleep":
            executor.sleep(1)
        elif operation == "wake":
            executor.wake_up(["weights"])
        else:
            executor.discard(("kv_cache",))
    affected = "kv_cache" if operation == "discard" else "weights"
    assert executor.sleep_resource_states[affected] == "unknown"


def make_logger():
    registry = CollectorRegistry()
    logger = object.__new__(PrometheusStatLogger)
    resource = Gauge(
        "vllm:engine_sleep_resource_state",
        "Resource state",
        ["engine", "resource", "state"],
        registry=registry,
    )
    logger.gauge_sleep_resource_state = {
        (name, state): {idx: resource.labels(idx, name, state) for idx in [0, 1]}
        for name, states in {
            "scheduler": ("running", "paused"),
            "weights": ("resident", "offloaded", "discarded", "unknown"),
            "kv_cache": ("resident", "released", "unknown"),
        }.items()
        for state in states
    }
    awake = Gauge(
        "vllm:engine_fully_awake", "Fully awake", ["engine"], registry=registry
    )
    logger.gauge_fully_awake = {idx: awake.labels(idx) for idx in [0, 1]}
    legacy = Gauge(
        "vllm:engine_sleep_state",
        "Legacy",
        ["engine", "sleep_state"],
        registry=registry,
    )
    logger.gauge_engine_sleep_state = {
        state: {idx: legacy.labels(idx, state) for idx in [0, 1]}
        for state in ["awake", "weights_offloaded", "discard_all"]
    }
    return logger, registry


def test_partial_wake_scrape_clears_stale_flags_and_isolates_dp_engine():
    logger, registry = make_logger()
    logger.record_sleep_snapshot(EngineSleepState(True, "offloaded", "released"), 0)
    logger.record_sleep_snapshot(EngineSleepState(), 1)
    logger.record_sleep_snapshot(EngineSleepState(True, "resident", "released"), 0)
    assert (
        registry.get_sample_value(
            "vllm:engine_sleep_state",
            {"engine": "0", "sleep_state": "weights_offloaded"},
        )
        == 0
    )
    assert registry.get_sample_value("vllm:engine_fully_awake", {"engine": "0"}) == 0
    assert registry.get_sample_value("vllm:engine_fully_awake", {"engine": "1"}) == 1
    assert b'resource="weights",state="resident"} 1.0' in generate_latest(registry)


@pytest.mark.parametrize(
    "state",
    [
        EngineSleepState(True),  # Level zero.
        EngineSleepState(True, "resident", "released"),  # KV release / partial wake.
        EngineSleepState(True, "discarded", "released"),  # Level two.
        EngineSleepState(False, "unknown", "resident"),  # Failed memory operation.
    ],
)
def test_resource_states_are_not_fully_awake(state):
    assert not state.fully_awake


def test_idle_state_snapshot_broadcast_and_logger_dispatch():
    engine = object.__new__(EngineCoreProc)
    engine.log_stats = True
    engine.output_queue = Queue()
    engine.addresses = SimpleNamespace(outputs=["frontend0", "frontend1"])
    engine.scheduler = SimpleNamespace(pause_state=None)
    engine.get_sleep_state = lambda: dict(
        scheduler_paused=True, weights="resident", kv_cache="released"
    )
    engine._publish_sleep_state()
    manager = object.__new__(StatLoggerManager)
    logger = Mock()
    manager.stat_loggers = [logger]
    for idx in [0, 1]:
        client_idx, output = engine.output_queue.get_nowait()
        assert client_idx == idx
        assert output.scheduler_stats.sleep_state_only
        manager.record(output.scheduler_stats, None, engine_idx=idx)
        logger.record_sleep_snapshot.assert_called_with(
            EngineSleepState(True, "resident", "released"), idx
        )
    logger.record.assert_not_called()
    assert engine.output_queue.empty()


def test_state_snapshot_does_not_reset_existing_metrics():
    manager = object.__new__(StatLoggerManager)
    normal = Mock()
    manager.stat_loggers = [normal]
    manager.record(
        SchedulerStats(sleep_state=EngineSleepState(True), sleep_state_only=True),
        None,
        engine_idx=0,
    )
    normal.record.assert_not_called()
    normal.record_sleep_snapshot.assert_called_once_with(EngineSleepState(True), 0)


def test_new_series_absent_until_first_snapshot():
    registry = CollectorRegistry()
    logger = object.__new__(PrometheusStatLogger)
    logger._sleep_resource_metric = Gauge(
        "vllm:engine_sleep_resource_state",
        "Resources",
        ["engine", "model_name", "resource", "state"],
        registry=registry,
    )
    logger._fully_awake_metric = Gauge(
        "vllm:engine_fully_awake",
        "Awake",
        ["engine", "model_name"],
        registry=registry,
    )
    logger._sleep_model_name = "test"
    logger._sleep_labelvalues = {0: ["0", "test"]}
    logger.gauge_sleep_resource_state = {}
    logger.gauge_fully_awake = {}
    logger.gauge_engine_sleep_state = {
        state: {0: Mock()} for state in ["awake", "weights_offloaded", "discard_all"]
    }
    assert (
        registry.get_sample_value(
            "vllm:engine_fully_awake", {"engine": "0", "model_name": "test"}
        )
        is None
    )
    logger.record_sleep_snapshot(EngineSleepState(), 0)
    assert (
        registry.get_sample_value(
            "vllm:engine_fully_awake", {"engine": "0", "model_name": "test"}
        )
        == 1
    )


def test_scheduler_snapshot_msgpack_backward_compatibility():
    old = msgspec.msgpack.encode({"num_running_reqs": 3})
    decoded = msgspec.msgpack.decode(old, type=SchedulerStats)
    assert decoded.num_running_reqs == 3
    assert decoded.sleep_state is None
    assert not decoded.sleep_state_only
    state = EngineSleepState(True, "resident", "released")
    snapshot = SchedulerStats(sleep_state=state, sleep_state_only=True)
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(snapshot), type=SchedulerStats
    )
    assert decoded.sleep_state == state
    assert decoded.sleep_state_only
