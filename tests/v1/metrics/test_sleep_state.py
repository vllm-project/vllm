# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec
import pytest
from prometheus_client import CollectorRegistry, Gauge, generate_latest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import PrometheusStatLogger, StatLoggerManager
from vllm.v1.metrics.stats import EngineSleepState, SchedulerStats

pytestmark = pytest.mark.cpu_test


class FakeExecutor:
    sleep = Executor.sleep
    wake_up = Executor.wake_up
    discard = Executor.discard
    is_sleeping = Executor.is_sleeping
    all_resources_resident = Executor.all_resources_resident
    _check_sleep_resource_states = Executor._check_sleep_resource_states

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
    assert not executor.all_resources_resident
    assert executor.sleep_resource_states == {
        "weights": "resident",
        "kv_cache": "released",
    }
    executor.wake_up(["invalid"])
    assert executor.sleep_resource_states["kv_cache"] == "released"
    executor.wake_up(["kv_cache"])
    assert not executor.is_sleeping
    assert executor.all_resources_resident
    executor.discard(("kv_cache",))
    assert executor.sleep_resource_states == {
        "weights": "resident",
        "kv_cache": "released",
    }
    executor.wake_up()
    executor.sleep(2)
    assert executor.sleep_resource_states["weights"] == "discarded"


def test_partial_wake_then_sleep_is_rejected():
    executor = FakeExecutor()
    executor.sleep(1)
    executor.wake_up(["weights"])
    with pytest.raises(RuntimeError, match="partially awake"):
        executor.sleep(1)
    assert executor.collective_rpc.call_count == 2
    assert executor.sleep_resource_states == {
        "weights": "resident",
        "kv_cache": "released",
    }


def test_failed_sleep_then_wake_keeps_scheduler_paused_and_metric_zero():
    executor = FakeExecutor()
    executor.collective_rpc.side_effect = RuntimeError("worker failed")
    engine = object.__new__(EngineCore)
    engine.model_executor = executor
    engine.pause_scheduler = Mock(return_value=None)
    engine.resume_scheduler = Mock()
    engine.is_scheduler_paused = Mock(return_value=True)

    with pytest.raises(RuntimeError, match="worker failed"):
        engine.sleep(1)
    assert not executor.all_resources_resident
    with pytest.raises(RuntimeError, match="unknown.*rebuild"):
        engine.wake_up()
    engine.resume_scheduler.assert_not_called()

    logger, registry = make_logger()
    state = engine.get_sleep_state()
    logger.record_sleep_snapshot(
        EngineSleepState(
            scheduler_paused=state["scheduler_paused"],
            weights=state["weights"],
            kv_cache=state["kv_cache"],
        ),
        0,
    )
    assert registry.get_sample_value("vllm:engine_fully_awake", {"engine": "0"}) == 0


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
    expected = {
        "sleep": {"weights": "unknown", "kv_cache": "unknown"},
        "wake": {"weights": "unknown", "kv_cache": "released"},
        "discard": {"weights": "resident", "kv_cache": "unknown"},
    }
    assert executor.sleep_resource_states == expected[operation]
    assert executor.is_sleeping
    assert executor.sleeping_tags == (
        {"weights", "kv_cache"} if operation == "wake" else set()
    )
    calls = executor.collective_rpc.call_count
    for retry in (
        lambda: executor.sleep(1),
        lambda: executor.wake_up(),
        lambda: executor.wake_up(["weights"]),
        lambda: executor.discard(("kv_cache",)),
    ):
        with pytest.raises(RuntimeError, match="unknown.*rebuild"):
            retry()
    assert executor.collective_rpc.call_count == calls
    logger, registry = make_logger()
    logger.record_sleep_snapshot(
        EngineSleepState(False, **executor.sleep_resource_states), 0
    )
    assert registry.get_sample_value("vllm:engine_fully_awake", {"engine": "0"}) == 0


@pytest.mark.parametrize("tags", [[], ["invalid"], ["weights", "invalid"]])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_empty_or_invalid_wake_leaves_scheduler_and_resources_unchanged(tags, level):
    engine = make_engine()
    engine.pause_scheduler = Mock(
        side_effect=lambda **kwargs: engine.scheduler.set_pause_state(
            PauseState.PAUSED_ALL
        )
    )
    engine.sleep(level)
    before = engine.get_sleep_state()
    engine.model_executor.collective_rpc.reset_mock()
    assert engine.wake_up(tags) is False
    assert engine.get_sleep_state() == before
    engine.model_executor.collective_rpc.assert_not_called()


def make_engine():
    engine = object.__new__(EngineCore)
    engine.model_executor = FakeExecutor()
    engine.scheduler = SimpleNamespace(pause_state=PauseState.PAUSED_ALL)
    engine.scheduler.set_pause_state = lambda state: setattr(
        engine.scheduler, "pause_state", state
    )
    return engine


@pytest.mark.parametrize("level", [0, 1, 2])
def test_full_wake_restores_scheduling_and_resources(level):
    engine = make_engine()
    if level:
        engine.model_executor.sleep(level)
    assert engine.wake_up(None) is True
    assert engine.get_sleep_state() == {
        "scheduler_paused": False,
        "weights": "resident",
        "kv_cache": "resident",
    }


def test_partial_wake_resumes_only_after_remaining_kv_cache():
    engine = make_engine()
    engine.model_executor.sleep(1)
    assert engine.wake_up(["weights"]) is False
    assert engine.is_scheduler_paused()
    assert engine.wake_up(["scheduling"]) is False
    assert engine.is_scheduler_paused()
    assert engine.wake_up(["kv_cache"]) is True
    assert not engine.is_scheduler_paused()


def test_scheduling_tag_resumes_level_zero_pause_without_memory_rpc():
    engine = make_engine()
    assert engine.wake_up(["scheduling"]) is True
    engine.model_executor.collective_rpc.assert_not_called()


def test_wake_result_checks_final_scheduler_state():
    engine = make_engine()
    engine.resume_scheduler = Mock()
    assert engine.wake_up(None) is False
    assert engine.is_scheduler_paused()


def test_kv_only_release_preserves_weights_and_final_wake_resumes():
    engine = make_engine()
    engine.scheduler.has_requests = lambda: False
    engine.batch_queue = None
    engine._reset_caches = Mock()
    engine.release_kv_cache_memory()
    assert engine.get_sleep_state() == {
        "scheduler_paused": True,
        "weights": "resident",
        "kv_cache": "released",
    }
    assert engine.wake_up(["kv_cache"]) is True


@pytest.mark.parametrize("tags", [[], ["invalid"], ["weights", "invalid"]])
def test_executor_noop_wake_does_not_dispatch_or_change_resources(tags):
    executor = FakeExecutor()
    executor.sleep(1)
    before = executor.sleep_resource_states.copy()
    executor.collective_rpc.reset_mock()
    executor.wake_up(tags)
    assert executor.sleep_resource_states == before
    assert executor.sleeping_tags == {"weights", "kv_cache"}
    executor.collective_rpc.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results, expected", [([True, False], False), ([True, True], True)]
)
async def test_dp_wake_requires_every_engine_to_be_awake(results, expected):
    client = SimpleNamespace(call_utility_all_async=AsyncMock(return_value=results))
    assert await AsyncMPClient.wake_up_async(client, None) is expected


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


def test_sync_snapshot_uses_local_engine_index_with_nonzero_dp_index():
    logger, registry = make_logger()
    engine = object.__new__(LLMEngine)
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=3)
    )
    engine.engine_core = SimpleNamespace(
        get_sleep_state=lambda: {
            "scheduler_paused": False,
            "weights": "resident",
            "kv_cache": "resident",
        }
    )
    engine.logger_manager = SimpleNamespace(
        record_sleep_snapshot=logger.record_sleep_snapshot
    )
    engine._record_sleep_snapshot()
    assert registry.get_sample_value("vllm:engine_fully_awake", {"engine": "0"}) == 1


def test_sync_logger_records_initial_snapshot():
    config = SimpleNamespace(
        logging_config=None,
        model_config=Mock(),
        observability_config=SimpleNamespace(otlp_traces_endpoint=None),
        parallel_config=SimpleNamespace(
            data_parallel_size=1, distributed_executor_backend="mp"
        ),
        scheduler_config=SimpleNamespace(stream_interval=1),
    )
    core_client = Mock()
    core_client.get_sleep_state.return_value = {
        "scheduler_paused": False,
        "weights": "resident",
        "kv_cache": "resident",
    }
    manager = Mock()
    with (
        patch("vllm.v1.engine.llm_engine.configure_logging_if_needed"),
        patch("vllm.v1.engine.llm_engine.renderer_from_config"),
        patch("vllm.v1.engine.llm_engine.InputProcessor"),
        patch("vllm.v1.engine.llm_engine.OutputProcessor"),
        patch(
            "vllm.v1.engine.llm_engine.EngineCoreClient.make_client",
            return_value=core_client,
        ),
        patch("vllm.v1.engine.llm_engine.StatLoggerManager", return_value=manager),
        patch.object(LLMEngine, "reset_mm_cache"),
    ):
        LLMEngine(config, Mock(), log_stats=True, multiprocess_mode=True)
    manager.record_sleep_snapshot.assert_called_once_with(EngineSleepState(), 0)


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


def test_identical_sleep_snapshots_are_not_recorded_twice():
    manager = object.__new__(StatLoggerManager)
    logger = Mock()
    manager.stat_loggers = [logger]
    manager.record_sleep_snapshot(EngineSleepState(), 0)
    manager.record_sleep_snapshot(EngineSleepState(), 0)
    manager.record_sleep_snapshot(EngineSleepState(), 1)
    assert logger.record_sleep_snapshot.call_count == 2


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
