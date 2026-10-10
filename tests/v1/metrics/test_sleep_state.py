# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from contextlib import suppress
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec
import pytest
from prometheus_client import CollectorRegistry, Gauge, generate_latest

from vllm.renderers.base import BaseRenderer
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import (
    PrometheusStatLogger,
    StatLoggerBase,
    StatLoggerManager,
)
from vllm.v1.metrics.stats import EngineSleepState, MultiModalCacheStats, SchedulerStats

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


def test_engine_partial_wake_sleep_rejection_preserves_resources_and_pause():
    engine = make_engine()
    engine.pause_scheduler = Mock(
        side_effect=lambda **kwargs: engine.scheduler.set_pause_state(
            PauseState.PAUSED_ALL
        )
    )
    engine.model_executor.sleep(1)
    engine.wake_up(["weights"])
    before = engine.get_sleep_state()
    with pytest.raises(RuntimeError, match="partially awake"):
        engine.sleep(1)
    assert engine.get_sleep_state() == before
    assert engine.model_executor.collective_rpc.call_count == 2
    assert engine.wake_up(["kv_cache"]) is True
    engine.sleep(1)
    assert engine.is_scheduler_paused()


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


def test_engine_snapshots_do_not_dispatch_legacy_logger_callback():
    legacy = SimpleNamespace(record_sleep_state=Mock())
    legacy.record_sleep_snapshot = (
        lambda state, idx: StatLoggerBase.record_sleep_snapshot(legacy, state, idx)
    )
    modern = Mock()
    manager = object.__new__(StatLoggerManager)
    manager.stat_loggers = [legacy, modern]
    state = EngineSleepState(True, "offloaded", "released")
    manager.record_sleep_snapshot(state, 0)
    legacy.record_sleep_state.assert_not_called()
    modern.record_sleep_state.assert_not_called()
    modern.record_sleep_snapshot.assert_called_once_with(state, 0)


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


def make_buffered_frontend(use_async):
    logger, registry = make_logger()
    normal = Mock()
    manager = object.__new__(StatLoggerManager)
    manager.stat_loggers = [
        SimpleNamespace(
            record=normal.record, record_sleep_snapshot=logger.record_sleep_snapshot
        )
    ]
    recorded = asyncio.Event()

    def record(**kwargs):
        StatLoggerManager.record(manager, **kwargs)
        recorded.set()

    manager.record = Mock(side_effect=record)
    manager.log_engine_initialized = Mock()
    stats = MultiModalCacheStats()
    stats.record(num_queries=5, num_hits=3)
    renderer = SimpleNamespace(
        _mm_cache_stats=stats,
        mm_processor_cache=None,
        clear_mm_cache_async=AsyncMock(),
        shutdown=Mock(),
        tokenizer=None,
    )
    renderer.stat_mm_cache = Mock(wraps=lambda: BaseRenderer.stat_mm_cache(renderer))
    queue: asyncio.Queue[EngineCoreOutputs] = asyncio.Queue()
    core = Mock(engine_ranks_managed=[0], get_output_async=queue.get)
    processor = object.__new__(OutputProcessor)
    processor.lora_states = Mock()
    processor.process_outputs = Mock(
        return_value=SimpleNamespace(request_outputs=[], reqs_to_abort=[])
    )
    processor.propagate_error = Mock()
    if use_async:
        config = Mock()
        config.observability_config.otlp_traces_endpoint = None
        config.profiler_config.should_profile_frontend = False
        with (
            patch("vllm.v1.engine.async_llm.configure_logging_if_needed"),
            patch("vllm.v1.engine.async_llm.maybe_register_config_serialize_by_value"),
            patch(
                "vllm.v1.engine.async_llm.load_stat_logger_plugin_factories",
                return_value=[],
            ),
            patch(
                "vllm.v1.engine.async_llm.renderer_from_config", return_value=renderer
            ),
            patch("vllm.v1.engine.async_llm.InputProcessor"),
            patch("vllm.v1.engine.async_llm.OutputProcessor", return_value=processor),
            patch(
                "vllm.v1.engine.async_llm.EngineCoreClient.make_async_mp_client",
                return_value=core,
            ),
            patch("vllm.v1.engine.async_llm.StatLoggerManager", return_value=manager),
        ):
            engine = AsyncLLM(config, Mock(), log_stats=True)
        assert engine.output_handler is None
        engine.shutdown = Mock()
    else:
        engine = object.__new__(LLMEngine)
        engine.renderer = renderer
        engine.engine_core = core
        engine.output_processor = processor
        engine.logger_manager = manager
        engine.log_stats = True
        engine.should_execute_dummy_batch = False
        engine.do_log_stats_with_interval = Mock()
    return SimpleNamespace(
        engine=engine,
        renderer=renderer,
        stats=stats,
        normal=normal,
        registry=registry,
        queue=queue,
        recorded=recorded,
        use_async=use_async,
    )


@pytest.fixture
def buffered_frontend(request):
    return make_buffered_frontend(request.param)


@pytest.mark.asyncio
@pytest.mark.parametrize("buffered_frontend", [False, True], indirect=True)
async def test_state_only_preserves_mm_stats_for_next_normal_output(buffered_frontend):
    f = buffered_frontend
    snapshot = EngineCoreOutputs(
        scheduler_stats=SchedulerStats(
            sleep_state=EngineSleepState(True), sleep_state_only=True
        )
    )
    normal = EngineCoreOutputs(
        outputs=[EngineCoreOutput(request_id="request", new_token_ids=[1])],
        scheduler_stats=SchedulerStats(num_running_reqs=7, kv_cache_usage=0.25),
    )
    if f.use_async:
        f.engine._run_output_handler()
    else:
        f.engine.engine_core.get_output.side_effect = [snapshot, normal]
    try:
        for output in (snapshot, normal):
            f.recorded.clear()
            if f.use_async:
                f.queue.put_nowait(output)
                await asyncio.wait_for(f.recorded.wait(), timeout=5)
            else:
                f.engine.step()
            if output is snapshot:
                f.renderer.stat_mm_cache.assert_not_called()
                assert f.renderer._mm_cache_stats is f.stats
                assert (f.stats.requests, f.stats.queries, f.stats.hits) == (1, 5, 3)
                f.normal.record.assert_not_called()
                f.engine.output_processor.lora_states.update_scheduler_stats.assert_not_called()
                assert (
                    f.registry.get_sample_value(
                        "vllm:engine_fully_awake", {"engine": "0"}
                    )
                    == 0
                )
        f.renderer.stat_mm_cache.assert_called_once_with()
        f.normal.record.assert_called_once()
        assert f.normal.record.call_args.kwargs["mm_cache_stats"] is f.stats
        assert f.normal.record.call_args.args[0] is normal.scheduler_stats
        assert f.normal.record.call_args.args[1] is not None
        assert f.renderer._mm_cache_stats.queries == 0
        f.engine.output_processor.lora_states.update_scheduler_stats.assert_called_once_with(
            normal.scheduler_stats
        )
    finally:
        if f.use_async:
            f.engine.output_handler.cancel()
            with suppress(asyncio.CancelledError):
                await f.engine.output_handler


@pytest.fixture
def idle_async_frontend():
    with pytest.raises(RuntimeError, match="no running event loop"):
        asyncio.get_running_loop()
    return make_buffered_frontend(True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        "pause_generation",
        "resume_generation",
        "sleep",
        "wake_up",
        "release_kv_cache_memory",
    ],
)
async def test_first_control_operation_starts_idle_output_handler(
    idle_async_frontend, operation
):
    f = idle_async_frontend
    engine = f.engine
    assert engine.output_handler is None
    methods = {
        "pause_generation": "pause_scheduler_async",
        "resume_generation": "resume_scheduler_async",
        "sleep": "sleep_async",
        "wake_up": "wake_up_async",
        "release_kv_cache_memory": "release_kv_cache_memory_async",
    }

    async def control(name):
        state = EngineSleepState(
            name not in ("resume_generation", "wake_up"),
            "offloaded" if name == "sleep" else "resident",
            "released" if name in ("sleep", "release_kv_cache_memory") else "resident",
        )

        def publish(*args, **kwargs):
            f.queue.put_nowait(
                EngineCoreOutputs(
                    scheduler_stats=SchedulerStats(
                        sleep_state=state, sleep_state_only=True
                    )
                )
            )
            return state.fully_awake

        setattr(engine.engine_core, methods[name], AsyncMock(side_effect=publish))
        f.recorded.clear()
        await getattr(engine, name)()
        await asyncio.wait_for(f.recorded.wait(), timeout=5)
        assert f.registry.get_sample_value(
            "vllm:engine_fully_awake", {"engine": "0"}
        ) == int(state.fully_awake)

    try:
        await control(operation)
        handler = engine.output_handler
        assert handler is not None
        for name in methods:
            await control(name)
            assert engine.output_handler is handler
        f.normal.record.assert_not_called()
        f.renderer.stat_mm_cache.assert_not_called()
        engine.output_processor.propagate_error.assert_not_called()
    finally:
        if engine.output_handler is not None:
            engine.output_handler.cancel()
            with suppress(asyncio.CancelledError):
                await engine.output_handler
