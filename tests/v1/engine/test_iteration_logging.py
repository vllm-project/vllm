# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace

import vllm.v1.engine.core as engine_core_module
from vllm.logging_utils import dump_input
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats


class FakeEngineCore:
    def _make_iteration_details_stats(
        self, iteration_details: SchedulerIterationDetails
    ) -> SchedulerStats:
        return SchedulerStats(iteration_details=iteration_details)


def make_iteration_details() -> SchedulerIterationDetails:
    return SchedulerIterationDetails(
        iteration_index=1,
        num_ctx_requests=2,
        num_ctx_tokens=3,
        num_generation_requests=4,
        num_generation_tokens=5,
        elapsed_ms=6.7,
    )


def make_fake_engine(log_stats: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        log_stats=log_stats,
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(
                enable_logging_iteration_details=True,
            )
        ),
    )


class FakeTimer:
    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


def test_capture_iteration_details_disabled_without_log_stats():
    engine = make_fake_engine(log_stats=False)

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is None

    assert not hasattr(engine, "_iteration_index")


def test_capture_iteration_details_fills_elapsed_time():
    engine = make_fake_engine()

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is not None
        assert iteration_details.elapsed_ms == 0.0
        assert iteration_details.is_dummy
        time.sleep(0.001)

    assert iteration_details is not None
    assert iteration_details.elapsed_ms > 0.0
    assert engine._iteration_index == 1


def test_attach_iteration_details_uses_existing_output():
    iteration_details = make_iteration_details()
    outputs = {
        2: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
        1: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
    }

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert 0 not in outputs
    assert outputs[2].scheduler_stats is not None
    assert outputs[2].scheduler_stats.iteration_details == iteration_details
    assert outputs[1].scheduler_stats is not None
    assert outputs[1].scheduler_stats.iteration_details is None


def test_attach_iteration_details_falls_back_to_client_zero_without_outputs():
    iteration_details = make_iteration_details()
    outputs: dict[int, EngineCoreOutputs] = {}

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert set(outputs) == {0}
    assert outputs[0].scheduler_stats is not None
    assert outputs[0].scheduler_stats.iteration_details == iteration_details


def test_engine_execution_timeout_dumper_cancels_timer(monkeypatch):
    timers = []
    dumps = []

    def timer_factory(interval, function):
        timer = FakeTimer(interval, function)
        timers.append(timer)
        return timer

    monkeypatch.setattr(dump_input.threading, "Timer", timer_factory)
    monkeypatch.setattr(
        dump_input,
        "dump_engine_execution_timeout",
        lambda *args: dumps.append(args),
    )

    with dump_input.EngineExecutionTimeoutDumper(
        config=SimpleNamespace(),
        scheduler_output=SimpleNamespace(),
        scheduler_stats=SchedulerStats(),
        timeout_s=3.0,
        stage="model_execution",
    ):
        pass

    assert len(timers) == 1
    assert timers[0].interval == 3.0
    assert timers[0].daemon
    assert timers[0].started
    assert timers[0].cancelled
    assert not dumps


def test_engine_execution_timeout_dumper_dumps_when_timer_fires(monkeypatch):
    timers = []
    dumps = []

    def timer_factory(interval, function):
        timer = FakeTimer(interval, function)
        timers.append(timer)
        return timer

    monkeypatch.setattr(dump_input.threading, "Timer", timer_factory)
    monkeypatch.setattr(
        dump_input,
        "dump_engine_execution_timeout",
        lambda *args: dumps.append(args),
    )

    scheduler_output = SimpleNamespace()
    scheduler_stats = SchedulerStats(num_running_reqs=1)
    with dump_input.EngineExecutionTimeoutDumper(
        config=SimpleNamespace(),
        scheduler_output=scheduler_output,
        scheduler_stats=scheduler_stats,
        timeout_s=2.0,
        stage="model_execution",
    ):
        timers[0].function()

    assert len(dumps) == 1
    assert dumps[0][1] is scheduler_output
    assert dumps[0][2] is scheduler_stats
    assert dumps[0][3] == 2.0
    assert dumps[0][4] == "model_execution"


def test_dump_on_slow_execution_uses_env_timeout_and_scheduler_stats(monkeypatch):
    calls = []
    scheduler_stats = SchedulerStats(num_waiting_reqs=2)

    class FakeDumper:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append("exit")

    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        scheduler=SimpleNamespace(make_stats=lambda: scheduler_stats),
    )
    scheduler_output = SimpleNamespace()

    monkeypatch.setattr(engine_core_module, "EngineExecutionTimeoutDumper", FakeDumper)
    monkeypatch.setattr(engine_core_module.envs, "VLLM_ENGINE_ITERATION_TIMEOUT_S", 7)

    with EngineCore.dump_on_slow_execution(
        engine, scheduler_output, "model_execution"
    ):
        calls.append("body")

    assert calls == [
        {
            "config": engine.vllm_config,
            "scheduler_output": scheduler_output,
            "scheduler_stats": scheduler_stats,
            "timeout_s": 7,
            "stage": "model_execution",
        },
        "enter",
        "body",
        "exit",
    ]
