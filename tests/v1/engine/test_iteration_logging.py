# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from contextlib import contextmanager
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


class FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class FakeStageScheduler:
    def __init__(self, scheduler_output, has_requests=True):
        self.scheduler_output = scheduler_output
        self._has_requests = has_requests
        self.updated_with = None

    def has_requests(self):
        return self._has_requests

    def schedule(self, throttle_prefills):
        return self.scheduler_output

    def get_grammar_bitmask(self, scheduler_output):
        return "grammar"

    def update_from_output(self, scheduler_output, model_output):
        self.updated_with = (scheduler_output, model_output)
        return {}


class FakeStageModelExecutor:
    def __init__(self, execute_result=None, sample_result=None):
        self.execute_future = FakeFuture(execute_result)
        self.sample_future = FakeFuture(sample_result)
        self.sample_result = sample_result
        self.sample_calls = []

    def execute_model(self, scheduler_output, non_block=False):
        return self.execute_future

    def sample_tokens(self, grammar_output, non_block=False):
        self.sample_calls.append((grammar_output, non_block))
        return self.sample_future if non_block else self.sample_result


class FakeStageEngine:
    def __init__(self, scheduler, model_executor=None):
        self.scheduler = scheduler
        self.model_executor = model_executor
        self.stages = []
        self.batch_queue = None
        self.batch_queue_size = 2
        self.is_ec_consumer = True
        self.is_pooling_model = False
        self.check_for_draft_tokens = False

    @contextmanager
    def capture_iteration_details(self, scheduler_output):
        yield None

    @contextmanager
    def log_error_detail(self, scheduler_output):
        yield

    @contextmanager
    def dump_on_slow_execution(self, scheduler_output, stage):
        self.stages.append(stage)
        yield

    def _should_throttle_prefills(self):
        return False

    def _process_aborts_queue(self):
        return

    def _attach_iteration_details(self, outputs, iteration_details):
        return


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
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
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
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    ):
        timers[0].function()

    assert len(dumps) == 1
    assert dumps[0][1] is scheduler_output
    assert dumps[0][2] is scheduler_stats
    assert dumps[0][3] == 2.0
    assert dumps[0][4] == engine_core_module.EXECUTE_MODEL_WAIT_STAGE


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
        engine, scheduler_output, engine_core_module.EXECUTE_MODEL_WAIT_STAGE
    ):
        calls.append("body")

    assert calls == [
        {
            "config": engine.vllm_config,
            "scheduler_output": scheduler_output,
            "scheduler_stats": scheduler_stats,
            "timeout_s": 7,
            "stage": engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        },
        "enter",
        "body",
        "exit",
    ]


def test_step_records_execute_and_sync_sample_timeout_stages():
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        pending_structured_output_tokens=False,
    )
    scheduler = FakeStageScheduler(scheduler_output)
    model_output = SimpleNamespace()
    model_executor = FakeStageModelExecutor(
        execute_result=None, sample_result=model_output
    )
    engine = FakeStageEngine(scheduler, model_executor)

    outputs, model_executed = EngineCore.step(engine)

    assert outputs == {}
    assert model_executed
    assert scheduler.updated_with == (scheduler_output, model_output)
    assert model_executor.sample_calls == [("grammar", False)]
    assert engine.stages == [
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        engine_core_module.SAMPLE_TOKENS_STAGE,
    ]


def test_step_with_batch_queue_enqueues_sample_tokens_wait_stage():
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        pending_structured_output_tokens=False,
    )
    scheduler = FakeStageScheduler(scheduler_output)
    model_executor = FakeStageModelExecutor(
        execute_result=SimpleNamespace(), sample_result=SimpleNamespace()
    )
    engine = FakeStageEngine(scheduler, model_executor)
    engine.batch_queue = deque(maxlen=engine.batch_queue_size)

    outputs, model_executed = EngineCore.step_with_batch_queue(engine)

    assert outputs is None
    assert model_executed
    assert len(engine.batch_queue) == 1
    future, queued_scheduler_output, exec_future, future_stage = engine.batch_queue[0]
    assert future is model_executor.sample_future
    assert queued_scheduler_output is scheduler_output
    assert exec_future is model_executor.execute_future
    assert future_stage == engine_core_module.SAMPLE_TOKENS_WAIT_STAGE
    assert model_executor.sample_calls == [("grammar", True)]


def test_step_with_batch_queue_uses_queued_future_stage():
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    scheduler = FakeStageScheduler(scheduler_output, has_requests=False)
    model_output = SimpleNamespace()
    exec_future = FakeFuture(SimpleNamespace())
    engine = FakeStageEngine(scheduler)
    engine.batch_queue = deque(
        [
            (
                FakeFuture(model_output),
                scheduler_output,
                exec_future,
                engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
            )
        ],
        maxlen=engine.batch_queue_size,
    )

    outputs, model_executed = EngineCore.step_with_batch_queue(engine)

    assert outputs == {}
    assert not model_executed
    assert scheduler.updated_with == (scheduler_output, model_output)
    assert engine.stages == [engine_core_module.SAMPLE_TOKENS_WAIT_STAGE]
