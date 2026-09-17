# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
import time
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import vllm.v1.engine.core as engine_core_module
from vllm.logging_utils import dump_input
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.scheduler import Scheduler
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


class FakeFuture:
    def __init__(self, result, events=None, label=None):
        self._result = result
        self._events = events
        self._label = label

    def result(self):
        if self._events is not None and self._label is not None:
            self._events.append(("call", self._label))
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
        self.events = []
        self.execute_future = FakeFuture(
            execute_result, self.events, "execute_model.result"
        )
        self.sample_future = FakeFuture(
            sample_result, self.events, "sample_tokens.result"
        )
        self.sample_result = sample_result
        self.sample_calls = []

    def execute_model(self, scheduler_output, non_block=False):
        self.events.append(("call", "execute_model"))
        return self.execute_future

    def sample_tokens(self, grammar_output, non_block=False):
        self.events.append(("call", "sample_tokens"))
        self.sample_calls.append((grammar_output, non_block))
        return self.sample_future if non_block else self.sample_result


class FakeStageEngine:
    def __init__(self, scheduler, model_executor=None):
        self.scheduler = scheduler
        self.model_executor = model_executor
        self.stages = []
        self.stage_states = []
        self.prepared_timeout_outputs = []
        self.events = model_executor.events if model_executor is not None else []
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
    def dump_on_slow_execution(self, scheduler_output, stage, scheduler_state):
        self.stages.append(stage)
        self.stage_states.append(scheduler_state)
        self.events.append(("enter", stage))
        try:
            yield
        finally:
            self.events.append(("exit", stage))

    def _prepare_timeout_diagnostic_state(self, scheduler_output):
        self.prepared_timeout_outputs.append(scheduler_output)
        return {"scheduler_output": scheduler_output}

    def _should_throttle_prefills(self):
        return False

    def _process_aborts_queue(self):
        return

    def _attach_iteration_details(self, outputs, iteration_details):
        return


def clear_engine_execution_timeout_dump_throttle():
    with dump_input._engine_execution_timeout_dump_lock:
        dump_input._engine_execution_timeout_dump_last_s.clear()


def make_timeout_scheduler_output() -> SimpleNamespace:
    sampling_params = SimpleNamespace(
        extra_args={"private": "private-extra-arg"},
        max_tokens=8,
        stop="private-stop-string",
        stop_token_ids=[1, 2],
        structured_outputs=SimpleNamespace(json="private-schema"),
        temperature=0.25,
        top_k=10,
        top_p=0.9,
    )
    return SimpleNamespace(
        finished_req_ids=set(),
        kv_connector_metadata=None,
        num_scheduled_tokens={"request-123": 4, "request-456": 2},
        pending_structured_output_tokens=False,
        preempted_req_ids=None,
        scheduled_cached_reqs=SimpleNamespace(
            all_token_ids={"request-456": [10, 11, 12, 13]},
            new_block_ids=[([1, 2],)],
            new_token_ids=[[10, 11]],
            num_computed_tokens=[3],
            num_output_tokens=[1],
            num_reqs=1,
            req_ids=["request-456"],
            resumed_req_ids=set(),
        ),
        scheduled_encoder_inputs={},
        scheduled_new_reqs=[
            SimpleNamespace(
                block_ids=([1, 2, 3],),
                lora_request=None,
                mm_features=[],
                num_computed_tokens=0,
                pooling_params=None,
                prefill_token_ids=None,
                prompt_embeds=None,
                prompt_token_ids=[101, 102, 103],
                req_id="request-123",
                sampling_params=sampling_params,
            )
        ],
        scheduled_spec_decode_tokens={},
        total_num_scheduled_tokens=6,
    )


def make_timeout_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            cache_dtype="auto",
            gpu_memory_utilization=0.9,
        ),
        model_config=SimpleNamespace(
            dtype="float16",
            enforce_eager=False,
            hf_config=SimpleNamespace(
                architectures=["TestArchitecture"],
                model_type="test-model-type",
            ),
            max_model_len=4096,
            model="private/model/path",
            runner_type="generate",
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=2048,
            max_num_seqs=128,
            policy="fcfs",
        ),
        speculative_config=None,
    )


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


def test_engine_execution_timeout_watchdog_reuses_thread():
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=10.0,
    )
    watchdog.start()
    thread = watchdog._thread

    try:
        first_generation = watchdog.arm(
            make_timeout_scheduler_output(),
            {"snapshot": 1},
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        watchdog.disarm(first_generation)
        second_generation = watchdog.arm(
            make_timeout_scheduler_output(),
            {"snapshot": 2},
            engine_core_module.SAMPLE_TOKENS_STAGE,
        )
        watchdog.disarm(second_generation)

        assert thread is not None
        assert watchdog._state is None
        assert watchdog._thread is thread
        assert thread.is_alive()
    finally:
        watchdog.stop()

    assert not thread.is_alive()


def test_engine_execution_timeout_watchdog_disabled_is_lazy():
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=0,
    )

    watchdog.start()
    generation = watchdog.arm(
        make_timeout_scheduler_output(),
        {},
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )
    watchdog.disarm(generation)
    watchdog.stop()

    assert generation is None
    assert watchdog._thread is None
    assert not watchdog.enabled


def test_engine_execution_timeout_watchdog_is_opt_in(monkeypatch):
    configured_timeout_s = engine_core_module.envs.VLLM_ENGINE_ITERATION_TIMEOUT_S
    env_name = engine_core_module._ENGINE_ITERATION_TIMEOUT_ENV

    monkeypatch.setattr(
        engine_core_module.envs,
        "is_set",
        lambda name: False,
    )
    assert engine_core_module._get_engine_execution_timeout_s() == 0

    monkeypatch.setattr(
        engine_core_module.envs,
        "is_set",
        lambda name: name == env_name,
    )
    assert engine_core_module._get_engine_execution_timeout_s() == configured_timeout_s


def test_engine_execution_timeout_watchdog_fails_open_on_thread_start_error(
    monkeypatch,
):
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=10.0,
    )

    def fail_start(_thread):
        raise RuntimeError("thread limit reached")

    snapshot_calls = []

    def record_snapshot(*args):
        snapshot_calls.append(args)
        return {}

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    monkeypatch.setattr(
        dump_input,
        "make_engine_execution_timeout_snapshot",
        record_snapshot,
    )

    watchdog.start()
    generation = watchdog.arm(
        make_timeout_scheduler_output(),
        {},
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )

    assert generation is None
    assert watchdog._thread is None
    assert not watchdog.enabled
    assert snapshot_calls == []


def test_engine_execution_timeout_watchdog_ignores_stale_disarm(monkeypatch):
    now_s = [0.0]
    dumps = []
    dump_completed = threading.Event()

    def record_dump(config, scheduler_output, scheduler_state, timeout_s, stage):
        dumps.append((scheduler_output, scheduler_state, stage))
        dump_completed.set()

    monkeypatch.setattr(dump_input, "dump_engine_execution_timeout", record_dump)
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=10.0,
        time_fn=lambda: now_s[0],
    )
    watchdog.start()

    try:
        stale_output = make_timeout_scheduler_output()
        stale_output.scheduled_new_reqs[0].req_id = "stale-generation-request"
        current_output = make_timeout_scheduler_output()
        current_output.scheduled_new_reqs[0].req_id = "current-generation-request"
        stale_generation = watchdog.arm(
            stale_output,
            {"snapshot": 1},
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        watchdog.arm(
            current_output,
            {"snapshot": 2},
            engine_core_module.SAMPLE_TOKENS_STAGE,
        )
        watchdog.disarm(stale_generation)
        now_s[0] = 11.0
        watchdog._wake_event.set()

        assert dump_completed.wait(timeout=1.0)
        assert len(dumps) == 1
        output_summary, queue_summary, stage = dumps[0]
        assert output_summary["request_samples"][0]["request_id"] == (
            "current-generation-request"
        )
        assert queue_summary == {"snapshot": 2}
        assert stage == engine_core_module.SAMPLE_TOKENS_STAGE
    finally:
        watchdog.stop()


def test_engine_execution_timeout_watchdog_keeps_arm_and_stop_nonblocking(
    monkeypatch,
):
    dump_started = threading.Event()
    release_dump = threading.Event()

    def blocking_dump(*args):
        dump_started.set()
        assert release_dump.wait(timeout=2.0)

    monkeypatch.setattr(dump_input, "dump_engine_execution_timeout", blocking_dump)
    monkeypatch.setattr(
        dump_input,
        "ENGINE_EXECUTION_TIMEOUT_WATCHDOG_STOP_TIMEOUT_S",
        0.01,
    )
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=0.01,
    )
    watchdog.start()
    arm_finished = threading.Event()
    stop_finished = threading.Event()
    generations = []

    try:
        watchdog.arm(
            make_timeout_scheduler_output(),
            {"snapshot": 1},
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        assert dump_started.wait(timeout=1.0)

        def arm_again():
            generations.append(
                watchdog.arm(
                    make_timeout_scheduler_output(),
                    {"snapshot": 2},
                    engine_core_module.SAMPLE_TOKENS_STAGE,
                )
            )
            arm_finished.set()

        arm_thread = threading.Thread(target=arm_again)
        arm_thread.start()
        assert arm_finished.wait(timeout=1.0)
        arm_thread.join(timeout=1.0)
        assert generations[0] is not None

        def stop_watchdog():
            watchdog.stop()
            stop_finished.set()

        stop_thread = threading.Thread(target=stop_watchdog)
        stop_thread.start()
        assert stop_finished.wait(timeout=1.0)
        stop_thread.join(timeout=1.0)
        assert watchdog._thread is not None
        assert watchdog._thread.is_alive()
    finally:
        release_dump.set()
        watchdog.stop()
        if watchdog._thread is not None:
            watchdog._thread.join(timeout=1.0)

    assert watchdog._thread is not None
    assert not watchdog._thread.is_alive()


def test_engine_execution_timeout_watchdog_fires_real_timeout(monkeypatch):
    contexts: list[tuple[Any, ...]] = []
    traceback_dumped = threading.Event()

    def record_context(*args: Any) -> None:
        contexts.append(args)

    def record_traceback(*args: Any, **kwargs: Any) -> None:
        traceback_dumped.set()

    clear_engine_execution_timeout_dump_throttle()
    monkeypatch.setattr(dump_input, "_dump_engine_timeout_context", record_context)
    monkeypatch.setattr(dump_input.faulthandler, "dump_traceback", record_traceback)
    config = make_timeout_config()
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=config,
        timeout_s=0.01,
    )
    scheduler_output = make_timeout_scheduler_output()
    scheduler_state = {"num_running_reqs": 1, "num_waiting_reqs": 0}
    watchdog.start()

    try:
        generation = watchdog.arm(
            scheduler_output,
            scheduler_state,
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        assert traceback_dumped.wait(timeout=1.0)
        watchdog.disarm(generation)
    finally:
        watchdog.stop()
        clear_engine_execution_timeout_dump_throttle()

    assert len(contexts) == 1
    context_config, output_summary, queue_summary = contexts[0]
    assert context_config is config
    assert output_summary["request_samples"][0]["request_id"] == "request-123"
    assert queue_summary == scheduler_state
    assert output_summary is not scheduler_output


def test_engine_execution_timeout_watchdog_disarm_suppresses_dump(monkeypatch):
    dumped = threading.Event()
    monkeypatch.setattr(
        dump_input,
        "dump_engine_execution_timeout",
        lambda *args: dumped.set(),
    )
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=0.01,
    )

    generation = watchdog.arm(
        make_timeout_scheduler_output(),
        {},
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )
    watchdog.disarm(generation)
    watchdog.start()
    try:
        assert not dumped.wait(timeout=0.05)
    finally:
        watchdog.stop()


def test_engine_execution_timeout_real_timeout_emits_useful_summary(monkeypatch):
    logs = []
    traceback_dumped = threading.Event()

    def record_log(message, *args):
        logs.append(message % args)

    clear_engine_execution_timeout_dump_throttle()
    monkeypatch.setattr(dump_input.logger, "error", record_log)
    monkeypatch.setattr(
        dump_input.faulthandler,
        "dump_traceback",
        lambda *args, **kwargs: traceback_dumped.set(),
    )
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=0.01,
    )
    watchdog.start()

    try:
        watchdog.arm(
            make_timeout_scheduler_output(),
            {"num_running_reqs": 1, "num_waiting_reqs": 0},
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        assert traceback_dumped.wait(timeout=1.0)
    finally:
        watchdog.stop()
        clear_engine_execution_timeout_dump_throttle()

    combined_logs = "\n".join(logs)
    assert engine_core_module.EXECUTE_MODEL_WAIT_STAGE in combined_logs
    assert "request-123" in combined_logs
    assert "request-456" in combined_logs
    assert "temperature" in combined_logs
    assert "num_running_reqs" in combined_logs
    assert "private-stop-string" not in combined_logs
    assert "private/model/path" not in combined_logs


def test_engine_execution_timeout_watchdog_fires_twice_on_same_thread(monkeypatch):
    dumped_stages = []
    dump_completed = threading.Event()

    def record_dump(config, scheduler_output, scheduler_state, timeout_s, stage):
        dumped_stages.append(stage)
        dump_completed.set()

    monkeypatch.setattr(dump_input, "dump_engine_execution_timeout", record_dump)
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=0.01,
    )
    watchdog.start()
    thread = watchdog._thread

    try:
        for stage in (
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
            engine_core_module.SAMPLE_TOKENS_STAGE,
        ):
            dump_completed.clear()
            watchdog.arm(make_timeout_scheduler_output(), {}, stage)
            assert dump_completed.wait(timeout=1.0)
        assert watchdog._thread is thread
        assert dumped_stages == [
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
            engine_core_module.SAMPLE_TOKENS_STAGE,
        ]
    finally:
        watchdog.stop()


def test_engine_execution_timeout_dump_is_throttled_by_stage(monkeypatch):
    contexts: list[tuple[Any, ...]] = []
    tracebacks: list[dict[str, Any]] = []
    times = iter([100.0, 101.0, 102.0, 401.0])

    def record_context(*args: Any) -> None:
        contexts.append(args)

    def record_traceback(*args: Any, **kwargs: Any) -> None:
        tracebacks.append(kwargs)

    clear_engine_execution_timeout_dump_throttle()
    monkeypatch.setattr(dump_input.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(dump_input, "_dump_engine_timeout_context", record_context)
    monkeypatch.setattr(dump_input.faulthandler, "dump_traceback", record_traceback)
    scheduler_output = make_timeout_scheduler_output()
    config = make_timeout_config()

    scheduler_state = {"num_running_reqs": 1, "num_waiting_reqs": 0}

    dump_input.dump_engine_execution_timeout(
        config=config,
        scheduler_output=scheduler_output,
        scheduler_state=scheduler_state,
        timeout_s=1.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )
    dump_input.dump_engine_execution_timeout(
        config=config,
        scheduler_output=scheduler_output,
        scheduler_state=scheduler_state,
        timeout_s=1.0,
        stage=engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
    )
    dump_input.dump_engine_execution_timeout(
        config=config,
        scheduler_output=scheduler_output,
        scheduler_state=scheduler_state,
        timeout_s=1.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )
    dump_input.dump_engine_execution_timeout(
        config=config,
        scheduler_output=scheduler_output,
        scheduler_state=scheduler_state,
        timeout_s=1.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )

    assert len(contexts) == 3
    assert len(tracebacks) == 3
    assert all(context[0] is config for context in contexts)
    assert all(context[1] is scheduler_output for context in contexts)

    clear_engine_execution_timeout_dump_throttle()


def test_engine_execution_timeout_context_balances_detail_and_privacy(monkeypatch):
    logs = []

    def record_log(message, *args):
        logs.append(message % args)

    monkeypatch.setattr(dump_input.logger, "error", record_log)
    dump_input._dump_engine_timeout_context(
        make_timeout_config(),
        make_timeout_scheduler_output(),
        {"num_running_reqs": 1, "num_waiting_reqs": 0},
    )

    combined_logs = "\n".join(logs)
    assert "request-123" in combined_logs
    assert "request-456" in combined_logs
    assert "temperature" in combined_logs
    assert "0.25" in combined_logs
    assert "TestArchitecture" in combined_logs
    assert "private/model/path" not in combined_logs
    assert "private-extra-arg" not in combined_logs
    assert "private-schema" not in combined_logs
    assert "private-stop-string" not in combined_logs
    assert "[101, 102, 103]" not in combined_logs
    assert "num_scheduled_new_reqs" in combined_logs
    assert "num_running_reqs" in combined_logs


def test_engine_execution_timeout_samples_include_cached_requests_when_truncated():
    scheduler_output = make_timeout_scheduler_output()
    template_request = scheduler_output.scheduled_new_reqs[0]
    scheduler_output.scheduled_new_reqs = []
    for index in range(dump_input.ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT):
        request_fields = vars(template_request).copy()
        request_fields["req_id"] = f"new-request-{index}"
        scheduler_output.scheduled_new_reqs.append(SimpleNamespace(**request_fields))

    samples = dump_input._make_request_samples(scheduler_output)

    assert len(samples) == dump_input.ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT
    assert samples[-1]["request_kind"] == "cached"
    assert samples[-1]["request_id"] == "request-456"


def test_engine_execution_timeout_bounds_oversized_request_ids(monkeypatch):
    logs = []
    scheduler_output = make_timeout_scheduler_output()
    oversized_request_id = "tenant-secret-" + "x" * 10_000
    scheduler_output.scheduled_new_reqs[0].req_id = oversized_request_id

    monkeypatch.setattr(
        dump_input.logger,
        "error",
        lambda message, *args: logs.append(message % args),
    )
    dump_input._dump_engine_timeout_context(make_timeout_config(), scheduler_output, {})

    combined_logs = "\n".join(logs)
    assert oversized_request_id not in combined_logs
    assert '"request_id_truncated": true' in combined_logs
    assert '"request_id_length": 10014' in combined_logs
    assert '"request_id_sha256"' in combined_logs
    assert len(logs[0]) <= dump_input.ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS + 100


def test_engine_execution_timeout_serialized_summary_has_hard_limit():
    serialized = dump_input._serialize_diagnostic({"value": "x" * 100_000})
    payload = json.loads(serialized)

    assert len(serialized) <= dump_input.ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS
    assert payload["diagnostic_output_truncated"] is True
    assert payload["original_length"] > len(serialized)
    assert len(payload["sha256"]) == 64
    assert payload["diagnostic_prefix"]


def test_engine_execution_timeout_logs_request_sample_truncation(monkeypatch):
    logs = []
    scheduler_output = make_timeout_scheduler_output()
    template_request = scheduler_output.scheduled_new_reqs[0]
    scheduler_output.scheduled_new_reqs = []
    for index in range(dump_input.ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT + 1):
        request_fields = vars(template_request).copy()
        request_fields["req_id"] = f"new-request-{index}"
        scheduler_output.scheduled_new_reqs.append(SimpleNamespace(**request_fields))

    monkeypatch.setattr(
        dump_input.logger,
        "error",
        lambda message, *args: logs.append(message % args),
    )
    dump_input._dump_engine_timeout_context(make_timeout_config(), scheduler_output, {})

    output_summary = logs[0]
    assert '"request_samples_truncated": true' in output_summary
    assert "new-request-0" in output_summary
    assert "new-request-20" in output_summary
    assert "new-request-19" not in output_summary


def test_engine_execution_timeout_spreads_samples_across_cached_requests():
    scheduler_output = make_timeout_scheduler_output()
    scheduler_output.scheduled_new_reqs = []
    cached_requests = scheduler_output.scheduled_cached_reqs
    cached_requests.req_ids = [f"cached-request-{index}" for index in range(100)]
    cached_requests.num_reqs = len(cached_requests.req_ids)
    cached_requests.all_token_ids = {
        request_id: [1] for request_id in cached_requests.req_ids
    }
    cached_requests.new_block_ids = [([],)] * cached_requests.num_reqs
    cached_requests.new_token_ids = [[]] * cached_requests.num_reqs
    cached_requests.num_computed_tokens = [1] * cached_requests.num_reqs
    cached_requests.num_output_tokens = [1] * cached_requests.num_reqs
    cached_requests.resumed_req_ids = set()

    samples = dump_input._make_request_samples(scheduler_output)

    assert [sample["request_id"] for sample in samples] == [
        "cached-request-0",
        "cached-request-5",
        "cached-request-10",
        "cached-request-15",
        "cached-request-20",
        "cached-request-26",
        "cached-request-31",
        "cached-request-36",
        "cached-request-41",
        "cached-request-46",
        "cached-request-52",
        "cached-request-57",
        "cached-request-62",
        "cached-request-67",
        "cached-request-72",
        "cached-request-78",
        "cached-request-83",
        "cached-request-88",
        "cached-request-93",
        "cached-request-99",
    ]


def test_engine_execution_timeout_cached_sample_includes_sampling_params():
    scheduler_output = make_timeout_scheduler_output()
    scheduler_output.scheduled_new_reqs = []
    samples = dump_input._make_request_samples(
        scheduler_output,
        cached_sampling_params={"request-456": {"temperature": 0.25, "top_p": 0.9}},
    )

    assert samples[0]["request_kind"] == "cached"
    assert samples[0]["sampling_params"] == {
        "temperature": 0.25,
        "top_p": 0.9,
    }


def test_engine_execution_timeout_cached_sample_marks_unreported_tokens_unknown():
    scheduler_output = make_timeout_scheduler_output()
    scheduler_output.scheduled_new_reqs = []
    scheduler_output.scheduled_cached_reqs.all_token_ids = {}

    samples = dump_input._make_request_samples(scheduler_output)

    assert samples[0]["num_all_tokens"] is None


def test_dump_on_slow_execution_arms_and_disarms_watchdog():
    calls: list[Any] = []
    scheduler_state = {"num_running_reqs": 2}

    class FakeWatchdog:
        enabled = True

        def arm(self, scheduler_output, state, stage):
            calls.append(("arm", scheduler_output, state, stage))
            return 123

        def disarm(self, generation):
            calls.append(("disarm", generation))

    engine = SimpleNamespace(
        execution_timeout_watchdog=FakeWatchdog(),
    )
    scheduler_output = SimpleNamespace()

    with EngineCore.dump_on_slow_execution(
        engine,
        scheduler_output,
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        scheduler_state,
    ):
        calls.append("body")

    assert calls == [
        (
            "arm",
            scheduler_output,
            scheduler_state,
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        ),
        "body",
        ("disarm", 123),
    ]


def test_dump_on_slow_execution_disarms_watchdog_after_exception():
    calls: list[Any] = []
    scheduler_state = {"num_running_reqs": 2}

    class FakeWatchdog:
        enabled = True

        def arm(self, scheduler_output, state, stage):
            calls.append(("arm", scheduler_output, state, stage))
            return 123

        def disarm(self, generation):
            calls.append(("disarm", generation))

    engine = SimpleNamespace(
        execution_timeout_watchdog=FakeWatchdog(),
    )
    scheduler_output = SimpleNamespace()

    try:
        with EngineCore.dump_on_slow_execution(
            engine,
            scheduler_output,
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
            scheduler_state,
        ):
            raise RuntimeError("stage failed")
    except RuntimeError as err:
        assert str(err) == "stage failed"
    else:
        raise AssertionError("expected stage failure")

    assert calls == [
        (
            "arm",
            scheduler_output,
            scheduler_state,
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        ),
        ("disarm", 123),
    ]


def test_dump_on_slow_execution_disabled_skips_snapshot():
    calls = []

    class FakeWatchdog:
        enabled = False

        def arm(self, scheduler_output, state, stage):
            calls.append(("arm", state))
            return None

        def disarm(self, generation):
            calls.append(("disarm", generation))

    def fail_snapshot(_output):
        raise AssertionError("disabled watchdog must not collect a snapshot")

    engine = SimpleNamespace(
        execution_timeout_watchdog=FakeWatchdog(),
        _make_scheduler_timeout_state=fail_snapshot,
    )

    scheduler_state = EngineCore._prepare_timeout_diagnostic_state(
        engine, SimpleNamespace()
    )
    with EngineCore.dump_on_slow_execution(
        engine,
        SimpleNamespace(),
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        scheduler_state,
    ):
        pass

    assert calls == [("arm", {}), ("disarm", None)]


def test_engine_shutdown_stops_watchdog_before_teardown(monkeypatch):
    teardown_order = []
    monkeypatch.setattr(
        engine_core_module.gc,
        "unfreeze",
        lambda: teardown_order.append("gc"),
    )
    monkeypatch.setattr(
        engine_core_module,
        "cleanup_dist_env_and_memory",
        lambda: teardown_order.append("distributed"),
    )
    engine = SimpleNamespace(
        execution_timeout_watchdog=SimpleNamespace(
            stop=lambda: teardown_order.append("watchdog")
        ),
        structured_output_manager=SimpleNamespace(
            clear_backend=lambda: teardown_order.append("structured_output")
        ),
        model_executor=SimpleNamespace(
            shutdown=lambda: teardown_order.append("executor")
        ),
        scheduler=SimpleNamespace(shutdown=lambda: teardown_order.append("scheduler")),
    )

    EngineCore.shutdown(engine)

    assert teardown_order == [
        "watchdog",
        "structured_output",
        "executor",
        "scheduler",
        "gc",
        "distributed",
    ]


def test_make_scheduler_timeout_state_is_rich_and_non_mutating():
    sampling_params = (
        make_timeout_scheduler_output().scheduled_new_reqs[0].sampling_params
    )
    scheduler = SimpleNamespace(
        make_timeout_diagnostic_state=lambda: {
            "kv_cache_usage": 0.75,
            "num_running_reqs": 2,
            "num_skipped_waiting_reqs": 1,
            "num_waiting_reqs": 3,
        },
    )
    scheduler_output = make_timeout_scheduler_output()
    scheduler_output.scheduled_new_reqs[0].req_id = "request-456"
    scheduler_output.scheduled_new_reqs[0].sampling_params = sampling_params
    engine = SimpleNamespace(
        scheduler=scheduler,
        _timeout_sampling_params_by_request={},
    )

    state = EngineCore._make_scheduler_timeout_state(engine, scheduler_output)

    assert state["kv_cache_usage"] == 0.75
    assert state["num_running_reqs"] == 2
    assert state["num_skipped_waiting_reqs"] == 1
    assert state["num_waiting_reqs"] == 3
    assert state["cached_request_sampling_params"]["request-456"]["temperature"] == 0.25

    scheduler_output.scheduled_new_reqs = []
    scheduler_output.finished_req_ids = {"request-456"}
    state = EngineCore._make_scheduler_timeout_state(engine, scheduler_output)

    assert state["cached_request_sampling_params"] == {}
    assert engine._timeout_sampling_params_by_request == {}


def test_make_scheduler_timeout_state_refreshes_rescheduled_request(monkeypatch):
    scheduler_output = make_timeout_scheduler_output()
    scheduler = SimpleNamespace(make_timeout_diagnostic_state=lambda: {})
    engine = SimpleNamespace(
        scheduler=scheduler,
        _timeout_sampling_params_by_request={},
    )
    calls = []

    def record_summary(sampling_params):
        calls.append(sampling_params.temperature)
        return {"temperature": sampling_params.temperature}

    monkeypatch.setattr(
        engine_core_module, "make_sampling_params_summary", record_summary
    )

    EngineCore._make_scheduler_timeout_state(engine, scheduler_output)
    scheduler_output.scheduled_new_reqs[0].sampling_params.temperature = 0.75
    EngineCore._make_scheduler_timeout_state(engine, scheduler_output)

    assert calls == [0.25, 0.75]
    assert (
        engine._timeout_sampling_params_by_request["request-123"]["temperature"] == 0.75
    )


def test_make_scheduler_timeout_state_failure_is_nonfatal():
    def fail_snapshot():
        raise RuntimeError("snapshot failed")

    engine = SimpleNamespace(
        scheduler=SimpleNamespace(make_timeout_diagnostic_state=fail_snapshot)
    )

    assert EngineCore._make_scheduler_timeout_state(
        engine, make_timeout_scheduler_output()
    ) == {"cached_request_sampling_params": {}}


def test_make_scheduler_timeout_state_failure_still_refreshes_sampling_cache():
    def fail_snapshot():
        raise RuntimeError("snapshot failed")

    scheduler_output = make_timeout_scheduler_output()
    request = scheduler_output.scheduled_new_reqs[0]
    request.req_id = "reused-request-id"
    scheduler_output.finished_req_ids = {request.req_id}
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(make_timeout_diagnostic_state=fail_snapshot),
        _timeout_sampling_params_by_request={
            request.req_id: {"temperature": 0.99},
            "request-456": {"temperature": 0.5},
        },
    )

    state = EngineCore._make_scheduler_timeout_state(engine, scheduler_output)

    assert state == {
        "cached_request_sampling_params": {"request-456": {"temperature": 0.5}}
    }
    assert (
        engine._timeout_sampling_params_by_request[request.req_id]["temperature"]
        == 0.25
    )


def test_scheduler_timeout_diagnostic_state_includes_resource_pressure():
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(usage=0.75),
        running=[1, 2],
        skipped_waiting=[1],
        waiting=[1, 2, 3],
    )

    assert Scheduler.make_timeout_diagnostic_state(scheduler) == {
        "kv_cache_usage": 0.75,
        "num_running_reqs": 2,
        "num_skipped_waiting_reqs": 1,
        "num_waiting_reqs": 3,
    }


def test_scheduler_timeout_diagnostic_state_keeps_custom_schedulers_compatible():
    scheduler = SimpleNamespace(get_request_counts=lambda: (2, 3))

    assert "make_timeout_diagnostic_state" not in SchedulerInterface.__abstractmethods__
    assert SchedulerInterface.make_timeout_diagnostic_state(scheduler) == {
        "num_running_reqs": 2,
        "num_waiting_reqs": 3,
    }


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
        engine_core_module.EXECUTE_MODEL_STAGE,
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        engine_core_module.SAMPLE_TOKENS_STAGE,
    ]
    assert engine.events == [
        ("enter", engine_core_module.EXECUTE_MODEL_STAGE),
        ("call", "execute_model"),
        ("exit", engine_core_module.EXECUTE_MODEL_STAGE),
        ("enter", engine_core_module.EXECUTE_MODEL_WAIT_STAGE),
        ("call", "execute_model.result"),
        ("exit", engine_core_module.EXECUTE_MODEL_WAIT_STAGE),
        ("enter", engine_core_module.SAMPLE_TOKENS_STAGE),
        ("call", "sample_tokens"),
        ("exit", engine_core_module.SAMPLE_TOKENS_STAGE),
    ]
    assert engine.prepared_timeout_outputs == [scheduler_output]
    assert all(state is engine.stage_states[0] for state in engine.stage_states)


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
    (
        future,
        queued_scheduler_output,
        exec_future,
        future_stage,
        timeout_state,
    ) = engine.batch_queue[0]
    assert future is model_executor.sample_future
    assert queued_scheduler_output is scheduler_output
    assert exec_future is model_executor.execute_future
    assert future_stage == engine_core_module.SAMPLE_TOKENS_WAIT_STAGE
    assert timeout_state is engine.stage_states[0]
    assert model_executor.sample_calls == [("grammar", True)]
    assert engine.stages == [
        engine_core_module.EXECUTE_MODEL_STAGE,
        engine_core_module.SAMPLE_TOKENS_STAGE,
    ]
    assert engine.events == [
        ("enter", engine_core_module.EXECUTE_MODEL_STAGE),
        ("call", "execute_model"),
        ("exit", engine_core_module.EXECUTE_MODEL_STAGE),
        ("enter", engine_core_module.SAMPLE_TOKENS_STAGE),
        ("call", "sample_tokens"),
        ("exit", engine_core_module.SAMPLE_TOKENS_STAGE),
    ]
    assert engine.prepared_timeout_outputs == [scheduler_output]
    assert all(state is timeout_state for state in engine.stage_states)


def test_step_with_batch_queue_reuses_snapshot_for_deferred_sampling():
    old_scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        pending_structured_output_tokens=True,
    )
    scheduler = FakeStageScheduler(scheduler_output)
    model_executor = FakeStageModelExecutor(
        execute_result=SimpleNamespace(), sample_result=SimpleNamespace()
    )
    engine = FakeStageEngine(scheduler, model_executor)
    old_timeout_state = {"snapshot": "old"}
    old_model_output = SimpleNamespace()
    engine.batch_queue = deque(
        [
            (
                FakeFuture(
                    old_model_output,
                    model_executor.events,
                    "old_sample_tokens.result",
                ),
                old_scheduler_output,
                FakeFuture(
                    SimpleNamespace(),
                    model_executor.events,
                    "old_execute_model.result",
                ),
                engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
                old_timeout_state,
            )
        ],
        maxlen=engine.batch_queue_size,
    )

    outputs, model_executed = EngineCore.step_with_batch_queue(engine)

    assert outputs == {}
    assert model_executed
    assert scheduler.updated_with == (old_scheduler_output, old_model_output)
    assert engine.prepared_timeout_outputs == [scheduler_output]
    assert engine.stages == [
        engine_core_module.EXECUTE_MODEL_STAGE,
        engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
        engine_core_module.SAMPLE_TOKENS_STAGE,
    ]
    scheduled_timeout_state = engine.stage_states[0]
    assert engine.stage_states == [
        scheduled_timeout_state,
        old_timeout_state,
        scheduled_timeout_state,
    ]
    assert engine.events == [
        ("enter", engine_core_module.EXECUTE_MODEL_STAGE),
        ("call", "execute_model"),
        ("exit", engine_core_module.EXECUTE_MODEL_STAGE),
        ("enter", engine_core_module.SAMPLE_TOKENS_WAIT_STAGE),
        ("call", "old_sample_tokens.result"),
        ("exit", engine_core_module.SAMPLE_TOKENS_WAIT_STAGE),
        ("enter", engine_core_module.SAMPLE_TOKENS_STAGE),
        ("call", "sample_tokens"),
        ("exit", engine_core_module.SAMPLE_TOKENS_STAGE),
    ]
    assert len(engine.batch_queue) == 1
    (
        future,
        queued_scheduler_output,
        exec_future,
        future_stage,
        timeout_state,
    ) = engine.batch_queue[0]
    assert future is model_executor.sample_future
    assert queued_scheduler_output is scheduler_output
    assert exec_future is model_executor.execute_future
    assert future_stage == engine_core_module.SAMPLE_TOKENS_WAIT_STAGE
    assert timeout_state is scheduled_timeout_state


def test_step_with_batch_queue_uses_queued_future_stage():
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    scheduler = FakeStageScheduler(scheduler_output, has_requests=False)
    model_output = SimpleNamespace()
    engine = FakeStageEngine(scheduler)
    exec_future = FakeFuture(SimpleNamespace(), engine.events, "execute_model.result")
    timeout_state = {"snapshot": 1}
    engine.batch_queue = deque(
        [
            (
                FakeFuture(model_output, engine.events, "sample_tokens.result"),
                scheduler_output,
                exec_future,
                engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
                timeout_state,
            )
        ],
        maxlen=engine.batch_queue_size,
    )

    outputs, model_executed = EngineCore.step_with_batch_queue(engine)

    assert outputs == {}
    assert not model_executed
    assert scheduler.updated_with == (scheduler_output, model_output)
    assert engine.stages == [engine_core_module.SAMPLE_TOKENS_WAIT_STAGE]
    assert engine.stage_states == [timeout_state]
    assert engine.events == [
        ("enter", engine_core_module.SAMPLE_TOKENS_WAIT_STAGE),
        ("call", "sample_tokens.result"),
        ("exit", engine_core_module.SAMPLE_TOKENS_WAIT_STAGE),
    ]
