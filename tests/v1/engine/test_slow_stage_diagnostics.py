# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
from concurrent.futures import Future
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import vllm.v1.engine.core as engine_core_module
from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.logging_utils import dump_input
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore


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
        offload_config=SimpleNamespace(
            offload_backend="auto",
            uva=SimpleNamespace(cpu_offload_gb=0),
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


def make_timeout_snapshot(
    scheduler_output: SimpleNamespace | None = None,
    scheduler_state: dict[str, Any] | None = None,
) -> dump_input.EngineExecutionTimeoutSnapshot:
    return dump_input.make_engine_execution_timeout_snapshot(
        scheduler_output or make_timeout_scheduler_output(),
        scheduler_state,
    )


def test_engine_execution_timeout_watchdog_fails_open_on_thread_start_error(
    monkeypatch,
):
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=10.0,
    )

    def fail_start():
        raise RuntimeError("thread limit reached")

    failing_thread = SimpleNamespace(start=fail_start)
    monkeypatch.setattr(watchdog, "_create_thread", lambda: failing_thread)

    watchdog.start()
    generation = watchdog.arm(
        make_timeout_snapshot(),
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )

    assert generation is None
    assert watchdog._thread is None
    assert not watchdog.enabled


def test_engine_execution_timeout_watchdog_ignores_stale_disarm(monkeypatch):
    now_s = [0.0]
    dumps = []
    dump_completed = threading.Event()

    def record_dump(config, snapshot, timeout_s, stage):
        dumps.append((snapshot, stage))
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
            make_timeout_snapshot(stale_output, {"snapshot": 1}),
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        watchdog.arm(
            make_timeout_snapshot(current_output, {"snapshot": 2}),
            engine_core_module.SAMPLE_TOKENS_STAGE,
        )
        watchdog.disarm(stale_generation)
        now_s[0] = 11.0
        watchdog._wake_event.set()

        assert dump_completed.wait(timeout=1.0)
        assert len(dumps) == 1
        snapshot, stage = dumps[0]
        assert snapshot.scheduler_output_summary["request_samples"][0][
            "request_id"
        ] == ("current-generation-request")
        assert snapshot.scheduler_queue_summary == {"snapshot": 2}
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
            make_timeout_snapshot(scheduler_state={"snapshot": 1}),
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        assert dump_started.wait(timeout=1.0)

        def arm_again():
            generations.append(
                watchdog.arm(
                    make_timeout_snapshot(scheduler_state={"snapshot": 2}),
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


def test_engine_execution_timeout_real_timeout_emits_useful_summary(monkeypatch):
    logs = []
    traceback_dumped = threading.Event()

    def record_log(message, *args):
        logs.append(message % args)

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
            make_timeout_snapshot(
                scheduler_state={"num_running_reqs": 1, "num_waiting_reqs": 0}
            ),
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        assert traceback_dumped.wait(timeout=1.0)
    finally:
        watchdog.stop()

    combined_logs = "\n".join(logs)
    assert engine_core_module.EXECUTE_MODEL_WAIT_STAGE in combined_logs
    assert "request-123" in combined_logs
    assert "request-456" in combined_logs
    assert "temperature" in combined_logs
    assert "num_running_reqs" in combined_logs
    assert "private-stop-string" not in combined_logs
    assert "private/model/path" not in combined_logs


def test_engine_execution_timeout_throttle_is_per_watchdog_and_stage():
    now_s = [100.0]

    def make_watchdog():
        return dump_input.EngineExecutionTimeoutWatchdog(
            config=make_timeout_config(),
            timeout_s=1.0,
            time_fn=lambda: now_s[0],
        )

    first_watchdog = make_watchdog()
    second_watchdog = make_watchdog()
    stage = engine_core_module.EXECUTE_MODEL_WAIT_STAGE

    assert first_watchdog._mark_dump_if_allowed(stage)
    assert not first_watchdog._mark_dump_if_allowed(stage)
    assert first_watchdog._mark_dump_if_allowed(
        engine_core_module.SAMPLE_TOKENS_WAIT_STAGE
    )
    assert second_watchdog._mark_dump_if_allowed(stage)

    now_s[0] += dump_input.ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S
    assert first_watchdog._mark_dump_if_allowed(stage)


def test_engine_timeout_config_allowlists_match_real_configs(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 32,
                "intermediate_size": 64,
                "max_position_embeddings": 32,
                "model_type": "llama",
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "vocab_size": 32,
            }
        ),
        encoding="utf-8",
    )
    config = VllmConfig(model_config=ModelConfig(model=str(tmp_path)))
    allowlists = (
        (config.model_config, dump_input._ENGINE_TIMEOUT_MODEL_CONFIG_FIELDS),
        (config.parallel_config, dump_input._ENGINE_TIMEOUT_PARALLEL_CONFIG_FIELDS),
        (config.scheduler_config, dump_input._ENGINE_TIMEOUT_SCHEDULER_CONFIG_FIELDS),
        (config.cache_config, dump_input._ENGINE_TIMEOUT_CACHE_CONFIG_FIELDS),
        (config.offload_config, dump_input._ENGINE_TIMEOUT_OFFLOAD_CONFIG_FIELDS),
        (
            config.offload_config.uva,
            dump_input._ENGINE_TIMEOUT_UVA_OFFLOAD_CONFIG_FIELDS,
        ),
    )

    for config_object, field_names in allowlists:
        assert not [name for name in field_names if not hasattr(config_object, name)]
    speculative_fields = {field.name for field in fields(SpeculativeConfig)}
    assert set(dump_input._ENGINE_TIMEOUT_SPECULATIVE_CONFIG_FIELDS).issubset(
        speculative_fields
    )

    summary = dump_input._make_engine_config_summary(config)
    assert "swap_space_bytes" not in summary["cache"]
    assert summary["offload"]["cpu_offload_gb"] == 0


def test_engine_execution_timeout_snapshot_uses_scheduler_dataclasses():
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id="new-request",
                prompt_token_ids=[1, 2, 3],
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=8, temperature=0.25),
                pooling_params=None,
                block_ids=([1, 2],),
                num_computed_tokens=0,
                lora_request=None,
            )
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["cached-request"],
            resumed_req_ids={"cached-request"},
            new_token_ids=[[4]],
            all_token_ids={"cached-request": [1, 2, 3, 4]},
            new_block_ids=[([3],)],
            num_computed_tokens=[3],
            num_output_tokens=[1],
        ),
        num_scheduled_tokens={"new-request": 3, "cached-request": 1},
        total_num_scheduled_tokens=4,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    snapshot = dump_input.make_engine_execution_timeout_snapshot(
        scheduler_output,
        {"cached_request_sampling_params": {"cached-request": {"temperature": 0.5}}},
    )

    samples = snapshot.scheduler_output_summary["request_samples"]
    assert [sample["request_id"] for sample in samples] == [
        "new-request",
        "cached-request",
    ]
    assert samples[0]["num_prefill_tokens"] is None
    assert samples[0]["sampling_params"]["temperature"] == 0.25
    assert samples[1]["is_resumed"]
    assert samples[1]["num_all_tokens"] == 4
    assert samples[1]["sampling_params"] == {"temperature": 0.5}


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
    dump_input._dump_engine_timeout_context(
        make_timeout_config(), make_timeout_snapshot(scheduler_output)
    )

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


def test_timeout_diagnostics_monitors_only_incomplete_futures():
    calls: list[Any] = []
    snapshot = make_timeout_snapshot(scheduler_state={"num_running_reqs": 2})

    class FakeWatchdog:
        enabled = True

        def arm(self, armed_snapshot, stage):
            calls.append(("arm", armed_snapshot, stage))
            return 123

        def disarm(self, generation):
            calls.append(("disarm", generation))

    diagnostics = dump_input.EngineExecutionTimeoutDiagnostics(
        config=make_timeout_config(), timeout_s=1, scheduler_state_fn=lambda: {}
    )
    diagnostics._watchdog = FakeWatchdog()
    future = Future()
    try:
        with diagnostics.monitor_future(
            future,
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
            snapshot,
        ):
            raise RuntimeError("stage failed")
    except RuntimeError as err:
        assert str(err) == "stage failed"
    else:
        raise AssertionError("expected stage failure")

    future.set_result(None)
    with diagnostics.monitor_future(
        future,
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        snapshot,
    ):
        pass

    assert calls == [
        (
            "arm",
            snapshot,
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        ),
        ("disarm", 123),
    ]


def test_timeout_diagnostics_disabled_is_lazy():
    def fail_snapshot():
        raise AssertionError("disabled watchdog must not collect a snapshot")

    diagnostics = dump_input.EngineExecutionTimeoutDiagnostics(
        config=make_timeout_config(), timeout_s=0, scheduler_state_fn=fail_snapshot
    )

    diagnostics.start()
    scheduler_state = diagnostics.make_snapshot(SimpleNamespace())
    with diagnostics.monitor(
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        scheduler_state,
    ):
        pass
    diagnostics.stop()

    assert scheduler_state.scheduler_output_summary == {}
    assert diagnostics._watchdog._thread is None
    assert not diagnostics._watchdog.enabled


def test_timeout_diagnostics_snapshot_fails_open(monkeypatch):
    def fail_snapshot(*args):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(
        dump_input,
        "make_engine_execution_timeout_snapshot",
        fail_snapshot,
    )
    diagnostics = dump_input.EngineExecutionTimeoutDiagnostics(
        config=make_timeout_config(), timeout_s=1, scheduler_state_fn=lambda: {}
    )

    snapshot = diagnostics.make_snapshot(make_timeout_scheduler_output())

    assert snapshot.scheduler_output_summary == {
        "scheduler_output_summary_unavailable": True
    }
    assert snapshot.scheduler_queue_summary == {}


def test_engine_shutdown_stops_timeout_diagnostics_before_teardown(monkeypatch):
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
        execution_timeout_diagnostics=SimpleNamespace(
            stop=lambda: teardown_order.append("diagnostics")
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
        "diagnostics",
        "structured_output",
        "executor",
        "scheduler",
        "gc",
        "distributed",
    ]


def test_timeout_diagnostics_snapshot_is_rich_and_prunes_finished_requests():
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
    diagnostics = dump_input.EngineExecutionTimeoutDiagnostics(
        config=make_timeout_config(),
        timeout_s=1,
        scheduler_state_fn=scheduler.make_timeout_diagnostic_state,
    )

    snapshot = diagnostics.make_snapshot(scheduler_output)

    assert snapshot.scheduler_queue_summary == {
        "kv_cache_usage": 0.75,
        "num_running_reqs": 2,
        "num_skipped_waiting_reqs": 1,
        "num_waiting_reqs": 3,
    }
    cached_sample = next(
        sample
        for sample in snapshot.scheduler_output_summary["request_samples"]
        if sample["request_kind"] == "cached"
    )
    assert cached_sample["sampling_params"]["temperature"] == 0.25

    scheduler_output.scheduled_new_reqs = []
    scheduler_output.finished_req_ids = {"request-456"}
    snapshot = diagnostics.make_snapshot(scheduler_output)

    cached_sample = snapshot.scheduler_output_summary["request_samples"][0]
    assert cached_sample["sampling_params"] is None


def test_timeout_diagnostics_only_summarizes_sampled_requests(monkeypatch):
    scheduler_output = make_timeout_scheduler_output()
    template_request = scheduler_output.scheduled_new_reqs[0]
    requests = [SimpleNamespace(**vars(template_request)) for _ in range(100)]
    for index, request in enumerate(requests):
        request.req_id = f"request-{index}"
    scheduler_output.scheduled_new_reqs = requests
    scheduler_output.scheduled_cached_reqs.num_reqs = 0
    scheduler_output.scheduled_cached_reqs.req_ids = []
    diagnostics = dump_input.EngineExecutionTimeoutDiagnostics(
        config=make_timeout_config(), timeout_s=1, scheduler_state_fn=lambda: {}
    )
    original_make_summary = dump_input.make_sampling_params_summary
    summarized_sampling_params = []

    def record_summary(sampling_params):
        summarized_sampling_params.append(sampling_params)
        return original_make_summary(sampling_params)

    monkeypatch.setattr(dump_input, "make_sampling_params_summary", record_summary)

    snapshot = diagnostics.make_snapshot(scheduler_output)

    assert len(snapshot.scheduler_output_summary["request_samples"]) == 20
    assert len(summarized_sampling_params) == 20


def test_scheduler_timeout_diagnostic_state_supports_builtin_and_custom_schedulers():
    scheduler = SimpleNamespace(get_request_counts=lambda: (2, 3))

    assert "make_timeout_diagnostic_state" not in SchedulerInterface.__abstractmethods__
    assert SchedulerInterface.make_timeout_diagnostic_state(scheduler) == {
        "num_running_reqs": 2,
        "num_waiting_reqs": 3,
    }

    scheduler = SimpleNamespace(
        deferred_waiting={"deferred"},
        get_kv_cache_usage=lambda: 0.5,
        running=["running"],
        waiting=["deferred", "waiting"],
    )
    assert Scheduler.make_timeout_diagnostic_state(scheduler) == {
        "kv_cache_usage": 0.5,
        "num_running_reqs": 1,
        "num_skipped_waiting_reqs": 1,
        "num_waiting_reqs": 1,
    }
