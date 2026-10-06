# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import numpy as np
import pytest
from transformers import AutoTokenizer

from tests.v1.core.utils import EOS_TOKEN_ID, create_requests, create_scheduler, mock_kv
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import EngineCoreClient, InprocClient, SyncMPClient
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.external import (
    _reset_external_metrics_providers_for_tests,
    collect_external_metrics,
    has_external_metrics_providers,
    register_external_metrics_provider,
    unregister_external_metrics_provider,
)
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def reset_external_metrics_providers():
    _reset_external_metrics_providers_for_tests()
    yield
    _reset_external_metrics_providers_for_tests()


def test_external_metrics_provider_is_namespaced_and_rate_limited():
    calls = 0

    def collect():
        nonlocal calls
        calls += 1
        return {"used_bytes": 42, "labels": {"pool": "kv"}}

    register_external_metrics_provider(
        "example.plugin", collect, collection_interval_s=2.0
    )

    assert collect_external_metrics(now=10.0) == {
        "example.plugin": {"used_bytes": 42, "labels": {"pool": "kv"}}
    }
    assert collect_external_metrics(now=11.9) is None
    assert collect_external_metrics(now=12.0) == {
        "example.plugin": {"used_bytes": 42, "labels": {"pool": "kv"}}
    }
    assert calls == 2


def test_external_metrics_provider_failures_are_isolated():
    def fail():
        raise RuntimeError("provider failed")

    register_external_metrics_provider("failing", fail)
    register_external_metrics_provider("healthy", lambda: {"value": 1})

    assert collect_external_metrics(now=10.0) == {"healthy": {"value": 1}}


def test_forced_external_metrics_collection_resets_interval():
    calls = 0

    def collect():
        nonlocal calls
        calls += 1
        return {"calls": calls}

    register_external_metrics_provider("example", collect)
    assert collect_external_metrics(now=10.0) == {"example": {"calls": 1}}
    assert collect_external_metrics(now=10.1) is None
    assert collect_external_metrics(now=10.1, force=True) == {"example": {"calls": 2}}
    assert collect_external_metrics(now=11.0) is None
    assert collect_external_metrics(now=11.1) == {"example": {"calls": 3}}


@pytest.mark.parametrize(
    "payload",
    [
        {"bad": object()},
        {"bad": {1: "non-string key"}},
        {"bad": float("nan")},
        {"bad": [float("inf")]},
        {"bad": -(2**63) - 1},
        {"bad": 2**64},
        {"bad": {"nested": [2**64]}},
        {"bad": "\ud800"},
        {"\udfff": "invalid key"},
        ["not", "a", "mapping"],
    ],
)
def test_invalid_external_metrics_payload_is_isolated(payload: object):
    register_external_metrics_provider(
        "invalid",
        lambda: payload,  # type: ignore[arg-type,return-value]
    )
    register_external_metrics_provider("healthy", lambda: {"value": 1})

    assert collect_external_metrics(now=10.0) == {"healthy": {"value": 1}}


@pytest.mark.parametrize("name", ["", "1plugin", "bad name", "bad/name"])
def test_external_metrics_provider_name_is_validated(name: str):
    with pytest.raises(ValueError, match="provider names"):
        register_external_metrics_provider(name, lambda: {})


def test_external_metrics_provider_registration_is_unique():
    assert not has_external_metrics_providers()
    register_external_metrics_provider("example", lambda: {})
    assert has_external_metrics_providers()

    with pytest.raises(ValueError, match="already registered"):
        register_external_metrics_provider("example", lambda: {})

    unregister_external_metrics_provider("example")
    assert not has_external_metrics_providers()
    register_external_metrics_provider("example", lambda: {"value": 2})
    assert collect_external_metrics(now=10.0) == {"example": {"value": 2}}


def test_external_metrics_provider_configuration_is_validated():
    with pytest.raises(TypeError, match="must be callable"):
        register_external_metrics_provider("example", object())  # type: ignore[arg-type]

    for interval in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite value greater than zero"):
            register_external_metrics_provider(
                "example", lambda: {}, collection_interval_s=interval
            )


@pytest.mark.parametrize("integer", [-(2**63), 2**64 - 1])
def test_external_metrics_snapshot_is_msgpack_serializable(integer: int):
    register_external_metrics_provider(
        "example.plugin",
        lambda: {
            "used_bytes": integer,
            "labels": {"pool": "kv"},
            "buckets": [1.0, 2.0, integer],
            "ratio": np.float64(0.5),
            "enabled": True,
            "optional": None,
        },
    )
    stats = SchedulerStats(
        external_metrics=collect_external_metrics(now=10.0),
    )

    encoded = msgspec.msgpack.encode(stats)
    decoded = msgspec.msgpack.decode(encoded, type=SchedulerStats)

    assert decoded.external_metrics == stats.external_metrics
    assert decoded.external_metrics is not None
    assert decoded.external_metrics["example.plugin"]["used_bytes"] == integer
    assert decoded.external_metrics["example.plugin"]["enabled"] is True
    assert decoded.external_metrics["example.plugin"]["ratio"] == 0.5


@pytest.mark.parametrize("multiprocess", [False, True])
@pytest.mark.parametrize("finish_method", ["stop", "abort", "eos", "remote_stop"])
def test_final_metrics_reach_logger_after_frontend_finish(
    monkeypatch, multiprocess, finish_method
):
    """Flush local work after termination without waiting for a remote decoder."""
    monkeypatch.setattr("vllm.v1.metrics.external.time.monotonic", lambda: 10.0)
    scheduler: Scheduler
    register_external_metrics_provider(
        "example",
        lambda: {
            "requests": len(scheduler.requests),
            "cache_usage": scheduler.get_kv_cache_usage(),
        },
    )
    overlapping = finish_method == "eos"
    remote = finish_method == "remote_stop"
    scheduler = create_scheduler(
        async_scheduling=overlapping,
        pipeline_parallel_size=2 if overlapping else 1,
        use_kv_connector=mock_kv(0, False) if overlapping or remote else None,
        kv_role="kv_consumer",
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    stop_token = tokenizer.encode("!", add_special_tokens=False)[0]
    (request,) = create_requests(
        num_requests=1, num_tokens=4, ignore_eos=not overlapping
    )
    scheduler.add_request(request)
    requests = [request]
    if remote:
        (transfer,) = create_requests(
            num_requests=1, num_tokens=4, max_tokens=1, req_ids=["transfer"]
        )
        scheduler.add_request(transfer)
        requests.append(transfer)

        def request_finished(req, block_ids):
            if req is transfer:
                return True, {"remote_request_id": "transfer"}
            return False, None

        monkeypatch.setattr(scheduler.connector, "request_finished", request_finished)

    core = EngineCoreProc.__new__(EngineCoreProc)
    core.scheduler = scheduler
    core.batch_queue = deque() if overlapping else None
    core.batch_queue_size = 3
    core.engines_running = False
    core.check_for_draft_tokens = False
    core._idle_state_callbacks = []
    core.aborts_queue = queue.Queue()
    core.log_stats = True
    core.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_logging_iteration_details=False)
    )
    core.is_mm_encoder_only = False
    core.is_pooling_model = False
    core.log_error_detail = lambda _: nullcontext()
    submissions = 0
    last_future = None

    def execute_model(scheduled, non_block):
        nonlocal submissions, last_future
        submissions += 1
        # Fail promptly if returning a result waits for its remote consumer.
        assert submissions <= 8, "Waiting for remote cleanup before returning outputs"
        req_ids = list(scheduled.num_scheduled_tokens)
        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[
                [EOS_TOKEN_ID if overlapping else stop_token] for _ in req_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        last_future = Future()
        last_future.set_result(output)
        return last_future

    core.model_executor = SimpleNamespace(
        execute_model=execute_model,
        sample_tokens=lambda *args, **kwargs: last_future,
    )
    core.step_fn = core.step_with_batch_queue if overlapping else core.step
    client: EngineCoreClient
    if multiprocess:
        client = SyncMPClient.__new__(SyncMPClient)
        client.outputs_queue = queue.Queue()
        client.engines_running = False
        client.resources = SimpleNamespace(engine_dead=False)
        core.output_queue = SimpleNamespace(
            put_nowait=lambda item: client.outputs_queue.put_nowait(item[1])
        )
        while client.outputs_queue.empty():
            core._process_engine_step()
        # A previous batch can already be queued when the frontend stops.
        client.outputs_queue.put_nowait(
            EngineCoreOutputs(
                scheduler_stats=SchedulerStats(
                    external_metrics={"example": {"requests": 1}}
                )
            )
        )
        monkeypatch.setattr(client, "_send_input", core._handle_client_request)

        def call_utility(method, *args):
            result = getattr(core, method)(*args)
            if isinstance(result, Future):
                while core.has_work():
                    core._process_engine_step()
                core._notify_idle_state_callbacks()
                result = result.result()
            return msgspec.msgpack.decode(msgspec.msgpack.encode(result))

        monkeypatch.setattr(client, "call_utility", call_utility)
    else:
        client = InprocClient.__new__(InprocClient)
        client.engine_core = core

    engine = LLMEngine.__new__(LLMEngine)
    engine.engine_core = client
    engine.log_stats = True
    engine.dp_group = None
    engine.should_execute_dummy_batch = False
    engine.renderer = SimpleNamespace(stat_mm_cache=lambda: None)
    engine.logger_manager = Mock()
    engine.output_processor = OutputProcessor(tokenizer, log_stats=True)
    for req in requests:
        engine.output_processor.add_request(
            EngineCoreRequest(
                request_id=req.request_id,
                external_req_id=req.request_id,
                prompt_token_ids=req.prompt_token_ids,
                mm_features=None,
                sampling_params=SamplingParams(
                    stop="!" if finish_method in ("stop", "remote_stop") else None,
                    ignore_eos=not overlapping,
                    max_tokens=req.max_tokens,
                ),
                pooling_params=None,
                arrival_time=req.arrival_time,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            ),
            prompt=None,
        )
    outputs = engine.step()
    while not outputs:
        outputs = engine.step()
    if finish_method == "abort":
        assert not outputs[0].finished
        engine.abort_request([request.request_id])
    else:
        assert outputs[0].finished
        if not overlapping:
            assert outputs[0].outputs[0].stop_reason == "!"
    assert not engine.has_unfinished_requests()
    if remote:
        assert scheduler.has_requests()
        assert transfer.request_id in scheduler.requests
        assert outputs[1].kv_transfer_params == {"remote_request_id": "transfer"}
    else:
        snapshots = [
            call.kwargs["scheduler_stats"].external_metrics
            for call in engine.logger_manager.record.call_args_list
            if call.kwargs["scheduler_stats"].external_metrics is not None
        ]
        assert snapshots[-1] == {"example": {"requests": 0, "cache_usage": 0.0}}
