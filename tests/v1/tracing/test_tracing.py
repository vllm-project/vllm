# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# type: ignore
import pytest
import time
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform
from vllm.tracing import SpanAttributes

# Import shared fixtures from the tracing conftest
from tests.tracing.conftest import (  # noqa: F401
    FAKE_TRACE_SERVER_ADDRESS,
    FakeTraceService,
    trace_service,
)


def test_traces(
    monkeypatch: pytest.MonkeyPatch,
    trace_service: FakeTraceService,
):
    with monkeypatch.context() as m:
        m.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        # The fake OTLP server starts gRPC worker threads before the engine
        # core is launched. gRPC's C-core is not fork-safe and can segfault
        # if forked.
        m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        sampling_params = SamplingParams(
            temperature=0.01,
            top_p=0.1,
            max_tokens=256,
        )
        model = "facebook/opt-125m"
        llm = None
        try:
            llm = LLM(
                model=model,
                otlp_traces_endpoint=FAKE_TRACE_SERVER_ADDRESS,
                gpu_memory_utilization=0.3,
                disable_log_stats=False,
            )
            prompts = ["This is a short prompt"]
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            print(f"test_traces outputs is : {outputs}")

            # Wait for the "llm_request" span to be exported.
            # The BatchSpanProcessor batches spans and exports them periodically,
            # so we need to wait specifically for the llm_request span to appear.
            timeout = 15
            deadline = time.time() + timeout
            llm_request_spans = []
            while time.time() < deadline:
                all_spans = trace_service.get_all_spans()
                llm_request_spans = [s for s in all_spans if s["name"] == "llm_request"]
                if llm_request_spans:
                    break
                time.sleep(0.5)

            assert len(llm_request_spans) == 1, (
                f"Expected exactly 1 'llm_request' span, but got "
                f"{len(llm_request_spans)}. "
                f"All span names: {[s['name'] for s in all_spans]}"
            )

            attributes = llm_request_spans[0]["attributes"]
            # assert attributes.get(SpanAttributes.GEN_AI_RESPONSE_MODEL) == model
            assert (
                attributes.get(SpanAttributes.GEN_AI_REQUEST_ID)
                == outputs[0].request_id
            )
            assert (
                attributes.get(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE)
                == sampling_params.temperature
            )
            assert (
                attributes.get(SpanAttributes.GEN_AI_REQUEST_TOP_P)
                == sampling_params.top_p
            )
            assert (
                attributes.get(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS)
                == sampling_params.max_tokens
            )
            assert attributes.get(SpanAttributes.GEN_AI_REQUEST_N) == sampling_params.n
            assert attributes.get(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS) == len(
                outputs[0].prompt_token_ids
            )
            completion_tokens = sum(len(o.token_ids) for o in outputs[0].outputs)
            assert (
                attributes.get(SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS)
                == completion_tokens
            )

            assert attributes.get(SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE) > 0
            assert attributes.get(SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN) > 0
            assert attributes.get(SpanAttributes.GEN_AI_LATENCY_E2E) > 0

            # Without the semconv opt-in, none of the current GenAI attributes
            # are emitted (default behavior unchanged).
            assert attributes.get(SpanAttributes.GEN_AI_REQUEST_MODEL) is None
            assert attributes.get(SpanAttributes.GEN_AI_OPERATION_NAME) is None
        finally:
            if llm is not None:
                shutdown_timeout = 60.0 if current_platform.is_rocm() else 5.0
                llm.llm_engine.engine_core.shutdown(timeout=shutdown_timeout)
            cleanup_dist_env_and_memory()


def test_traces_gen_ai_latest_semconv(
    monkeypatch: pytest.MonkeyPatch,
    trace_service: FakeTraceService,
):
    """With OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental, the span
    additionally carries gen_ai.request.model while still emitting the legacy
    attributes (non-breaking dual-emit).

    This drives the offline LLM path, so gen_ai.operation.name (an
    endpoint-level value) is not set and the span name stays "llm_request".
    """
    with monkeypatch.context() as m:
        m.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        m.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")

        sampling_params = SamplingParams(temperature=0.01, top_p=0.1, max_tokens=256)
        model = "facebook/opt-125m"
        llm = None
        try:
            llm = LLM(
                model=model,
                otlp_traces_endpoint=FAKE_TRACE_SERVER_ADDRESS,
                gpu_memory_utilization=0.3,
                disable_log_stats=False,
            )
            outputs = llm.generate(
                ["This is a short prompt"], sampling_params=sampling_params
            )

            timeout = 15
            deadline = time.time() + timeout
            llm_request_spans = []
            while time.time() < deadline:
                all_spans = trace_service.get_all_spans()
                llm_request_spans = [s for s in all_spans if s["name"] == "llm_request"]
                if llm_request_spans:
                    break
                time.sleep(0.5)

            assert len(llm_request_spans) == 1
            attributes = llm_request_spans[0]["attributes"]

            # Current OTel GenAI semconv attribute is emitted...
            assert attributes.get(SpanAttributes.GEN_AI_REQUEST_MODEL) == model
            # ...alongside the legacy names (dual-emit, non-breaking).
            assert attributes.get(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS) == len(
                outputs[0].prompt_token_ids
            )
            # Offline path: no endpoint -> operation.name unset, default span name.
            assert attributes.get(SpanAttributes.GEN_AI_OPERATION_NAME) is None
        finally:
            if llm is not None:
                shutdown_timeout = 60.0 if current_platform.is_rocm() else 5.0
                llm.llm_engine.engine_core.shutdown(timeout=shutdown_timeout)
            cleanup_dist_env_and_memory()
