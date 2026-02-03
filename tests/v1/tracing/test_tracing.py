# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# type: ignore
import pytest
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE

from vllm import LLM, SamplingParams
from vllm.tracing import SpanAttributes

# Import shared fixtures from the tracing conftest
from tests.tracing.conftest import (  # noqa: F401
    FAKE_TRACE_SERVER_ADDRESS,
    FakeTraceService,
    decode_attributes,
)


def test_traces(
    monkeypatch: pytest.MonkeyPatch,
    trace_service: FakeTraceService,
):
    with monkeypatch.context() as m:
        m.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")

        sampling_params = SamplingParams(
            temperature=0.01,
            top_p=0.1,
            max_tokens=256,
        )
        model = "facebook/opt-125m"
        llm = LLM(
            model=model,
            otlp_traces_endpoint=FAKE_TRACE_SERVER_ADDRESS,
            gpu_memory_utilization=0.3,
            disable_log_stats=False,
        )
        prompts = ["This is a short prompt"]
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        print(f"test_traces outputs is : {outputs}")

        timeout = 10
        if not trace_service.evt.wait(timeout):
            raise TimeoutError(
                f"The fake trace service didn't receive a trace within "
                f"the {timeout} seconds timeout"
            )

        request = trace_service.request
        assert len(request.resource_spans) == 1, (
            f"Expected 1 resource span, but got {len(request.resource_spans)}"
        )
        assert len(request.resource_spans[0].scope_spans) == 1, (
            f"Expected 1 scope span, "
            f"but got {len(request.resource_spans[0].scope_spans)}"
        )
        assert len(request.resource_spans[0].scope_spans[0].spans) == 1, (
            f"Expected 1 span, "
            f"but got {len(request.resource_spans[0].scope_spans[0].spans)}"
        )

        attributes = decode_attributes(
            request.resource_spans[0].scope_spans[0].spans[0].attributes
        )
        # assert attributes.get(SpanAttributes.GEN_AI_RESPONSE_MODEL) == model
        assert attributes.get(SpanAttributes.GEN_AI_REQUEST_ID) == outputs[0].request_id
        assert (
            attributes.get(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE)
            == sampling_params.temperature
        )
        assert (
            attributes.get(SpanAttributes.GEN_AI_REQUEST_TOP_P) == sampling_params.top_p
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
