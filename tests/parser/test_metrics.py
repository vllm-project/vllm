# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for parser Prometheus metrics (completed tool-call counter)."""

from __future__ import annotations

import pytest
from openai.types.responses import FunctionTool
from prometheus_client import REGISTRY

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.metrics import (
    RequestType,
    ValidationOutcome,
    classify_completed_tool_call,
    init_parser_metrics,
    record_tool_call_completed,
)

pytestmark = pytest.mark.cpu_test

_WEATHER_PARAMS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}

FUNCTION_TOOL = FunctionTool(
    type="function",
    name="get_weather",
    parameters=_WEATHER_PARAMS,
)

CHAT_TOOL = ChatCompletionToolsParam(
    type="function",
    function=FunctionDefinition(name="get_weather", parameters=_WEATHER_PARAMS),
)

NO_PARAMS_TOOL = ChatCompletionToolsParam(
    type="function",
    function=FunctionDefinition(name="no_params", parameters=None),
)

_MODEL_NAME = "test-completed-tool-calls"
_ZERO_MODEL_NAME = "test-completed-tool-calls-zeros"


def _chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    )


def _responses_request() -> ResponsesRequest:
    return ResponsesRequest(input="hi")


def _completed_samples(*, model_name: str):
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if (
                sample.name == "vllm:tool_calls_completed_total"
                and sample.labels.get("model_name") == model_name
            ):
                yield sample


def _completed_value(
    *,
    model_name: str,
    request_type: str,
    validation_outcome: str,
) -> float:
    for sample in _completed_samples(model_name=model_name):
        if (
            sample.labels.get("request_type") == request_type
            and sample.labels.get("validation_outcome") == validation_outcome
        ):
            return sample.value
    raise AssertionError(
        f"missing series model_name={model_name} "
        f"request_type={request_type} validation_outcome={validation_outcome}"
    )


class TestClassifyCompletedToolCall:
    @pytest.mark.parametrize("tools", [[FUNCTION_TOOL], [CHAT_TOOL]])
    def test_valid_args(self, tools):
        outcome = classify_completed_tool_call(
            "get_weather",
            '{"city": "Dallas"}',
            tools,
        )
        assert outcome is ValidationOutcome.VALID

    def test_unknown_tool_name(self):
        outcome = classify_completed_tool_call(
            "not_a_tool",
            '{"city": "Dallas"}',
            [FUNCTION_TOOL],
        )
        assert outcome is ValidationOutcome.UNKNOWN_TOOL

    @pytest.mark.parametrize("tools", [None, []])
    def test_unknown_tool_when_catalog_missing(self, tools):
        outcome = classify_completed_tool_call(
            "get_weather",
            '{"city": "Dallas"}',
            tools,
        )
        assert outcome is ValidationOutcome.UNKNOWN_TOOL

    def test_unknown_tool_wins_over_invalid_json(self):
        outcome = classify_completed_tool_call(
            "not_a_tool",
            "{not-json",
            [FUNCTION_TOOL],
        )
        assert outcome is ValidationOutcome.UNKNOWN_TOOL

    def test_invalid_json_args(self):
        outcome = classify_completed_tool_call(
            "get_weather",
            "{not-json",
            [FUNCTION_TOOL],
        )
        assert outcome is ValidationOutcome.INVALID_ARGS

    def test_schema_invalid_args(self):
        outcome = classify_completed_tool_call(
            "get_weather",
            "{}",
            [FUNCTION_TOOL],
        )
        assert outcome is ValidationOutcome.INVALID_ARGS

    def test_missing_parameters_defaults_to_empty_object_schema(self):
        valid = classify_completed_tool_call("no_params", "{}", [NO_PARAMS_TOOL])
        assert valid is ValidationOutcome.VALID

        extra = classify_completed_tool_call(
            "no_params",
            '{"x": 1}',
            [NO_PARAMS_TOOL],
        )
        assert extra is ValidationOutcome.VALID

        not_object = classify_completed_tool_call(
            "no_params",
            "[]",
            [NO_PARAMS_TOOL],
        )
        assert not_object is ValidationOutcome.INVALID_ARGS

    def test_invalid_schema_maps_to_invalid_args(self):
        bad_schema_tool = ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="bad_schema",
                parameters={"type": "not-a-json-schema-type"},
            ),
        )
        outcome = classify_completed_tool_call(
            "bad_schema",
            "{}",
            [bad_schema_tool],
        )
        assert outcome is ValidationOutcome.INVALID_ARGS

    def test_surprises_map_to_invalid_args(self):
        outcome = classify_completed_tool_call("get_weather", "{}", 123)
        assert outcome is ValidationOutcome.INVALID_ARGS

    def test_never_raises(self):
        outcome = classify_completed_tool_call("x", None, object())  # type: ignore[arg-type]
        assert outcome is ValidationOutcome.INVALID_ARGS


class TestRecordToolCallCompleted:
    def test_pre_registers_zero_series(self):
        init_parser_metrics(model_name=_ZERO_MODEL_NAME)
        samples = list(_completed_samples(model_name=_ZERO_MODEL_NAME))
        combos = {
            (sample.labels["request_type"], sample.labels["validation_outcome"])
            for sample in samples
        }
        expected = {
            (request_type.value, outcome.value)
            for request_type in RequestType
            for outcome in ValidationOutcome
        }
        assert combos == expected
        assert all(sample.value == 0 for sample in samples)

    @pytest.mark.parametrize(
        ("api_request", "request_type"),
        [
            (_chat_request(), RequestType.CHAT_COMPLETIONS),
            (_responses_request(), RequestType.RESPONSES),
            (object(), RequestType.OTHER),
        ],
    )
    def test_increments_matching_labels(self, api_request, request_type):
        init_parser_metrics(model_name=_MODEL_NAME)
        before = _completed_value(
            model_name=_MODEL_NAME,
            request_type=request_type.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        record_tool_call_completed(
            request=api_request,
            outcome=ValidationOutcome.VALID,
        )
        after = _completed_value(
            model_name=_MODEL_NAME,
            request_type=request_type.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        assert after == before + 1

    def test_noop_when_unregistered(self, monkeypatch):
        import vllm.parser.metrics as metrics

        monkeypatch.setattr(metrics, "_tool_calls_completed", None)
        record_tool_call_completed(
            request=_chat_request(),
            outcome=ValidationOutcome.VALID,
        )
