# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ParserEngine recording of vllm:tool_calls_completed_total."""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.kimi_k2 import KimiK2Parser
from vllm.parser.metrics import (
    RequestType,
    ValidationOutcome,
    init_parser_metrics,
)

pytestmark = pytest.mark.cpu_test

_VOCAB: dict[str, int] = {
    "<tool_call>": 202,
    "</tool_call>": 203,
}

_WEATHER_PARAMS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}

_WEATHER_TOOL = ChatCompletionToolsParam(
    type="function",
    function=FunctionDefinition(name="get_weather", parameters=_WEATHER_PARAMS),
)

_HERMES_VALID = (
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Dallas"}}</tool_call>'
)
_HERMES_INVALID_ARGS = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
_HERMES_UNKNOWN = (
    '<tool_call>{"name": "not_a_tool", "arguments": {"city": "Dallas"}}</tool_call>'
)
_HERMES_BAD_JSON = '<tool_call>{"name": "get_weather", "arguments": {</tool_call>'


def _hermes_config(**kwargs) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="hermes_completed_metrics",
        terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        token_id_terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        **kwargs,
    )


def _make_engine(
    *,
    tools: list | None = None,
    validate_tool_names: bool = False,
) -> ParserEngine:
    return ParserEngine(
        make_mock_tokenizer(_VOCAB),
        tools=tools,
        parser_engine_config=_hermes_config(validate_tool_names=validate_tool_names),
    )


def _chat_request(tools: list | None = None) -> ChatCompletionRequest:
    body: dict = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    if tools is not None:
        body["tools"] = tools
    return ChatCompletionRequest.model_validate(body)


def _responses_request() -> ResponsesRequest:
    return ResponsesRequest(input="hi", tools=[])


def _completed_value(
    *,
    model_name: str,
    request_type: str,
    validation_outcome: str,
) -> float:
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if (
                sample.name == "vllm:tool_calls_completed_total"
                and sample.labels.get("model_name") == model_name
                and sample.labels.get("request_type") == request_type
                and sample.labels.get("validation_outcome") == validation_outcome
            ):
                return sample.value
    raise AssertionError(
        f"missing series model_name={model_name} "
        f"request_type={request_type} validation_outcome={validation_outcome}"
    )


def _init(model_name: str) -> None:
    init_parser_metrics(model_name=model_name)


class TestParserEngineCompletedMetrics:
    def test_parse_valid_chat_completions(self):
        model = "pe-completed-valid-chat"
        _init(model)
        engine = _make_engine(tools=[_WEATHER_TOOL])
        before = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        _, _, tool_calls = engine.parse(_HERMES_VALID, _chat_request([_WEATHER_TOOL]))
        assert tool_calls is not None
        assert tool_calls[0].name == "get_weather"
        assert '"city": "Dallas"' in tool_calls[0].arguments
        after = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        assert after == before + 1

    def test_parse_delta_uses_same_counter_as_parse(self):
        model = "pe-completed-stream-share"
        _init(model)
        request = _chat_request([_WEATHER_TOOL])
        engine = _make_engine(tools=[_WEATHER_TOOL])
        engine.initialize_streaming()
        before = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        engine.parse_delta(_HERMES_VALID, [], request, finished=True)
        after = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        assert after == before + 1

    def test_responses_request_type(self):
        model = "pe-completed-responses"
        _init(model)
        engine = _make_engine(tools=[_WEATHER_TOOL])
        before = _completed_value(
            model_name=model,
            request_type=RequestType.RESPONSES.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        engine.parse(_HERMES_VALID, _responses_request())
        after = _completed_value(
            model_name=model,
            request_type=RequestType.RESPONSES.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        assert after == before + 1

    def test_unknown_tool_when_name_not_in_catalog(self):
        model = "pe-completed-unknown"
        _init(model)
        engine = _make_engine(tools=[_WEATHER_TOOL])
        before = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.UNKNOWN_TOOL.value,
        )
        engine.parse(_HERMES_UNKNOWN, _chat_request([_WEATHER_TOOL]))
        after = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.UNKNOWN_TOOL.value,
        )
        assert after == before + 1

    def test_invalid_args_when_schema_fails(self):
        model = "pe-completed-invalid-args"
        _init(model)
        engine = _make_engine(tools=[_WEATHER_TOOL])
        before = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.INVALID_ARGS.value,
        )
        engine.parse(_HERMES_INVALID_ARGS, _chat_request([_WEATHER_TOOL]))
        after = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.INVALID_ARGS.value,
        )
        assert after == before + 1

    def test_classifies_client_pair_not_raw_slot_args(self):
        """Raw Hermes envelope would fail additionalProperties=False."""
        model = "pe-completed-client-pair"
        _init(model)
        engine = _make_engine(tools=[_WEATHER_TOOL])
        _, _, tool_calls = engine.parse(_HERMES_VALID, _chat_request([_WEATHER_TOOL]))
        assert tool_calls is not None
        slot = engine._tool_slots[0]
        assert slot.completed is True
        assert slot.function_call == (tool_calls[0].name, tool_calls[0].arguments)
        assert slot.args != tool_calls[0].arguments
        assert "get_weather" in slot.args

    def test_rejected_name_is_not_a_completed_call(self):
        model = "pe-completed-rejected"
        _init(model)
        engine = _make_engine(tools=[_WEATHER_TOOL], validate_tool_names=True)
        before_unknown = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.UNKNOWN_TOOL.value,
        )
        before_valid = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        _, _, tool_calls = engine.parse(_HERMES_UNKNOWN, _chat_request([_WEATHER_TOOL]))
        assert tool_calls is None
        assert (
            _completed_value(
                model_name=model,
                request_type=RequestType.CHAT_COMPLETIONS.value,
                validation_outcome=ValidationOutcome.UNKNOWN_TOOL.value,
            )
            == before_unknown
        )
        assert (
            _completed_value(
                model_name=model,
                request_type=RequestType.CHAT_COMPLETIONS.value,
                validation_outcome=ValidationOutcome.VALID.value,
            )
            == before_valid
        )

    def test_content_only_does_not_increment(self):
        model = "pe-completed-content-only"
        _init(model)
        engine = _make_engine(tools=[_WEATHER_TOOL])
        before = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        engine.parse("Hello world", _chat_request([_WEATHER_TOOL]))
        after = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        assert after == before

    def test_handle_tool_end_is_idempotent(self):
        model = "pe-completed-idempotent"
        _init(model)
        engine = _make_engine(tools=[_WEATHER_TOOL])
        engine.parse(_HERMES_VALID, _chat_request([_WEATHER_TOOL]))
        mid = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        engine._handle_tool_end(
            SemanticEvent(EventType.TOOL_CALL_END, tool_index=0),
            [],
        )
        after = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        assert after == mid

    def test_bad_json_args_are_invalid_args(self):
        model = "pe-completed-bad-json"
        _init(model)
        engine = _make_engine(tools=[_WEATHER_TOOL])
        before = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.INVALID_ARGS.value,
        )
        engine.parse(_HERMES_BAD_JSON, _chat_request([_WEATHER_TOOL]))
        after = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.INVALID_ARGS.value,
        )
        assert after == before + 1

    def test_extract_tool_calls_from_content_records_chat_type(self):
        model = "pe-completed-from-content"
        _init(model)
        engine = _make_engine(tools=[_WEATHER_TOOL])
        before = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        result = engine.extract_tool_calls_from_content(
            _HERMES_VALID, _chat_request([_WEATHER_TOOL])
        )
        assert result.tools_called is True
        after = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        assert after == before + 1


class TestKimiDoubleToolCallEnd:
    """Kimi stays in TOOL_PREAMBLE after a well-formed call; finish() emits
    TOOL_CALL_END again. Recording must stay at one increment.
    """

    def test_records_once_on_kimi_section_close(self):
        model = "pe-completed-kimi-double-end"
        _init(model)
        vocab = {
            "<think>": 1,
            "</think>": 2,
            "<|tool_calls_section_begin|>": 3,
            "<|tool_calls_section_end|>": 4,
            "<|tool_call_begin|>": 5,
            "<|tool_call_end|>": 6,
            "<|tool_call_argument_begin|>": 7,
        }
        parser = KimiK2Parser(
            make_mock_tokenizer(vocab),
            tools=[_WEATHER_TOOL],
            chat_template_kwargs={"thinking": False},
        )
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0"
            '<|tool_call_argument_begin|>{"city": "Dallas"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        before = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        _, _, tool_calls = parser.parse(text, _chat_request([_WEATHER_TOOL]))
        assert tool_calls is not None
        assert tool_calls[0].name == "get_weather"
        after = _completed_value(
            model_name=model,
            request_type=RequestType.CHAT_COMPLETIONS.value,
            validation_outcome=ValidationOutcome.VALID.value,
        )
        assert after == before + 1
        assert parser._tool_slots[0].completed is True
