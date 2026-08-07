# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import build_response_output_items
from vllm.exceptions import VLLMValidationError
from vllm.parser.kimi_k3 import KimiK3Parser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.kimi_k3_reasoning_parser import KimiK3ReasoningParser
from vllm.tool_parsers.kimi_k3_tool_parser import KimiK3ToolParser

OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"
THINK_OPEN = f"{OPEN}think{SEP}"
THINK_CLOSE = f"{CLOSE}think{SEP}"
RESPONSE_CLOSE = f"{CLOSE}response{SEP}"


class DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == THINK_OPEN:
            return [1, 2, 3]
        if text == THINK_CLOSE:
            return [4, 2, 3]
        return [ord(ch) for ch in text]


class KimiK3DelegatingParser(KimiK3Parser):
    reasoning_parser_cls = KimiK3ReasoningParser
    tool_parser_cls = KimiK3ToolParser


def test_parser_manager_selects_kimi_k3_parser():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="kimi_k3",
        reasoning_parser_name="kimi_k3",
        enable_auto_tools=True,
    )

    assert parser_cls is not None
    assert issubclass(parser_cls, KimiK3Parser)
    assert parser_cls.reasoning_parser_cls is KimiK3ReasoningParser
    assert parser_cls.tool_parser_cls is KimiK3ToolParser


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )


def _named_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "calc"}},
    )


def _responses_request(*, tool_choice="auto") -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "Call the calc tool.",
            "tools": [
                {
                    "type": "function",
                    "name": "calc",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": tool_choice,
        }
    )


def _arg(key: str, typ: str, value: str) -> str:
    return f'{OPEN}argument key="{key}" type="{typ}"{SEP}{value}{CLOSE}argument{SEP}'


def _call(tool: str, index: int, *args: str) -> str:
    body = "".join(args)
    return f'{OPEN}call tool="{tool}" index="{index}"{SEP}{body}{CLOSE}call{SEP}'


def _response(content: str) -> str:
    return f"{OPEN}response{SEP}{content}{RESPONSE_CLOSE}"


def _tools(*calls: str) -> str:
    return f"{OPEN}tools{SEP}{''.join(calls)}{CLOSE}tools{SEP}"


def test_extract_tool_calls_with_response_and_typed_arguments():
    parser = KimiK3ToolParser(DummyTokenizer())

    output = _response("answer") + _tools(
        _call(
            "calc",
            1,
            _arg("x", "number", "1"),
            _arg("flag", "boolean", "true"),
            _arg("text", "string", "raw"),
        )
    )
    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.content == "answer"
    assert len(extracted.tool_calls) == 1
    tool_call = extracted.tool_calls[0]
    assert tool_call.function.name == "calc"
    assert json.loads(tool_call.function.arguments) == {
        "x": 1,
        "flag": True,
        "text": "raw",
    }


def test_delegating_parser_preserves_tool_calls_after_reasoning():
    parser = KimiK3DelegatingParser(DummyTokenizer())
    output = (
        f"{THINK_OPEN}step{THINK_CLOSE}"
        + _response("answer")
        + _tools(_call("calc", 1, _arg("x", "number", "1")))
    )

    reasoning, content, tool_calls = parser.parse(
        output,
        _request(),
        enable_auto_tools=True,
    )

    assert reasoning == "step"
    assert content == "answer"
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "calc"
    assert json.loads(tool_calls[0].arguments) == {"x": 1}


def test_delegating_parser_required_tool_choice_uses_xtml_parser():
    parser = KimiK3DelegatingParser(DummyTokenizer())
    request = _request().model_copy(update={"tool_choice": "required"})
    output = (
        f"{THINK_OPEN}step{THINK_CLOSE}"
        + _response("")
        + _tools(_call("calc", 1, _arg("x", "number", "1")))
    )

    reasoning, content, tool_calls = parser.parse(
        output,
        request,
        enable_auto_tools=True,
    )

    assert reasoning == "step"
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "calc"
    assert json.loads(tool_calls[0].arguments) == {"x": 1}


def test_delegating_parser_named_tool_choice_uses_xtml_parser():
    parser = KimiK3DelegatingParser(DummyTokenizer())
    output = (
        f"{THINK_OPEN}step{THINK_CLOSE}"
        + _response("")
        + _tools(_call("calc", 1, _arg("x", "number", "1")))
    )

    reasoning, content, tool_calls = parser.parse(
        output,
        _named_request(),
        enable_auto_tools=True,
    )

    assert reasoning == "step"
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "calc"
    assert json.loads(tool_calls[0].arguments) == {"x": 1}


def test_delegating_parser_auto_no_call_strips_consumed_response_prefix():
    parser = KimiK3DelegatingParser(
        DummyTokenizer(), chat_template_kwargs={"thinking": False}
    )
    request = _request().model_copy(
        update={"chat_template_kwargs": {"thinking": False}}
    )

    reasoning, content, tool_calls = parser.parse(
        f"answer{RESPONSE_CLOSE}",
        request,
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content == "answer"
    assert tool_calls is None


def test_delegating_parser_required_call_strips_consumed_response_prefix():
    parser = KimiK3DelegatingParser(
        DummyTokenizer(), chat_template_kwargs={"thinking": False}
    )
    request = _request().model_copy(
        update={
            "tool_choice": "required",
            "chat_template_kwargs": {"thinking": False},
        }
    )
    output = RESPONSE_CLOSE + _tools(_call("calc", 1, _arg("x", "number", "1")))

    reasoning, content, tool_calls = parser.parse(
        output,
        request,
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "calc"
    assert json.loads(tool_calls[0].arguments) == {"x": 1}


def test_delegating_parser_truncated_tools_do_not_leak_xtml():
    parser = KimiK3DelegatingParser(
        DummyTokenizer(), chat_template_kwargs={"thinking": False}
    )
    request = _request().model_copy(
        update={
            "tool_choice": "required",
            "chat_template_kwargs": {"thinking": False},
        }
    )

    reasoning, content, tool_calls = parser.parse(
        (f'{RESPONSE_CLOSE}{OPEN}tools{SEP}{OPEN}call tool="calc" index="1"'),
        request,
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content is None
    assert tool_calls is None


def test_extract_tool_calls_unescapes_attributes():
    parser = KimiK3ToolParser(DummyTokenizer())

    output = _tools(_call("a&amp;b&quot;c", 1, _arg("k&amp;q", "string", "v")))
    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.tool_calls[0].function.name == 'a&b"c'
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"k&q": "v"}


def test_extract_tool_calls_allows_less_than_in_attributes():
    parser = KimiK3ToolParser(DummyTokenizer())

    output = _tools(_call("calc<beta", 1, _arg("foo<bar", "string", "raw")))
    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.tool_calls[0].function.name == "calc<beta"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"foo<bar": "raw"}


def test_extract_content_from_whitespace_degraded_markers():
    parser = KimiK3ToolParser(DummyTokenizer())

    extracted = parser.extract_tool_calls(
        f"{OPEN} response {SEP}answer{CLOSE} response {SEP}",
        _request(),
    )

    assert extracted.tools_called is False
    assert extracted.content == "answer"


def test_streaming_split_markers_do_not_leak():
    parser = KimiK3ToolParser(DummyTokenizer())
    request = _request()
    previous_text = ""
    previous_ids: list[int] = []
    messages: list[DeltaMessage] = []
    chunks = [
        OPEN,
        "response",
        f"{SEP}Hi",
        OPEN,
        "tools",
        SEP,
        f'{OPEN}call tool="calc" index="1"{SEP}',
        _arg("x", "number", "1"),
        f"{CLOSE}call",
        SEP,
    ]

    for i, chunk in enumerate(chunks, start=1):
        current_text = previous_text + chunk
        current_ids = previous_ids + [i]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_ids,
            current_token_ids=current_ids,
            delta_token_ids=[i],
            request=request,
        )
        if delta is not None:
            messages.append(delta)
        previous_text = current_text
        previous_ids = current_ids

    content = "".join(message.content or "" for message in messages)
    tool_deltas = [
        tool_call for message in messages for tool_call in (message.tool_calls or [])
    ]

    assert content == "Hi"
    assert OPEN not in content
    assert SEP not in content
    assert len(tool_deltas) == 1
    assert tool_deltas[0].function.name == "calc"
    assert json.loads(tool_deltas[0].function.arguments) == {"x": 1}


def test_tool_call_ids_are_unique_across_messages():
    output = _tools(_call("calc", 1))

    first = KimiK3ToolParser(DummyTokenizer()).extract_tool_calls(output, _request())
    second = KimiK3ToolParser(DummyTokenizer()).extract_tool_calls(output, _request())

    assert first.tool_calls[0].id != second.tool_calls[0].id


def test_streaming_consumed_response_prefix_no_call_keeps_content():
    parser = KimiK3ToolParser(DummyTokenizer())
    request = _request()
    previous_text = ""
    previous_ids: list[int] = []
    messages: list[DeltaMessage] = []
    chunks = ["O", "K", CLOSE, f"response{SEP}"]

    for i, chunk in enumerate(chunks, start=1):
        current_text = previous_text + chunk
        current_ids = previous_ids + [i]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_ids,
            current_token_ids=current_ids,
            delta_token_ids=[i],
            request=request,
        )
        if delta is not None:
            messages.append(delta)
        previous_text = current_text
        previous_ids = current_ids

    assert "".join(message.content or "" for message in messages) == "OK"
    assert all(CLOSE not in (message.content or "") for message in messages)


def test_delegating_parser_tool_choice_none_strips_xtml_and_suppresses_calls():
    parser = KimiK3DelegatingParser(
        DummyTokenizer(), chat_template_kwargs={"thinking": False}
    )
    request = _request().model_copy(
        update={
            "tool_choice": "none",
            "chat_template_kwargs": {"thinking": False},
        }
    )
    messages: list[DeltaMessage] = []
    chunks = [
        OPEN,
        "response",
        f"{SEP}answer",
        RESPONSE_CLOSE,
        _tools(_call("calc", 1, _arg("x", "number", "1"))),
    ]

    for index, chunk in enumerate(chunks, start=1):
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[index],
            request=request,
            prompt_token_ids=[1],
            finished=index == len(chunks),
        )
        if delta is not None:
            messages.append(delta)

    content = "".join(message.content or "" for message in messages)
    assert content == "answer"
    assert OPEN not in content
    assert CLOSE not in content
    assert SEP not in content
    assert all(not message.tool_calls for message in messages)


def test_adjust_request_keeps_xtml_markers_contiguous():
    parser = KimiK3ToolParser(DummyTokenizer())
    request = _request()

    adjusted = parser.adjust_request(request)

    assert adjusted.skip_special_tokens is False
    if hasattr(adjusted, "spaces_between_special_tokens"):
        assert adjusted.spaces_between_special_tokens is False
    assert KimiK3ToolParser.supports_required_and_named is False


def test_adjust_request_required_uses_xtml_parser_not_json_guidance():
    parser = KimiK3ToolParser(DummyTokenizer())
    request = _request().model_copy(update={"tool_choice": "required"})

    adjusted = parser.adjust_request(request)

    assert adjusted.structured_outputs is None
    assert adjusted.skip_special_tokens is False
    if hasattr(adjusted, "spaces_between_special_tokens"):
        assert adjusted.spaces_between_special_tokens is False


@pytest.mark.parametrize(
    "tool_request",
    [
        _named_request(),
        _responses_request(
            tool_choice={"type": "function", "name": "calc"},
        ),
    ],
)
def test_adjust_request_rejects_named_tool_choice(tool_request):
    parser = KimiK3ToolParser(DummyTokenizer())

    with pytest.raises(VLLMValidationError) as exc_info:
        parser.adjust_request(tool_request)

    assert exc_info.value.parameter == "tool_choice"
    assert "requires strict tool calling" in str(exc_info.value)


def test_responses_chat_params_carries_tool_choice_metadata():
    request = _responses_request(tool_choice="required")

    chat_params = request.build_chat_params(
        default_template=None,
        default_template_content_format="auto",
    )

    assert chat_params.tool_choice == "required"


def test_responses_chat_params_keeps_template_tool_choice_when_api_auto():
    request = _responses_request().model_copy(
        update={"chat_template_kwargs": {"tool_choice": "required"}}
    )

    chat_params = request.build_chat_params(
        default_template=None,
        default_template_content_format="auto",
    )

    assert chat_params.chat_template_kwargs["tool_choice"] == "required"
    assert chat_params.tool_choice == "auto"


def test_responses_required_tool_choice_uses_xtml_parser():
    parser = KimiK3DelegatingParser(
        DummyTokenizer(), chat_template_kwargs={"thinking": False}
    )
    request = _responses_request(tool_choice="required").model_copy(
        update={"chat_template_kwargs": {"thinking": False}}
    )
    output = RESPONSE_CLOSE + _tools(_call("calc", 1, _arg("x", "number", "1")))

    reasoning, content, tool_calls = parser.parse(
        output, request, enable_auto_tools=True, model_output_token_ids=[]
    )
    response_outputs = build_response_output_items(
        reasoning=reasoning,
        content=content,
        tool_calls=tool_calls,
        tools=request.tools,
    )

    assert len(response_outputs) == 1
    tool_call = response_outputs[0]
    assert tool_call.type == "function_call"
    assert tool_call.name == "calc"
    assert json.loads(tool_call.arguments) == {"x": 1}


def test_responses_named_tool_choice_uses_xtml_parser():
    parser = KimiK3DelegatingParser(
        DummyTokenizer(), chat_template_kwargs={"thinking": False}
    )
    request = _responses_request(
        tool_choice={"type": "function", "name": "calc"}
    ).model_copy(update={"chat_template_kwargs": {"thinking": False}})
    output = RESPONSE_CLOSE + _tools(_call("calc", 1, _arg("x", "number", "1")))

    reasoning, content, tool_calls = parser.parse(
        output, request, enable_auto_tools=True, model_output_token_ids=[]
    )
    response_outputs = build_response_output_items(
        reasoning=reasoning,
        content=content,
        tool_calls=tool_calls,
        tools=request.tools,
    )

    assert len(response_outputs) == 1
    tool_call = response_outputs[0]
    assert tool_call.type == "function_call"
    assert tool_call.name == "calc"
    assert json.loads(tool_call.arguments) == {"x": 1}


def test_chat_params_carries_tool_choice_metadata():
    request = _request().model_copy(update={"tool_choice": "required"})

    chat_params = request.build_chat_params(
        default_template=None,
        default_template_content_format="auto",
    )

    assert chat_params.tool_choice == "required"


def test_chat_params_carries_response_format_metadata():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        response_format={"type": "json_object"},
    )

    chat_params = request.build_chat_params(
        default_template=None,
        default_template_content_format="auto",
    )

    assert chat_params.response_format is request.response_format
    assert chat_params.tool_choice is None


def test_chat_params_keeps_template_tool_choice_when_api_auto():
    request = _request().model_copy(
        update={
            "tool_choice": "auto",
            "chat_template_kwargs": {"tool_choice": "required"},
        }
    )

    chat_params = request.build_chat_params(
        default_template=None,
        default_template_content_format="auto",
    )

    assert chat_params.chat_template_kwargs["tool_choice"] == "required"
    assert chat_params.tool_choice == "auto"


# A tools marker can appear inside the response channel (the model typing it, or
# an echo from the prompt). The real call must still be found, and the marker
# must not surface as user-visible content.
def test_extract_skips_marker_in_response_body_and_finds_real_call():
    parser = KimiK3ToolParser(DummyTokenizer())
    output = (
        "<|open|>response<|sep|>Let me check.<|open|>tools<|sep|><|close|>tools<|sep|>"
        "<|close|>response<|sep|>"
        '<|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|>'
        "<|close|>call<|sep|><|close|>tools<|sep|>"
    )

    result = parser.extract_tool_calls(output, _request())

    assert result.tools_called
    assert [tc.function.name for tc in result.tool_calls] == ["get_weather"]
    assert result.content == "Let me check."


def test_extract_keeps_content_that_merely_looks_like_a_marker_start():
    # A trailing "<" is content, not a truncated marker: an end-to-end probe
    # showed an answer of "3 <" losing its last character.
    parser = KimiK3ToolParser(DummyTokenizer())

    for body in ("3 <", "a < b", "compare 5 <"):
        result = parser.extract_tool_calls(
            f"<|open|>response<|sep|>{body}<|close|>response<|sep|>", _request()
        )
        assert result.content == body


def test_extract_does_not_leak_a_dangling_marker_into_content():
    # Generation stopped mid-marker inside the response channel: the partial
    # marker is held back, like the streaming path does.
    parser = KimiK3ToolParser(DummyTokenizer())

    result = parser.extract_tool_calls(
        "<|open|>response<|sep|>Let me check.<|open|>tools", _request()
    )

    assert not result.tools_called
    assert result.content == "Let me check."
