# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import json
import random
import statistics
import time
from typing import Any, cast

import pytest

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import build_response_output_items
from vllm.exceptions import VLLMValidationError
from vllm.parser.kimi_k3 import KimiK3Parser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.kimi_k3_reasoning_parser import KimiK3ReasoningParser
from vllm.tokenizers.protocol import TokenizerLike
from vllm.tool_parsers.kimi_k3_tool_parser import KimiK3ToolParser

OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"
THINK_OPEN = f"{OPEN}think{SEP}"
THINK_CLOSE = f"{CLOSE}think{SEP}"
RESPONSE_OPEN = f"{OPEN}response{SEP}"
RESPONSE_CLOSE = f"{CLOSE}response{SEP}"
TOOLS_OPEN = f"{OPEN}tools{SEP}"
TOOLS_CLOSE = f"{CLOSE}tools{SEP}"
MESSAGE_CLOSE = f"{CLOSE}message{SEP}"


class DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == THINK_OPEN:
            return [1, 2, 3]
        if text == THINK_CLOSE:
            return [4, 2, 3]
        return [ord(ch) for ch in text]


def _dummy_tokenizer() -> TokenizerLike:
    return cast(TokenizerLike, DummyTokenizer())


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
    parser = KimiK3ToolParser(_dummy_tokenizer())

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
    parser = KimiK3DelegatingParser(_dummy_tokenizer())
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
    parser = KimiK3DelegatingParser(_dummy_tokenizer())
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
    parser = KimiK3DelegatingParser(_dummy_tokenizer())
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
        _dummy_tokenizer(), chat_template_kwargs={"thinking": False}
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
        _dummy_tokenizer(), chat_template_kwargs={"thinking": False}
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
        _dummy_tokenizer(), chat_template_kwargs={"thinking": False}
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
    parser = KimiK3ToolParser(_dummy_tokenizer())

    output = _tools(_call("a&amp;b&quot;c", 1, _arg("k&amp;q", "string", "v")))
    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.tool_calls[0].function.name == 'a&b"c'
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"k&q": "v"}


def test_extract_tool_calls_allows_less_than_in_attributes():
    parser = KimiK3ToolParser(_dummy_tokenizer())

    output = _tools(_call("calc<beta", 1, _arg("foo<bar", "string", "raw")))
    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.tool_calls[0].function.name == "calc<beta"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"foo<bar": "raw"}


def test_extract_content_from_whitespace_degraded_markers():
    parser = KimiK3ToolParser(_dummy_tokenizer())

    extracted = parser.extract_tool_calls(
        f"{OPEN} response {SEP}answer{CLOSE} response {SEP}",
        _request(),
    )

    assert extracted.tools_called is False
    assert extracted.content == "answer"


def test_streaming_split_markers_do_not_leak():
    parser = KimiK3ToolParser(_dummy_tokenizer())
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
    assert tool_deltas[0].function is not None
    assert tool_deltas[0].function.name == "calc"
    assert json.loads(tool_deltas[0].function.arguments) == {"x": 1}


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 16, 31, 128, 4096])
@pytest.mark.parametrize(
    "output",
    [
        _response("answer"),
        "answer" + RESPONSE_CLOSE,
        _response("answer") + f'{OPEN}tools{SEP}{OPEN}call tool="calc"{SEP}partial',
        _response("answer")
        + _tools(
            _call("calc", 1, _arg("text", "string", "long value")),
            _call("calc", 2, _arg("x", "number", "2")),
        ),
        _response("answer") + f"{OPEN}{' ' * 32}tools{SEP}" + _call("calc", 1),
        _response("answer")
        + _tools(_call("calc", 1).replace(f"{CLOSE}call", f"{CLOSE}{' ' * 32}call")),
        _response("answer") + _tools(f"{OPEN}call!" + _call("calc", 1)),
        _response("answer")
        + _tools(
            f'{OPEN}call tool="a{CLOSE}call{SEP}b"{SEP}'
            f"{CLOSE}call{SEP}" + _call("calc", 2)
        ),
        _response("answer")
        + _tools(f'{OPEN}call index="1"{SEP}{CLOSE}call{SEP}' + _call("calc", 2)),
    ],
    ids=[
        "response-only",
        "consumed-response-prefix",
        "truncated-call",
        "parallel",
        "spaced-tools",
        "spaced-close",
        "invalid-opener",
        "close-in-attributes",
        "missing-name",
    ],
)
def test_streaming_call_boundaries(output, chunk_size):
    request = _request()
    parser = KimiK3ToolParser(_dummy_tokenizer())
    expected = KimiK3ToolParser(_dummy_tokenizer()).extract_tool_calls(output, request)
    previous_text = ""
    content = []
    calls = []
    for chunk_start in range(0, len(output), chunk_size):
        delta_text = output[chunk_start : chunk_start + chunk_size]
        current_text = previous_text + delta_text
        message = parser.extract_tool_calls_streaming(
            previous_text, current_text, delta_text, [], [], [], request
        )
        if message is not None:
            content.append(message.content or "")
            calls.extend(message.tool_calls or [])
        previous_text = current_text

    assert "".join(content) == "answer"
    assert [call.index for call in calls] == list(range(len(expected.tool_calls)))
    functions = []
    for call in calls:
        assert call.function is not None
        functions.append(call.function.model_dump())
    assert functions == [
        {"name": call.function.name, "arguments": call.function.arguments}
        for call in expected.tool_calls
    ]


def test_streaming_does_not_redecode_completed_calls(monkeypatch):
    request = _request()
    call = _call("calc", 1, _arg("text", "string", "x" * 4096))
    output = _tools(call) + "trailing output"
    parser = KimiK3ToolParser(_dummy_tokenizer())
    decode_count = 0
    original_decode_call = parser._decode_call

    def counted_decode_call(attrs: str, body: str):
        nonlocal decode_count
        decode_count += 1
        return original_decode_call(attrs, body)

    monkeypatch.setattr(parser, "_decode_call", counted_decode_call)

    previous_text = ""
    for chunk_start in range(0, len(output), 11):
        delta_text = output[chunk_start : chunk_start + 11]
        current_text = previous_text + delta_text
        parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        previous_text = current_text

    assert decode_count == 1


def test_tool_call_ids_are_unique_across_messages():
    output = _tools(_call("calc", 1))

    first = KimiK3ToolParser(_dummy_tokenizer()).extract_tool_calls(output, _request())
    second = KimiK3ToolParser(_dummy_tokenizer()).extract_tool_calls(output, _request())

    assert first.tool_calls[0].id != second.tool_calls[0].id


def test_streaming_consumed_response_prefix_no_call_keeps_content():
    parser = KimiK3ToolParser(_dummy_tokenizer())
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
        _dummy_tokenizer(), chat_template_kwargs={"thinking": False}
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


@pytest.mark.parametrize(
    "req",
    [
        _request(),
        _responses_request(),
    ],
)
def test_adjust_request_keeps_xtml_markers_contiguous(req):
    parser = KimiK3ToolParser(_dummy_tokenizer())

    adjusted = parser.adjust_request(req)

    assert adjusted.skip_special_tokens is False
    assert adjusted.spaces_between_special_tokens is False
    assert KimiK3ToolParser.supports_required_and_named is False


def test_adjust_request_required_uses_xtml_parser_not_json_guidance():
    parser = KimiK3ToolParser(_dummy_tokenizer())
    request = _request().model_copy(update={"tool_choice": "required"})

    adjusted = parser.adjust_request(request)

    assert adjusted.structured_outputs is None
    assert adjusted.skip_special_tokens is False
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
    parser = KimiK3ToolParser(_dummy_tokenizer())

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
        _dummy_tokenizer(), chat_template_kwargs={"thinking": False}
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
        _dummy_tokenizer(), chat_template_kwargs={"thinking": False}
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


# The streaming extraction used to search the whole accumulated text again on
# every delta. The reference below is that implementation; the incremental one
# has to reproduce it call for call.
def _reference_partial_tag_overlap(text: str, tag: str) -> int:
    max_len = min(len(text), len(tag) - 1)
    for n in range(max_len, 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class _FullRescanToolParser(KimiK3ToolParser):
    def _extract_response_content(self, current_text):  # type: ignore[override]
        m_open = self._response_open_re.search(current_text)
        body_start = m_open.end() if m_open is not None else 0
        m_tools = self._tools_open_re.search(current_text, body_start)
        m_rclose = self._response_close_re.search(current_text, body_start)
        tools_start = m_tools.start() if m_tools else -1
        response_end = m_rclose.start() if m_rclose else -1

        candidates = [i for i in (tools_start, response_end) if i != -1]
        if candidates:
            sendable_idx = min(candidates)
        else:
            overlap = max(
                _reference_partial_tag_overlap(current_text, self.response_open),
                _reference_partial_tag_overlap(current_text, self.response_close),
                _reference_partial_tag_overlap(current_text, self.tools_open),
            )
            sendable_idx = len(current_text) - overlap

        if sendable_idx <= body_start:
            return None
        if self._sent_content_idx < body_start:
            self._sent_content_idx = body_start
        if sendable_idx <= self._sent_content_idx:
            return None

        content = current_text[self._sent_content_idx : sendable_idx]
        self._sent_content_idx = sendable_idx
        return content or None

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        content = self._extract_response_content(current_text)
        m_tools = self._tools_open_re.search(current_text)
        if m_tools is None:
            return DeltaMessage(content=content) if content else None

        self._stream_call_parse_idx = max(self._stream_call_parse_idx, m_tools.end())
        self._stream_call_close_idx = max(self._stream_call_close_idx, m_tools.end())
        new_calls = []
        while True:
            m_close = self._call_close_re.search(
                current_text, self._stream_call_close_idx, partial=True
            )
            if m_close.partial:
                self._stream_call_close_idx = m_close.start()
                break
            self._stream_call_close_idx = m_close.end()
            for call_match in self._call_re.finditer(
                current_text, self._stream_call_parse_idx, m_close.end()
            ):
                tool_call = self._decode_call(call_match["attrs"], call_match["body"])
                if tool_call is not None:
                    new_calls.append(tool_call)
                self._stream_call_parse_idx = call_match.end()

        if not new_calls:
            return DeltaMessage(content=content) if content else None

        deltas = [
            DeltaToolCall(
                index=self._sent_tool_call_count + i,
                id=tc.id,
                type="function",
                function=DeltaFunctionCall(
                    name=tc.function.name, arguments=tc.function.arguments
                ).model_dump(exclude_none=True),
            )
            for i, tc in enumerate(new_calls)
        ]
        self._sent_tool_call_count += len(new_calls)
        return DeltaMessage(content=content, tool_calls=deltas)


class _ReferenceDelegatingParser(KimiK3Parser):
    reasoning_parser_cls = KimiK3ReasoningParser
    tool_parser_cls = _FullRescanToolParser


# Channel markers, their fragments, whitespace-tolerant spellings and near misses:
# the streams below keep splitting, completing and abandoning markers, and put
# them in orders a model would not produce.
_STREAM_FRAGMENTS = [
    "Hi",
    " there",
    " ",
    "\n",
    "x y",
    "<",
    "|",
    "<|",
    "|>",
    "open",
    "close",
    "tools",
    "response",
    "call",
    "message",
    RESPONSE_OPEN,
    RESPONSE_OPEN,
    RESPONSE_CLOSE,
    RESPONSE_CLOSE,
    TOOLS_OPEN,
    TOOLS_OPEN,
    TOOLS_CLOSE,
    MESSAGE_CLOSE,
    THINK_CLOSE,
    f"{OPEN} response {SEP}",
    f"{CLOSE}\nresponse\n{SEP}",
    f"{OPEN} tools {SEP}",
    f'{OPEN}call tool="f" index="1"{SEP}',
    f"{CLOSE}call{SEP}",
    _call("lookup", 1, _arg("q", "string", "abc"), _arg("n", "number", "2")),
    _call("lookup", 1, _arg("q", "string", "abc"), _arg("n", "number", "2")),
    _arg("k", "string", "v"),
    "<|clo",
    "se|>response<|se",
    "p|>",
    "<|open|>resp",
    "<|open|>to",
    "ols<|sep",
]


def _chop(rnd: random.Random, text: str) -> list[str]:
    """`text` cut into deltas of 0-34 characters (empty deltas included), so a
    marker is cut at arbitrary points."""
    chunks, i = [], 0
    while i < len(text):
        size = rnd.choice([0, 1, 1, 2, 3, 5, 8, 13, 34])
        chunks.append(text[i : i + size])
        i += size
    return chunks


def _random_chunks(rnd: random.Random) -> list[str]:
    text = "".join(rnd.choice(_STREAM_FRAGMENTS) for _ in range(rnd.randrange(1, 18)))
    return _chop(rnd, text)


def _structured_chunks(rnd: random.Random) -> list[str]:
    """A well-formed response and tools channel, with a few stray fragments,
    cut at arbitrary points: what a model emits, plus what a model might."""
    parts = []
    if rnd.random() < 0.8:
        opening = rnd.choice([RESPONSE_OPEN, RESPONSE_OPEN, ""])
        parts.append(opening + "answer text " * rnd.randrange(0, 6) + RESPONSE_CLOSE)
    if rnd.random() < 0.85:
        calls = [
            _call(
                rnd.choice(["lookup", "write"]),
                1,
                _arg("q", "string", "abc" * rnd.randrange(4)),
                _arg("n", "number", "2"),
            )
            for _ in range(rnd.randrange(0, 4))
        ]
        parts.append(_tools(*calls))
    parts.append(MESSAGE_CLOSE)
    for _ in range(rnd.randrange(0, 3)):
        parts.insert(rnd.randrange(len(parts) + 1), rnd.choice(_STREAM_FRAGMENTS))
    return _chop(rnd, "".join(parts))


def _normalized(message: DeltaMessage | None):
    """The message without the random ids of its tool calls."""
    if message is None:
        return None
    dumped = message.model_dump()
    for tool_call in dumped.get("tool_calls") or []:
        tool_call["id"] = None
    return dumped


def _indices(parser: KimiK3ToolParser) -> tuple[int, int, int, int]:
    return (
        parser._sent_content_idx,
        parser._sent_tool_call_count,
        parser._stream_call_close_idx,
        parser._stream_call_parse_idx,
    )


def _stream_in_lockstep(chunks: list[str], *, copy_previous: bool = False) -> None:
    """Feed one stream to the parser and to the reference, call by call."""
    parser = KimiK3ToolParser(_dummy_tokenizer())
    reference = _FullRescanToolParser(_dummy_tokenizer())
    request = _request()
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        # An equal but distinct string: trusted by content, so it must agree too.
        given = (previous + "x")[:-1] if copy_previous and previous else previous
        args: tuple[Any, ...] = (given, current, chunk, [], [], [], request)
        expected = reference.extract_tool_calls_streaming(*args)
        assert _normalized(parser.extract_tool_calls_streaming(*args)) == _normalized(
            expected
        ), (previous, chunk)
        assert _indices(parser) == _indices(reference), (previous, chunk)
        previous = current


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("seed", range(24))
def test_streaming_matches_full_rescan_over_marker_dense_streams(seed):
    rnd = random.Random(seed)
    for _ in range(150):
        _stream_in_lockstep(_random_chunks(rnd))


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("seed", range(12))
def test_streaming_matches_full_rescan_over_structured_streams(seed):
    rnd = random.Random(seed)
    for _ in range(150):
        _stream_in_lockstep(_structured_chunks(rnd))


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("seed", range(4))
def test_streaming_matches_full_rescan_for_equal_but_distinct_previous_text(seed):
    rnd = random.Random(seed)
    for _ in range(100):
        _stream_in_lockstep(_random_chunks(rnd), copy_previous=True)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("seed", range(4))
def test_streaming_matches_full_rescan_when_streams_take_turns_on_one_parser(seed):
    """Two streams alternating on one parser share its sent-content indices, so
    they are not meaningful as output, but the marker positions each call finds
    must not leak from one stream's scan into the other's: the reference, fed
    the same calls, has to answer the same."""
    rnd = random.Random(seed)
    parser = KimiK3ToolParser(_dummy_tokenizer())
    reference = _FullRescanToolParser(_dummy_tokenizer())
    request = _request()

    for _ in range(100):
        streams = [_random_chunks(rnd), _random_chunks(rnd)]
        texts = ["", ""]
        for turn in range(2 * max(len(s) for s in streams)):
            which = turn % 2
            if turn // 2 >= len(streams[which]):
                continue
            chunk = streams[which][turn // 2]
            current = texts[which] + chunk
            args: tuple[Any, ...] = (texts[which], current, chunk, [], [], [], request)
            assert _normalized(
                parser.extract_tool_calls_streaming(*args)
            ) == _normalized(reference.extract_tool_calls_streaming(*args))
            assert _indices(parser) == _indices(reference)
            texts[which] = current


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("seed", range(4))
def test_streaming_matches_full_rescan_for_calls_that_do_not_append(seed):
    rnd = random.Random(seed)
    parser = KimiK3ToolParser(_dummy_tokenizer())
    reference = _FullRescanToolParser(_dummy_tokenizer())
    request = _request()

    def text() -> str:
        return "".join(rnd.choice(_STREAM_FRAGMENTS) for _ in range(rnd.randrange(8)))

    for _ in range(500):
        previous, current, delta = text(), text(), text()
        args: tuple[Any, ...] = (previous, current, delta, [], [], [], request)
        assert _normalized(parser.extract_tool_calls_streaming(*args)) == _normalized(
            reference.extract_tool_calls_streaming(*args)
        ), args
        assert _indices(parser) == _indices(reference), args


_HEAD = f"{RESPONSE_OPEN}Hi there"


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("current", "delta"),
    [
        # A marker sits before the offset the scan of "abc" would resume at.
        (_HEAD, "z" * (len(_HEAD) - 3)),  # right length, does not end with the delta
        (_HEAD, "there"),  # ends with the delta, longer than previous + delta
        (_HEAD, _HEAD),  # ends with the delta, shorter than previous + delta
        (_HEAD, "zz"),  # wrong length, does not end with the delta
        ("abc" + _HEAD, "q" * len(_HEAD)),  # extends previous, no delta suffix
    ],
)
def test_streaming_falls_back_when_a_call_does_not_append_to_the_scanned_text(
    current, delta
):
    """The scan is only valid for the text it was taken on. Handed that text back
    with a `current_text` that is not `previous_text + delta_text`, the call
    must be answered from scratch, as before."""
    parser = KimiK3ToolParser(_dummy_tokenizer())
    reference = _FullRescanToolParser(_dummy_tokenizer())
    request = _request()
    previous = "abc"
    for p in (parser, reference):
        # Two appends, so the scan has resumed and its offsets are tight.
        p.extract_tool_calls_streaming("", "ab", "ab", [], [], [], request)
        p.extract_tool_calls_streaming("ab", previous, "c", [], [], [], request)

    args: tuple[Any, ...] = (previous, current, delta, [], [], [], request)
    assert _normalized(parser.extract_tool_calls_streaming(*args)) == _normalized(
        reference.extract_tool_calls_streaming(*args)
    )
    assert _indices(parser) == _indices(reference)


@pytest.mark.skip_global_cleanup
def test_delegating_parser_streaming_matches_full_rescan():
    """Through `DelegatingParser.parse_delta`, which owns the text it passes
    back as `previous_text`, with the reasoning parser in front."""
    request = _request()
    rnd = random.Random(1234)
    for _ in range(300):
        chunks = (_structured_chunks if rnd.random() < 0.6 else _random_chunks)(rnd)
        # The reasoning parser leaves the think channel on the close marker's
        # token ids, so the first delta carries them; a few streams never close.
        closes = rnd.random() < 0.85
        if closes:
            chunks = [f"{rnd.choice(['', 'think hard'])}{THINK_CLOSE}", *chunks]
        parsers = [
            KimiK3DelegatingParser(_dummy_tokenizer()),
            _ReferenceDelegatingParser(_dummy_tokenizer()),
        ]
        for i, chunk in enumerate(chunks):
            delta_ids = [ord(ch) % 5 + 1 for ch in chunk]
            if closes and i == 0:
                delta_ids = [9, 4, 2, 3]
            messages = [
                _normalized(
                    parser.parse_delta(
                        chunk,
                        delta_ids,
                        request,
                        prompt_token_ids=[7],
                        finished=i == len(chunks) - 1,
                    )
                )
                for parser in parsers
            ]
            assert messages[0] == messages[1], (chunks, i)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("copy_previous", [False, True])
def test_streaming_appends_resume_the_scan_instead_of_rescanning(copy_previous):
    """Deltas that complete no response-open marker are answered from the resumed
    scan, also when the caller hands back an equal copy of the previous text
    rather than the very string it was given."""
    rescans = 0

    class CountingParser(KimiK3ToolParser):
        def _scan_from_start(self, text):
            nonlocal rescans
            rescans += 1
            return super()._scan_from_start(text)

    parser = CountingParser(_dummy_tokenizer())
    request = _request()
    stream = (
        "answer " * 20
        + RESPONSE_CLOSE
        + _tools(_call("calc", 1, _arg("x", "number", "1")))
        + MESSAGE_CLOSE
    )
    previous = ""
    for i in range(0, len(stream), 3):
        delta = stream[i : i + 3]
        current = previous + delta
        given = (previous + "x")[:-1] if copy_previous and previous else previous
        parser.extract_tool_calls_streaming(given, current, delta, [], [], [], request)
        previous = current

    assert rescans == 0


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("prefix", ["", RESPONSE_OPEN], ids=["no-open", "open"])
def test_streaming_cost_does_not_grow_with_the_accumulated_text(prefix):
    """Each call used to search the whole accumulated text again, so the cost of
    one delta grew with the length of the response; the median over the last
    calls of a long stream must stay close to that of a short one, whether or
    not the response-open marker was consumed as the generation prefix."""

    def tail_median(num_chars: int) -> float:
        parser = KimiK3ToolParser(_dummy_tokenizer())
        request = _request()
        text = prefix + "response text " * (num_chars // 14)
        deltas = [text[i : i + 4] for i in range(0, len(text), 4)]
        previous, samples = "", []
        for i, delta in enumerate(deltas):
            current = previous + delta
            start = time.perf_counter()
            parser.extract_tool_calls_streaming(
                previous, current, delta, [], [], [], request
            )
            elapsed = time.perf_counter() - start
            if i >= len(deltas) - 300:
                samples.append(elapsed)
            previous = current
        return statistics.median(samples)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        short, long = tail_median(3_000), tail_median(300_000)
    finally:
        if gc_was_enabled:
            gc.enable()

    assert long < 6 * short, (short, long)
