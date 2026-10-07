# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Llama 3.x/4 JSON parser (llama3_json /
llama4_json).

Staged commit 1/7: minimal envelope-balance core, Granite-equivalent scope.
Later commits grow both llama_json.py and this file together.
"""

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.llama_json import LlamaJsonParser

PYTHON_TAG = "<|python_tag|>"

_LLAMA_VOCAB = {
    PYTHON_TAG: 128010,
    "<|eot_id|>": 128009,
    "<|eom_id|>": 128008,
}


def _mock_tokenizer():
    return make_mock_tokenizer(_LLAMA_VOCAB)


def _parser():
    return LlamaJsonParser(_mock_tokenizer())


def _request(tools=None, tool_choice=None):
    return ChatCompletionRequest(
        model="llama",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice=tool_choice,
    )


def _stream(parser, request, text: str, chunk_size: int):
    """Character-granularity streaming; markers aren't tokenized here yet
    (the mock-tokenizer marker plumbing lands with phantom-retraction in
    commit 2 -- this stage only needs plain text)."""
    results = []
    previous = ""
    for start in range(0, len(text), chunk_size):
        delta_text = text[start : start + chunk_size]
        results.append(
            parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=previous + delta_text,
                delta_text=delta_text,
                previous_token_ids=(),
                current_token_ids=(),
                delta_token_ids=(),
                request=request,
            )
        )
        previous += delta_text
    results.append(parser.finish_streaming())
    return results


def _accumulate(results):
    content_parts = []
    calls: dict[int, dict] = {}
    for delta in results:
        if delta is None:
            continue
        if delta.content:
            content_parts.append(delta.content)
        for tc in delta.tool_calls or []:
            call = calls.setdefault(tc.index, {"name": None, "args": ""})
            if tc.function and tc.function.name:
                call["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                call["args"] += tc.function.arguments
    return "".join(content_parts), [{"index": i, **calls[i]} for i in sorted(calls)]


class TestBasicCall:
    def test_non_streaming_envelope_with_parameters(self):
        text = '{"name": "get_weather", "parameters": {"city": "SF"}}'
        result = LlamaJsonParser(_mock_tokenizer()).extract_tool_calls_from_content(
            text, _request()
        )
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].function.arguments == '{"city": "SF"}'

    def test_non_streaming_envelope_with_arguments_key(self):
        text = '{"name": "get_weather", "arguments": {"city": "SF"}}'
        result = LlamaJsonParser(_mock_tokenizer()).extract_tool_calls_from_content(
            text, _request()
        )
        assert result.tool_calls[0].function.arguments == '{"city": "SF"}'

    def test_non_streaming_plain_content_has_no_call(self):
        text = "This is just some text without any tool calls"
        result = LlamaJsonParser(_mock_tokenizer()).extract_tool_calls_from_content(
            text, _request()
        )
        assert result.tools_called is False

    def test_streaming_basic_call(self):
        text = '{"name": "get_weather", "parameters": {"city": "SF"}}'
        results = _stream(_parser(), _request(), text, chunk_size=3)
        content, calls = _accumulate(results)
        assert content == ""
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["args"] == '{"city": "SF"}'

    def test_streaming_single_char_chunks(self):
        text = '{"name": "f", "parameters": {"x": 1}}'
        results = _stream(_parser(), _request(), text, chunk_size=1)
        _, calls = _accumulate(results)
        assert len(calls) == 1
        assert calls[0]["name"] == "f"
        assert calls[0]["args"] == '{"x": 1}'
