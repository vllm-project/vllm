# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_STRICT_TOOL_CALLING_ENV = "VLLM_ENFORCE_STRICT_TOOL_CALLING"
_STRICT_TOOL_CALLING_ENV_VALUE = os.environ.get(_STRICT_TOOL_CALLING_ENV)
os.environ[_STRICT_TOOL_CALLING_ENV] = "0"

from tests.parser.engine.conftest import (  # noqa: E402
    VOCAB,
    CombinedDelegating,
    make_mock_tokenizer,
)
from tests.parser.engine.replay_harness import MockTokenizer  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest  # noqa: E402
from vllm.parser.abstract_parser import DelegatingParser  # noqa: E402
from vllm.parser.engine.adapters import make_adapters  # noqa: E402
from vllm.parser.gemma4 import Gemma4Parser  # noqa: E402
from vllm.parser.qwen3 import Qwen3Parser  # noqa: E402
from vllm.parser.utils import count_history_tool_calls  # noqa: E402
from vllm.reasoning.basic_parsers import (  # noqa: E402
    BaseThinkingReasoningParser,
)
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def restore_strict_tool_calling_env():
    yield
    if _STRICT_TOOL_CALLING_ENV_VALUE is None:
        os.environ.pop(_STRICT_TOOL_CALLING_ENV, None)
    else:
        os.environ[_STRICT_TOOL_CALLING_ENV] = _STRICT_TOOL_CALLING_ENV_VALUE


class ThinkReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"


MODEL_OUTPUT = (
    "<think>let me think about this</think>"
    '<tool_call>\n{"name": "get_weather", '
    '"arguments": {"city": "Dallas"}}\n</tool_call>'
)

PLAIN_TEXT = "The weather in Dallas is sunny and 75°F."

TOOL_CALL_ONLY = (
    '<tool_call>\n{"name": "get_weather", '
    '"arguments": {"city": "Dallas"}}\n</tool_call>'
)

TOOL_ARGUMENTS = '{"city": "Dallas"}'


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


def make_request(**overrides):
    base = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return ChatCompletionRequest.model_validate(base)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


KIMI_K2_MODEL_CONFIG = SimpleNamespace(
    hf_text_config=SimpleNamespace(model_type="kimi_k2"),
    hf_overrides=None,
)

HISTORY_MESSAGES = [
    {"role": "user", "content": "first"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "functions.get_current_weather:0",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": "{}",
                },
            },
            {
                "id": "functions.get_forecast:1",
                "type": "function",
                "function": {
                    "name": "get_forecast",
                    "arguments": "{}",
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "functions.get_current_weather:0",
        "content": "{}",
    },
    {
        "role": "tool",
        "tool_call_id": "functions.get_forecast:1",
        "content": "{}",
    },
    {"role": "user", "content": "again"},
]


def make_parser(tokenizer, reasoning=False, tool=False, **kwargs):
    class TestParser(DelegatingParser):
        reasoning_parser_cls = ThinkReasoningParser if reasoning else None
        tool_parser_cls = Hermes2ProToolParser if tool else None

    return TestParser(tokenizer, **kwargs)


@pytest.mark.parametrize(
    "reasoning,tool",
    [(False, False), (False, True)],
    ids=["neither", "tool-only"],
)
def test_parse_plain_text_no_reasoning_parser(tokenizer, reasoning, tool):
    parser = make_parser(tokenizer, reasoning=reasoning, tool=tool)
    request = make_request()
    r, content, tool_calls = parser.parse(PLAIN_TEXT, request)

    assert r is None
    assert content == PLAIN_TEXT
    assert tool_calls is not None
    assert len(tool_calls) == 0


@pytest.mark.parametrize(
    "reasoning,tool",
    [(True, False), (True, True)],
    ids=["reasoning-only", "both"],
)
def test_parse_plain_text_with_reasoning_parser(tokenizer, reasoning, tool):
    parser = make_parser(tokenizer, reasoning=reasoning, tool=tool)
    request = make_request()
    r, content, tool_calls = parser.parse(PLAIN_TEXT, request)

    assert r == PLAIN_TEXT
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 0


def test_parse_both_parsers(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    request = make_request(tools=TOOLS)
    reasoning, content, tool_calls = parser.parse(
        MODEL_OUTPUT, request, enable_auto_tools=True
    )

    assert reasoning is not None
    assert "let me think about this" in reasoning
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}
    assert not content or content.strip() == ""


def test_parse_reasoning_only(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    request = make_request()
    reasoning, content, tool_calls = parser.parse(MODEL_OUTPUT, request)

    assert reasoning is not None
    assert "let me think about this" in reasoning
    assert content is not None
    assert "<tool_call>" in content
    assert "get_weather" in content
    assert tool_calls is not None
    assert len(tool_calls) == 0


def test_parse_tool_only(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = make_request(tools=TOOLS)
    reasoning, content, tool_calls = parser.parse(
        MODEL_OUTPUT, request, enable_auto_tools=True
    )

    assert reasoning is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}


def test_parse_named_tool_choice(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = make_request(
        tools=TOOLS,
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
    )
    reasoning, content, tool_calls = parser.parse(
        TOOL_ARGUMENTS, request, enable_auto_tools=True
    )

    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == TOOL_ARGUMENTS


def test_parse_named_tool_choice_with_reasoning(tokenizer):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    model_output = f"<think>thinking</think>{TOOL_ARGUMENTS}"
    request = make_request(
        tools=TOOLS,
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
    )
    reasoning, content, tool_calls = parser.parse(
        model_output, request, enable_auto_tools=True
    )

    assert reasoning is not None
    assert "thinking" in reasoning
    assert content is None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == TOOL_ARGUMENTS


def test_parse_required_tool_choice(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    functions_json = json.dumps(
        [
            {"name": "get_weather", "parameters": {"city": "Dallas"}},
            {"name": "get_time", "parameters": {"timezone": "UTC"}},
        ]
    )
    request = make_request(tools=TOOLS, tool_choice="required")
    reasoning, content, tool_calls = parser.parse(
        functions_json, request, enable_auto_tools=True
    )

    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}
    assert tool_calls[1].name == "get_time"
    assert json.loads(tool_calls[1].arguments) == {"timezone": "UTC"}


def test_parse_required_tool_choice_kimi_k2_ids(tokenizer):
    parser = make_parser(
        tokenizer, reasoning=False, tool=True, model_config=KIMI_K2_MODEL_CONFIG
    )
    functions_json = json.dumps(
        [
            {"name": "get_current_weather", "parameters": {"city": "Dallas"}},
            {"name": "get_forecast", "parameters": {"city": "Dallas", "days": 2}},
        ]
    )
    request = make_request(tools=TOOLS, tool_choice="required")
    _, content, tool_calls = parser.parse(
        functions_json, request, enable_auto_tools=True
    )

    assert content is None
    assert tool_calls is not None
    assert [tc.id for tc in tool_calls] == [
        "functions.get_current_weather:0",
        "functions.get_forecast:1",
    ]


def test_parse_required_tool_choice_kimi_k2_ids_after_history(tokenizer):
    parser = make_parser(
        tokenizer, reasoning=False, tool=True, model_config=KIMI_K2_MODEL_CONFIG
    )
    functions_json = json.dumps(
        [{"name": "get_current_weather", "parameters": {"city": "Dallas"}}]
    )
    request = make_request(
        messages=HISTORY_MESSAGES,
        tools=TOOLS,
        tool_choice="required",
    )
    _, _, tool_calls = parser.parse(functions_json, request, enable_auto_tools=True)

    assert tool_calls is not None
    assert tool_calls[0].id == "functions.get_current_weather:2"


def test_count_history_tool_calls_responses_request():
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_0",
                    "name": "get_current_weather",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_forecast",
                    "arguments": "{}",
                },
            ],
        }
    )

    assert count_history_tool_calls(request) == 2


def test_parse_required_tool_choice_random_ids_deferred(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    functions_json = json.dumps(
        [{"name": "get_current_weather", "parameters": {"city": "Dallas"}}]
    )
    request = make_request(
        messages=HISTORY_MESSAGES,
        tools=TOOLS,
        tool_choice="required",
    )
    _, _, tool_calls = parser.parse(functions_json, request, enable_auto_tools=True)

    assert tool_calls is not None
    assert tool_calls[0].id is None


def test_parse_named_tool_choice_kimi_k2_id(tokenizer):
    parser = make_parser(
        tokenizer, reasoning=False, tool=True, model_config=KIMI_K2_MODEL_CONFIG
    )
    request = make_request(
        tools=TOOLS,
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
    )
    _, content, tool_calls = parser.parse(
        TOOL_ARGUMENTS, request, enable_auto_tools=True
    )

    assert content is None
    assert tool_calls is not None
    assert tool_calls[0].id == "functions.get_weather:0"


def test_parse_named_tool_choice_content_none(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = make_request(
        tools=TOOLS,
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
    )
    reasoning, content, tool_calls = parser.parse("", request, enable_auto_tools=True)
    assert reasoning is None
    assert content is None
    assert tool_calls is not None


def test_parse_required_tool_choice_content_none(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = make_request(tools=TOOLS, tool_choice="required")
    reasoning, content, tool_calls = parser.parse("", request, enable_auto_tools=True)
    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 0


def test_parse_auto_tools_no_parser(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=False)
    request = make_request()
    reasoning, content, tool_calls = parser.parse(
        TOOL_CALL_ONLY, request, enable_auto_tools=True
    )

    assert reasoning is None
    assert content == TOOL_CALL_ONLY
    assert tool_calls is not None
    assert len(tool_calls) == 0


def test_parse_auto_tools_no_calls_returns_none(tokenizer):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = make_request(tools=TOOLS)
    reasoning, content, tool_calls = parser.parse(
        PLAIN_TEXT, request, enable_auto_tools=True
    )

    assert reasoning is None
    assert content == PLAIN_TEXT
    assert tool_calls is None


# ── Token ID forwarding tests ────────────────────────────────────────


def _make_engine_delegating_parser():
    return CombinedDelegating(make_mock_tokenizer(VOCAB))


@contextmanager
def _spy_feed(engine):
    calls: list[tuple[str, list[int]]] = []
    original = engine._feed

    def _spy(text, tids):
        calls.append((text, list(tids)))
        return original(text, tids)

    with patch.object(engine, "_feed", side_effect=_spy):
        yield calls


def test_parse_forwards_token_ids_to_engine_reasoning():
    parser = _make_engine_delegating_parser()
    request = make_request()
    model_output = "<think>thoughts</think>content"
    token_ids = [200, 10, 11, 201, 12, 13]

    with _spy_feed(parser._reasoning_parser._parser_engine) as feed_calls:
        parser.parse(model_output, request, model_output_token_ids=token_ids)

    assert len(feed_calls) == 1
    assert feed_calls[0][1] == token_ids


def test_parse_forwards_token_ids_to_engine_tool_calls():
    parser = _make_engine_delegating_parser()
    request = make_request(tools=TOOLS)
    model_output = (
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Dallas"}}\n</tool_call>'
    )
    token_ids = [202, 30, 31, 32, 203]

    with _spy_feed(parser._tool_parser._parser_engine) as feed_calls:
        parser.parse(
            model_output,
            request,
            enable_auto_tools=True,
            model_output_token_ids=token_ids,
        )

    assert len(feed_calls) == 1
    assert feed_calls[0][1] == token_ids


def test_parse_does_not_forward_token_ids_to_non_engine_reasoning(
    tokenizer,
):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    request = make_request(tools=TOOLS)
    token_ids = [1, 2, 3, 4, 5]

    reasoning, content, tool_calls = parser.parse(
        MODEL_OUTPUT,
        request,
        enable_auto_tools=True,
        model_output_token_ids=token_ids,
    )
    assert reasoning is not None
    assert "let me think about this" in reasoning


def test_parse_threads_token_ids_end_to_end():
    parser = _make_engine_delegating_parser()
    request = make_request(tools=TOOLS)
    model_output = (
        "<think>thoughts</think>"
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Dallas"}}\n</tool_call>'
    )
    token_ids = [200, 10, 11, 201, 202, 30, 31, 32, 203]

    with (
        _spy_feed(parser._reasoning_parser._parser_engine) as r_calls,
        _spy_feed(parser._tool_parser._parser_engine) as t_calls,
    ):
        reasoning, content, tool_calls = parser.parse(
            model_output,
            request,
            enable_auto_tools=True,
            model_output_token_ids=token_ids,
        )

    assert reasoning is not None
    assert "thoughts" in reasoning

    assert len(r_calls) == 1
    assert r_calls[0][1] == token_ids

    assert len(t_calls) == 1
    assert t_calls[0][1] == [202, 30, 31, 32, 203]


# ── DelegatingParser tests with real engine-based parsers ────────────


Qwen3ReasoningAdapter, Qwen3ToolAdapter = make_adapters(Qwen3Parser)
Gemma4ReasoningAdapter, Gemma4ToolAdapter = make_adapters(Gemma4Parser)

# -- Qwen3 fixtures --

_QWEN3_VOCAB: dict[str, int] = {
    "<think>": 200,
    "</think>": 201,
    "<tool_call>": 202,
    "</tool_call>": 203,
}

_QWEN3_REASONING_TOOL_TOKENS: list[tuple[int, str]] = [
    (200, "<think>"),
    (10, "I need "),
    (11, "the weather."),
    (201, "</think>"),
    (13, "\n"),
    (202, "<tool_call>"),
    (14, "\n"),
    (15, "<function=get_weather>"),
    (16, "\n"),
    (17, "<parameter=city>"),
    (18, "Dallas"),
    (19, "</parameter>"),
    (20, "\n"),
    (21, "</function>"),
    (22, "\n"),
    (203, "</tool_call>"),
]


def _qwen3_delegating(tokens=None, vocab=None, **kwargs):
    class Qwen3Delegating(DelegatingParser):
        reasoning_parser_cls = Qwen3ReasoningAdapter
        tool_parser_cls = Qwen3ToolAdapter

    tokens = tokens or _QWEN3_REASONING_TOOL_TOKENS
    vocab = vocab or dict(_QWEN3_VOCAB)
    tok = MockTokenizer(vocab=vocab, tokens=tokens)
    return Qwen3Delegating(tok, **kwargs)


@pytest.mark.parametrize("with_token_ids", [True, False], ids=["with_ids", "no_ids"])
def test_qwen3_parse_reasoning_and_tools(with_token_ids):
    parser = _qwen3_delegating()
    tokens = _QWEN3_REASONING_TOOL_TOKENS
    text = "".join(t for _, t in tokens)
    ids = [tid for tid, _ in tokens] if with_token_ids else None
    request = make_request(tools=TOOLS)

    reasoning, content, tool_calls = parser.parse(
        text, request, enable_auto_tools=True, model_output_token_ids=ids
    )

    assert reasoning == "I need the weather."
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}


def test_qwen3_parse_thinking_disabled_with_token_ids():
    tokens: list[tuple[int, str]] = [
        (30, "Here is the answer."),
        (13, "\n"),
        (202, "<tool_call>"),
        (14, "\n"),
        (15, "<function=get_weather>"),
        (16, "\n"),
        (17, "<parameter=city>"),
        (18, "Dallas"),
        (19, "</parameter>"),
        (20, "\n"),
        (21, "</function>"),
        (22, "\n"),
        (203, "</tool_call>"),
    ]
    parser = _qwen3_delegating(
        tokens=tokens,
        chat_template_kwargs={"enable_thinking": False},
    )
    text = "".join(t for _, t in tokens)
    ids = [tid for tid, _ in tokens]
    request = make_request(tools=TOOLS)

    reasoning, content, tool_calls = parser.parse(
        text, request, enable_auto_tools=True, model_output_token_ids=ids
    )

    assert reasoning is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"


# -- Gemma4 fixtures --

_GEMMA4_VOCAB: dict[str, int] = {
    "<|channel>": 50,
    "<channel|>": 51,
    "<|tool_call>": 48,
    "<tool_call|>": 49,
    '<|"|>': 52,
}

_GEMMA4_REASONING_TOOL_TOKENS: list[tuple[int, str]] = [
    (50, "<|channel>"),
    (60, "thought\n"),
    (61, "The user wants weather."),
    (51, "<channel|>"),
    (48, "<|tool_call>"),
    (62, "call:get_weather"),
    (63, "{"),
    (64, "city:"),
    (52, '<|"|>'),
    (65, "Dallas"),
    (52, '<|"|>'),
    (66, "}"),
    (49, "<tool_call|>"),
]


def _gemma4_delegating(tokens=None, vocab=None, **kwargs):
    class Gemma4Delegating(DelegatingParser):
        reasoning_parser_cls = Gemma4ReasoningAdapter
        tool_parser_cls = Gemma4ToolAdapter

    tokens = tokens or _GEMMA4_REASONING_TOOL_TOKENS
    vocab = vocab or dict(_GEMMA4_VOCAB)
    tok = MockTokenizer(vocab=vocab, tokens=tokens)
    return Gemma4Delegating(tok, **kwargs)


@pytest.mark.parametrize("with_token_ids", [True, False], ids=["with_ids", "no_ids"])
def test_gemma4_parse_reasoning_and_tools(with_token_ids):
    parser = _gemma4_delegating()
    tokens = _GEMMA4_REASONING_TOOL_TOKENS
    text = "".join(t for _, t in tokens)
    ids = [tid for tid, _ in tokens] if with_token_ids else None
    request = make_request(tools=TOOLS)

    reasoning, content, tool_calls = parser.parse(
        text, request, enable_auto_tools=True, model_output_token_ids=ids
    )

    assert reasoning == "The user wants weather."
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {"city": "Dallas"}


def test_gemma4_bare_thought_injection_with_token_ids():
    """Gemma4 _preprocess_feed injects <|channel> when model omits it."""
    tokens: list[tuple[int, str]] = [
        (60, "thought\n"),
        (61, "Reasoning here."),
        (51, "<channel|>"),
        (70, "Final answer"),
    ]
    parser = _gemma4_delegating(tokens=tokens)
    text = "".join(t for _, t in tokens)
    ids = [tid for tid, _ in tokens]
    request = make_request()

    reasoning, content, tool_calls = parser.parse(
        text, request, model_output_token_ids=ids
    )

    assert reasoning == "Reasoning here."
    assert content == "Final answer"
