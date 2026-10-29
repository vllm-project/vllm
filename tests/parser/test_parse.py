# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.parser_manager import ParserManager
from vllm.parser.qwen3 import qwen3_config
from vllm.parser.utils import count_history_tool_calls
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


@pytest.fixture(autouse=True)
def enable_hermes_required_named_parsing(monkeypatch):
    # With VLLM_ENFORCE_STRICT_TOOL_CALLING (default on), Hermes sets
    # supports_required_and_named=False and uses structural-tag guided tool
    # calling. Force it True to test the non-guided JSON required/named parse path.
    monkeypatch.setattr(Hermes2ProToolParser, "supports_required_and_named", True)


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


# --------------------------------------------------------------------------
# Prompt that already closes the reasoning block
# --------------------------------------------------------------------------

CONTINUE_MESSAGES = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Sure,"},
]
REASONED_OUTPUT = "<think>\nlet me think\n</think>\n\n" + PLAIN_TEXT
TOOL_TURN_MESSAGES = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "<think>\nneed a tool\n</think>\n\n",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
]


def reasoning_parser(tokenizer, name="qwen3"):
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name=name, tokenizer=tokenizer
    )
    assert parser_cls is not None
    return parser_cls(tokenizer, None)


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def prompt_case(tokenizer, case):
    """(rendered prompt ids, request, model output) for each prompt shape."""
    if case == "continue_final_message":
        # The template renders an empty, closed think block before "Sure,".
        prompt = tokenizer.apply_chat_template(
            CONTINUE_MESSAGES,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        request = make_request(
            messages=CONTINUE_MESSAGES,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        return encode(tokenizer, prompt), request, " here is the answer"
    if case == "template_closed_thinking":
        # The template switches thinking off, but the parser is not told
        # through chat_template_kwargs (#53284).
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return encode(tokenizer, prompt), make_request(), PLAIN_TEXT
    if case == "earlier_turn_closed_reasoning":
        # Only a previous assistant turn closes reasoning; this one is open.
        prompt = tokenizer.apply_chat_template(
            TOOL_TURN_MESSAGES,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return encode(tokenizer, prompt), make_request(), REASONED_OUTPUT
    assert case == "reasoning_open"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return encode(tokenizer, prompt), make_request(), REASONED_OUTPUT


@pytest.mark.parametrize("case", ["continue_final_message", "template_closed_thinking"])
def test_parse_prompt_closed_reasoning_is_content(tokenizer, case):
    prompt_ids, request, output = prompt_case(tokenizer, case)
    parser = reasoning_parser(tokenizer)
    parser.set_prompt_token_ids(prompt_ids)

    reasoning, content, _ = parser.parse(
        output, request, model_output_token_ids=encode(tokenizer, output)
    )

    assert reasoning is None
    assert content == output
    assert parser.count_reasoning_tokens(encode(tokenizer, output)) == 0


@pytest.mark.parametrize(
    ("parser_name", "case"),
    [
        ("qwen3", "reasoning_open"),
        ("qwen3", "earlier_turn_closed_reasoning"),
        # BaseThinkingReasoningParser scans the prompt without turn
        # boundaries, so it must not decide non-streaming classification.
        ("deepseek_r1", "earlier_turn_closed_reasoning"),
    ],
)
def test_parse_prompt_open_reasoning_still_extracted(tokenizer, parser_name, case):
    prompt_ids, request, output = prompt_case(tokenizer, case)
    parser = reasoning_parser(tokenizer, parser_name)
    parser.set_prompt_token_ids(prompt_ids)

    reasoning, content, _ = parser.parse(
        output, request, model_output_token_ids=encode(tokenizer, output)
    )

    assert reasoning.strip() == "let me think"
    assert content.strip() == PLAIN_TEXT
    assert parser.count_reasoning_tokens(encode(tokenizer, output)) > 0


def test_parse_prompt_ignored_without_turn_boundaries(tokenizer):
    """Without turn boundaries, an earlier turn's </think> looks like this
    turn's, so the engine must not start the output in content."""
    prompt_ids, request, output = prompt_case(
        tokenizer, "earlier_turn_closed_reasoning"
    )
    parser = ParserEngine(tokenizer, parser_engine_config=qwen3_config(thinking=True))
    parser.set_prompt_token_ids(prompt_ids)

    reasoning, content, _ = parser.parse(
        output, request, model_output_token_ids=encode(tokenizer, output)
    )

    assert reasoning is not None and content is not None
    assert reasoning.strip() == "let me think"
    assert content.strip() == PLAIN_TEXT


@pytest.mark.parametrize(
    "case",
    [
        "continue_final_message",
        "template_closed_thinking",
        "reasoning_open",
        "earlier_turn_closed_reasoning",
    ],
)
def test_prompt_reasoning_streaming_matches_non_streaming(tokenizer, case):
    prompt_ids, request, output = prompt_case(tokenizer, case)
    output_ids = encode(tokenizer, output)

    parser = reasoning_parser(tokenizer)
    parser.set_prompt_token_ids(prompt_ids)
    reasoning, content, _ = parser.parse(
        output, request, model_output_token_ids=output_ids
    )
    non_streaming = (reasoning or "", content or "")
    non_streaming_count = parser.count_reasoning_tokens(output_ids)

    parser = reasoning_parser(tokenizer)
    streamed_reasoning = streamed_content = ""
    for i, token_id in enumerate(output_ids):
        delta = parser.parse_delta(
            tokenizer.decode([token_id]),
            [token_id],
            request,
            prompt_token_ids=prompt_ids,
            finished=i == len(output_ids) - 1,
        )
        if delta is not None:
            streamed_reasoning += delta.reasoning or ""
            streamed_content += delta.content or ""

    assert (streamed_reasoning, streamed_content) == non_streaming
    assert parser.count_reasoning_tokens(output_ids) == non_streaming_count
