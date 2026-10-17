# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.parser.kimi_k2 import KimiK2Parser
from vllm.reasoning.kimi_k2_reasoning_parser import KimiK2ReasoningParser
from vllm.tokenizers import TokenizerLike, get_tokenizer

REASONING_MODEL_NAME = "moonshotai/Kimi-K2.5"


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return get_tokenizer(tokenizer_name=REASONING_MODEL_NAME, trust_remote_code=True)


def test_extract_reasoning_with_think_tags(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think>step by step reasoning</think>final answer", request
    )
    assert reasoning == "step by step reasoning"
    assert content == "final answer"


def test_extract_reasoning_empty_thinking(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think></think>final answer", request
    )
    assert reasoning is None
    assert content == "final answer"


def test_extract_reasoning_implicit_start(kimi_k2_tokenizer):
    """When there's no <think> tag, everything is treated as reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "implicit reasoning with no tags", request
    )
    assert reasoning == "implicit reasoning with no tags"
    assert content is None


def test_extract_reasoning_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> implicitly ends reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    text = "some reasoning<|tool_calls_section_begin|>tool call data"
    reasoning, content = parser.extract_reasoning(text, request)
    assert reasoning == "some reasoning"
    assert content == "<|tool_calls_section_begin|>tool call data"


def test_streaming_reasoning_then_content(kimi_k2_tokenizer):
    """Token-by-token streaming: reasoning tokens then content after </think>."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    parser_engine = parser._parser_engine
    assert isinstance(parser_engine, KimiK2Parser)

    think_id = parser_engine._start_token_id
    end_think_id = parser_engine._end_token_id
    assert think_id is not None
    assert end_think_id is not None
    # Use a real token ID from the tokenizer for regular content
    regular_id = kimi_k2_tokenizer.encode("hello", add_special_tokens=False)[0]

    # First token: <think> — single special token should be skipped
    result = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<think>",
        delta_text="<think>",
        previous_token_ids=[],
        current_token_ids=[think_id],
        delta_token_ids=[think_id],
    )
    assert result is None

    # Reasoning token
    result = parser.extract_reasoning_streaming(
        previous_text="<think>",
        current_text="<think>step one",
        delta_text="step one",
        previous_token_ids=[think_id],
        current_token_ids=[think_id, regular_id],
        delta_token_ids=[regular_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.reasoning == "step one"
    assert result.content is None

    # End token </think> as single token — should be skipped
    result = parser.extract_reasoning_streaming(
        previous_text="<think>step one",
        current_text="<think>step one</think>",
        delta_text="</think>",
        previous_token_ids=[think_id, regular_id],
        current_token_ids=[think_id, regular_id, end_think_id],
        delta_token_ids=[end_think_id],
    )
    assert result is None

    # Content after </think>
    content_id = kimi_k2_tokenizer.encode("world", add_special_tokens=False)[0]
    result = parser.extract_reasoning_streaming(
        previous_text="<think>step one</think>",
        current_text="<think>step one</think>answer",
        delta_text="answer",
        previous_token_ids=[think_id, regular_id, end_think_id],
        current_token_ids=[think_id, regular_id, end_think_id, content_id],
        delta_token_ids=[content_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "answer"


def test_streaming_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> in delta ends reasoning during streaming."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    parser_engine = parser._parser_engine
    assert isinstance(parser_engine, KimiK2Parser)

    think_id = parser_engine._start_token_id
    tool_begin_id = parser_engine._tool_section_start_token_id
    assert think_id is not None
    assert tool_begin_id is not None
    regular_id = kimi_k2_tokenizer.encode("hello", add_special_tokens=False)[0]

    # Tool section token arrives — should transition from reasoning to content
    result = parser.extract_reasoning_streaming(
        previous_text="<think>thinking",
        current_text="<think>thinking<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[think_id, regular_id],
        current_token_ids=[think_id, regular_id, tool_begin_id],
        delta_token_ids=[tool_begin_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "<|tool_calls_section_begin|>"


SYSTEM_TURN = "<|im_system|>system<|im_middle|>You are Kimi.<|im_end|>"
USER_TURN = "<|im_user|>user<|im_middle|>Weather in Paris?<|im_end|>"
GENERATION_PROMPT = "<|im_assistant|>assistant<|im_middle|>"
TOOL_CALL = (
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
    '<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>'
    "<|tool_calls_section_end|>"
)
# Kimi-K2-Thinking replays <think>...</think> in earlier assistant turns and
# its generation prompt does not open <think>; the model emits it.
K2_THINKING_PROMPTS = {
    "first_turn": SYSTEM_TURN + USER_TURN + GENERATION_PROMPT,
    "second_turn": SYSTEM_TURN
    + "<|im_user|>user<|im_middle|>2+2?<|im_end|>"
    + "<|im_assistant|>assistant<|im_middle|><think>Simple sum.</think>4<|im_end|>"
    + USER_TURN
    + GENERATION_PROMPT,
    "tool_loop": SYSTEM_TURN
    + USER_TURN
    + "<|im_assistant|>assistant<|im_middle|><think>Need the tool.</think>"
    + TOOL_CALL
    + "<|im_end|>"
    + "<|im_system|>tool<|im_middle|>## Return of functions.get_weather:0\n"
    + "sunny<|im_end|>"
    + GENERATION_PROMPT,
}
ParsedTurn = tuple[str | None, str | None, list[tuple[str, str]]]


def _make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        }
    )


def _make_parser(
    tokenizer: TokenizerLike,
    request: ChatCompletionRequest,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> Parser:
    parser_cls = ParserManager.get_parser(
        tool_parser_name="kimi_k2",
        reasoning_parser_name="kimi_k2",
        enable_auto_tools=True,
        tokenizer=tokenizer,
    )
    assert parser_cls is not None
    return parser_cls(
        tokenizer, request.tools, chat_template_kwargs=chat_template_kwargs
    )


def _parse_streaming(
    tokenizer: TokenizerLike, prompt_ids: list[int], output: str
) -> ParsedTurn:
    request = _make_request()
    parser = _make_parser(tokenizer, request)
    output_ids = tokenizer.encode(output, add_special_tokens=False)
    reasoning, content = "", ""
    calls: dict[int, list[str]] = {}
    for i, token_id in enumerate(output_ids):
        delta = parser.parse_delta(
            tokenizer.decode([token_id]),
            [token_id],
            request,
            prompt_token_ids=prompt_ids,
            finished=i == len(output_ids) - 1,
        )
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for tc in delta.tool_calls:
            call = calls.setdefault(tc.index, ["", ""])
            if tc.function is not None:
                call[0] += tc.function.name or ""
                call[1] += tc.function.arguments or ""
    return (
        reasoning or None,
        content or None,
        [(name, args) for name, args in calls.values()],
    )


def _parse_non_streaming(
    tokenizer: TokenizerLike, prompt_ids: list[int], output: str
) -> ParsedTurn:
    request = _make_request()
    parser = _make_parser(tokenizer, request)
    parser.set_prompt_token_ids(prompt_ids)
    reasoning, content, tool_calls = parser.parse(
        output,
        request,
        enable_auto_tools=True,
        model_output_token_ids=tokenizer.encode(output, add_special_tokens=False),
    )
    return (
        reasoning,
        content,
        [(tc.name, tc.arguments) for tc in tool_calls or []],
    )


@pytest.mark.parametrize(
    "prompt", K2_THINKING_PROMPTS.values(), ids=K2_THINKING_PROMPTS
)
@pytest.mark.parametrize(
    "output, expected",
    [
        (
            "<think>The tool says sunny.</think>It is sunny in Paris.",
            ("The tool says sunny.", "It is sunny in Paris.", []),
        ),
        (
            "<think>Check Rome too.</think>" + TOOL_CALL.replace("Paris", "Rome"),
            ("Check Rome too.", None, [("get_weather", '{"city": "Rome"}')]),
        ),
    ],
    ids=["answer", "tool_call"],
)
def test_reasoning_from_earlier_turns_does_not_end_current_turn(
    kimi_k2_tokenizer: TokenizerLike,
    prompt: str,
    output: str,
    expected: ParsedTurn,
) -> None:
    """A </think> replayed in an earlier turn must not mark the new turn's
    reasoning as finished, for streaming, non-streaming and the structured
    output gate alike."""
    prompt_ids = kimi_k2_tokenizer.encode(prompt, add_special_tokens=False)
    assert _parse_streaming(kimi_k2_tokenizer, prompt_ids, output) == expected
    assert _parse_non_streaming(kimi_k2_tokenizer, prompt_ids, output) == expected

    parser = _make_parser(kimi_k2_tokenizer, _make_request())
    assert parser.is_reasoning_end(prompt_ids) is False
    assert _make_request().resolve_reasoning_ended(parser, prompt_ids) is False


@pytest.mark.parametrize(
    "prompt, chat_template_kwargs, ended",
    [
        # Kimi-K2.5 opens <think> in the generation prompt.
        (K2_THINKING_PROMPTS["second_turn"] + "<think>", None, False),
        (
            K2_THINKING_PROMPTS["second_turn"] + "<think></think>",
            {"thinking": False},
            True,
        ),
        # Continuing the final assistant message of the current turn.
        (K2_THINKING_PROMPTS["second_turn"] + "<think>Done.</think>Part", None, True),
        (K2_THINKING_PROMPTS["second_turn"] + "<think>Still going", None, False),
    ],
    ids=[
        "k2_5_thinking",
        "k2_5_thinking_disabled",
        "continued_closed",
        "continued_open",
    ],
)
def test_prompt_reasoning_end_within_current_turn(
    kimi_k2_tokenizer: TokenizerLike,
    prompt: str,
    chat_template_kwargs: dict[str, Any] | None,
    ended: bool,
) -> None:
    prompt_ids = kimi_k2_tokenizer.encode(prompt, add_special_tokens=False)
    parser = _make_parser(kimi_k2_tokenizer, _make_request(), chat_template_kwargs)

    assert parser.is_reasoning_end(prompt_ids) is ended
