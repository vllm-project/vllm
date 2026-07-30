# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.kimi_k3 import KimiK3Parser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.kimi_k3_reasoning_parser import KimiK3ReasoningParser

OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"
THINK_OPEN = f"{OPEN}think{SEP}"
THINK_CLOSE = f"{CLOSE}think{SEP}"
RESPONSE_OPEN = f"{OPEN}response{SEP}"


class DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == THINK_OPEN:
            return [1, 2, 3]
        if text == THINK_CLOSE:
            return [4, 2, 3]
        return [ord(ch) for ch in text]


def test_parser_manager_selects_kimi_k3_parser_for_reasoning_only():
    parser_cls = ParserManager.get_parser(reasoning_parser_name="kimi_k3")

    assert parser_cls is not None
    assert parser_cls.reasoning_parser_cls is KimiK3ReasoningParser
    assert parser_cls.tool_parser_cls is None


def test_reasoning_registry_uses_kimi_k3_compatibility_entry_point():
    assert (
        ReasoningParserManager.get_reasoning_parser("kimi_k3") is KimiK3ReasoningParser
    )


def test_reasoning_adapter_delegates_to_kimi_k3_engine():
    parser = KimiK3ReasoningParser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    reasoning, content = parser.extract_reasoning(
        f"{THINK_OPEN}step{THINK_CLOSE}{RESPONSE_OPEN}answer",
        request,
    )

    assert reasoning == "step"
    assert content == "answer"


def test_parser_selection_thinking_disabled():
    parser = KimiK3Parser(DummyTokenizer(), chat_template_kwargs={"thinking": False})

    assert parser.parser_engine_config.initial_state.name == "CONTENT"


def test_extract_reasoning_with_xtml_tags():
    parser = KimiK3Parser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    reasoning, content = parser.extract_reasoning(
        f"{THINK_OPEN}step{THINK_CLOSE}{RESPONSE_OPEN}answer",
        request,
    )

    assert reasoning == "step"
    assert content == "answer"


def test_extract_reasoning_with_generation_prefix_consumed():
    parser = KimiK3Parser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    reasoning, content = parser.extract_reasoning(
        f"step{THINK_CLOSE}{RESPONSE_OPEN}answer",
        request,
    )

    assert reasoning == "step"
    assert content == "answer"


def test_parser_engine_parses_next_assistant_turn_after_multi_turn_prompt():
    parser = KimiK3Parser(DummyTokenizer())
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow-up question"},
        ],
    )

    reasoning, content, tool_calls = parser.parse(
        f"second reasoning{THINK_CLOSE}{RESPONSE_OPEN}second answer",
        request,
    )

    assert reasoning == "second reasoning"
    assert content == "second answer"
    assert tool_calls is None


def test_parser_engine_strips_response_wrapper():
    parser = KimiK3Parser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    reasoning, content, tool_calls = parser.parse(
        f"{THINK_OPEN}step{THINK_CLOSE}{RESPONSE_OPEN}answer",
        request,
    )

    assert reasoning == "step"
    assert content == "answer"
    assert tool_calls is None


def test_is_reasoning_end_uses_full_input_ids():
    parser = KimiK3Parser(DummyTokenizer())

    assert not parser.is_reasoning_end([4, 2])
    assert parser.is_reasoning_end([4, 2, 3])


def test_is_reasoning_end_ignores_stale_close_from_prior_turn():
    # DummyTokenizer: THINK_OPEN -> [1, 2, 3], THINK_CLOSE -> [4, 2, 3].
    # Multi-turn / agent continuation: a prior turn's think channel (its close
    # marker) is kept in the prompt, then the current turn opens a new think
    # block that has not closed yet. Reasoning must read as NOT ended, otherwise
    # the structured-output gate constrains the current turn's reasoning.
    parser = KimiK3Parser(DummyTokenizer())

    stale_close = [4, 2, 3]
    new_open = [1, 2, 3]
    # prior close, then current-turn open still unclosed -> not ended
    assert not parser.is_reasoning_end([*stale_close, *new_open])
    # ...then the current turn emits its own close -> ended
    assert parser.is_reasoning_end([*stale_close, *new_open, *stale_close])
    # open with no close yet -> not ended
    assert not parser.is_reasoning_end([*new_open])


def test_streaming_split_open_marker_is_held_back():
    parser = KimiK3Parser(DummyTokenizer())

    first = parser.extract_reasoning_streaming(
        previous_text="",
        current_text=OPEN,
        delta_text=OPEN,
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
    )
    second = parser.extract_reasoning_streaming(
        previous_text=OPEN,
        current_text=f"{OPEN}think",
        delta_text="think",
        previous_token_ids=[1],
        current_token_ids=[1, 2],
        delta_token_ids=[2],
    )
    third = parser.extract_reasoning_streaming(
        previous_text=f"{OPEN}think",
        current_text=THINK_OPEN + "step",
        delta_text=f"{SEP}step",
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, 3, 9],
        delta_token_ids=[3, 9],
    )

    assert first is None
    assert second is None
    assert isinstance(third, DeltaMessage)
    assert third.reasoning == "step"


def test_streaming_split_close_marker_hands_content_downstream():
    parser = KimiK3Parser(DummyTokenizer())

    previous_text = f"{THINK_OPEN}step"
    partial_close = parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=previous_text + CLOSE,
        delta_text=CLOSE,
        previous_token_ids=[1, 2, 3, 9],
        current_token_ids=[1, 2, 3, 9, 4],
        delta_token_ids=[4],
    )
    closed = parser.extract_reasoning_streaming(
        previous_text=previous_text + CLOSE,
        current_text=previous_text + f"{THINK_CLOSE}{RESPONSE_OPEN}answer",
        delta_text=f"think{SEP}{RESPONSE_OPEN}answer",
        previous_token_ids=[1, 2, 3, 9, 4],
        current_token_ids=[1, 2, 3, 9, 4, 2, 3, 10],
        delta_token_ids=[2, 3, 10],
    )

    assert partial_close is None
    assert isinstance(closed, DeltaMessage)
    assert closed.reasoning is None
    assert closed.content == "answer"
    assert parser.extract_content_ids([4, 2, 3, 10]) == [10]


def test_thinking_disabled_streams_content():
    parser = KimiK3Parser(
        DummyTokenizer(), chat_template_kwargs={"enable_thinking": False}
    )

    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text=f"{RESPONSE_OPEN}answer",
        delta_text=f"{RESPONSE_OPEN}answer",
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
    )

    assert isinstance(delta, DeltaMessage)
    assert delta.content == "answer"
    assert delta.reasoning is None


def test_parser_engine_thinking_false_streams_response_content():
    parser = KimiK3Parser(DummyTokenizer(), chat_template_kwargs={"thinking": False})
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"thinking": False},
    )

    first = parser.parse_delta(
        delta_text="OK",
        delta_token_ids=[10],
        request=request,
        prompt_token_ids=[1],
        finished=False,
    )
    partial_close = parser.parse_delta(
        delta_text=CLOSE,
        delta_token_ids=[2],
        request=request,
        prompt_token_ids=[1],
        finished=False,
    )
    closed = parser.parse_delta(
        delta_text=f"response{SEP}",
        delta_token_ids=[3, 4],
        request=request,
        prompt_token_ids=[1],
        finished=False,
    )

    assert first is not None
    assert first.content == "OK"
    assert first.reasoning is None
    assert partial_close is None
    assert closed is None


def test_adjust_request_keeps_xtml_markers_contiguous():
    parser = KimiK3Parser(DummyTokenizer())
    request = ChatCompletionRequest(model="test-model", messages=[])

    adjusted = parser.adjust_request(request)

    assert adjusted.skip_special_tokens is False
    if hasattr(adjusted, "spaces_between_special_tokens"):
        assert adjusted.spaces_between_special_tokens is False
