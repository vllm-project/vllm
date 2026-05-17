# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Invariant checks for the Qwen3 chat templates."""

import copy
from typing import Any

import pytest
from transformers import AutoTokenizer

from tests.renderers.chat_templates.conversation_builder import create_conversation
from tests.renderers.chat_templates.invariant_checks import (
    BASIC_CASES,
    PARALLEL_TOOL_CALL_CASES,
    PARALLEL_TOOL_CALL_W_REASONING_CASES,
    REASONING_CASES,
    TOOL_CALL_CASES,
    TOOL_CALL_W_REASONING_CASES,
    TestChatTemplateInvariants,
    delimiter_state,
)

SUPPORTED_CASES = {
    **BASIC_CASES,
    **TOOL_CALL_CASES,
    **PARALLEL_TOOL_CALL_CASES,
    **REASONING_CASES,
    **TOOL_CALL_W_REASONING_CASES,
    **PARALLEL_TOOL_CALL_W_REASONING_CASES,
}


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("Qwen/Qwen3.6-27B", id="qwen36"),
        pytest.param("Qwen/Qwen3.5-397B-A17B-FP8", id="qwen35"),
        pytest.param("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", id="nemotron3"),
    ],
)
def tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


def _render(tokenizer, messages: list[dict[str, object]], **kwargs) -> str:
    kwargs.setdefault("tokenize", False)
    kwargs.setdefault("add_generation_prompt", False)

    # Qwen3 only supports reasoning_content, not reasoning.
    normalized_messages = []
    for message in messages:
        message = copy.deepcopy(message)
        reasoning = message.get("reasoning")
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        normalized_messages.append(message)

    return tokenizer.apply_chat_template(normalized_messages, **kwargs)


class TestQwen3ChatTemplateInvariants(TestChatTemplateInvariants):
    turn_delimiter = ("<|im_start|>", "<|im_end|>")
    reasoning_delimiter = ("<think>", "</think>")
    tool_call_delimiter = ("<tool_call>", "</tool_call>")
    tool_response_delimiter = ("<tool_response>", "</tool_response>")

    @classmethod
    def _build_markers(cls, messages):
        markers = []
        turn_start, turn_end = cls.turn_delimiter
        reasoning_start, reasoning_end = cls.reasoning_delimiter
        tool_call_start, tool_call_end = cls.tool_call_delimiter
        tool_response_start, tool_response_end = cls.tool_response_delimiter

        last_non_assistant_or_tool_message_idx = 0
        for idx, msg in enumerate(messages):
            if msg.get("role") not in ("assistant", "tool"):
                last_non_assistant_or_tool_message_idx = idx

        for idx, msg in enumerate(messages):
            if msg.get("role") == "tool":
                markers.append(tool_response_start)
            else:
                markers.append(turn_start)

            if idx > last_non_assistant_or_tool_message_idx:
                reasoning = msg.get("reasoning")
                if reasoning is not None:
                    markers.append(reasoning_start)
                    markers.append(reasoning)
                    markers.append(reasoning_end)

            content = msg.get("content")
            if content is not None:
                markers.append(content)

            tool_calls = msg.get("tool_calls", ())
            for tool_call in tool_calls:
                tool_call_name = tool_call.get("function", {}).get("name")
                if tool_call_name is not None:
                    markers.append(tool_call_start)
                    markers.append(tool_call_name)
                    markers.append(tool_call_end)

            if msg.get("role") == "tool":
                markers.append(tool_response_end)
            else:
                markers.append(turn_end)

        return markers

    @classmethod
    def _check_delimiters(cls, messages: list[dict[str, Any]], result: str):
        del messages

        turn_state = delimiter_state(result, cls.turn_delimiter)
        assert turn_state == 0

        reasoning_state = delimiter_state(result, cls.reasoning_delimiter)
        assert reasoning_state == 0

        tool_call_state = delimiter_state(result, cls.tool_call_delimiter)
        assert tool_call_state == 0

        tool_response_state = delimiter_state(result, cls.tool_response_delimiter)
        assert tool_response_state == 0

    @pytest.mark.parametrize(
        "test_case",
        SUPPORTED_CASES.values(),
        ids=SUPPORTED_CASES.keys(),
    )
    def test_invariants(
        self,
        tokenizer,
        test_case,
    ):
        messages = create_conversation(*test_case)
        result = _render(tokenizer, messages, enable_thinking=True)
        self._test_case(messages, result)
