# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Invariant checks for the Mistral Small 4 chat template."""

import copy

import pytest

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
from vllm.tokenizers.mistral import MistralTokenizer

SUPPORTED_CASES = {
    **BASIC_CASES,
    **TOOL_CALL_CASES,
    **PARALLEL_TOOL_CALL_CASES,
    **REASONING_CASES,
    **TOOL_CALL_W_REASONING_CASES,
    **PARALLEL_TOOL_CALL_W_REASONING_CASES,
}


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = MistralTokenizer.from_pretrained("mistralai/Mistral-Small-4-119B-2603")
    return tokenizer


def _render(tokenizer, messages: list[dict[str, object]], **kwargs) -> str:
    id_map: dict[str, str] = {}

    def remap_tool_call_id(tool_call_id: object) -> object:
        if not isinstance(tool_call_id, str):
            return tool_call_id

        normalized_id = id_map.get(tool_call_id)
        if normalized_id is None:
            normalized_id = f"{abs(hash(tool_call_id)):x}"[-9:].zfill(9)
            id_map[tool_call_id] = normalized_id
        return normalized_id

    # Mistral 4 requires 9-character alphanumeric tool call IDs.
    normalized_messages = []
    for message in messages:
        message = copy.deepcopy(message)

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and "id" in tool_call:
                    tool_call["id"] = remap_tool_call_id(tool_call["id"])

        if "tool_call_id" in message:
            message["tool_call_id"] = remap_tool_call_id(message["tool_call_id"])

        normalized_messages.append(message)

    kwargs.setdefault("tokenize", False)
    kwargs.setdefault("add_generation_prompt", False)
    kwargs.setdefault("continue_final_message", messages[-1].get("role") == "assistant")
    return tokenizer.apply_chat_template(normalized_messages, **kwargs)


class TestMistralSmall4ChatTemplateInvariants(TestChatTemplateInvariants):
    inst_delimiter = ("[INST]", "[/INST]")
    reasoning_delimiter = ("[THINK]", "[/THINK]")
    tool_response_delimiter = ("[TOOL_RESULTS]", "[/TOOL_RESULTS]")
    tool_call_marker = "[TOOL_CALLS]"

    @classmethod
    def _build_markers(cls, messages):
        markers = []
        inst_start, inst_end = cls.inst_delimiter
        reasoning_start, reasoning_end = cls.reasoning_delimiter
        tool_response_start, tool_response_end = cls.tool_response_delimiter

        last_non_user_message_idx = 0
        for idx, msg in enumerate(messages):
            if msg.get("role") != "user":
                last_non_user_message_idx = idx

        for idx, msg in enumerate(messages):
            msg_role = msg.get("role")
            if msg_role == "tool":
                markers.append(tool_response_start)
            elif msg_role == "user":
                markers.append(inst_start)

            if idx > last_non_user_message_idx:
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
                    markers.append(cls.tool_call_marker)
                    markers.append(tool_call_name)

            if msg_role == "tool":
                markers.append(tool_response_end)
            elif msg_role == "user":
                markers.append(inst_end)

        return markers

    @classmethod
    def _check_delimiters(cls, messages, result):
        inst_state = delimiter_state(result, cls.inst_delimiter)
        assert inst_state == 0

        reasoning_state = delimiter_state(result, cls.reasoning_delimiter)
        assert reasoning_state == 0

        tool_response_state = delimiter_state(result, cls.tool_response_delimiter)
        assert tool_response_state == 0

    @pytest.mark.parametrize(
        "test_case", SUPPORTED_CASES.values(), ids=SUPPORTED_CASES.keys()
    )
    def test_invariants(
        self,
        tokenizer,
        test_case,
    ):
        messages = create_conversation(*test_case)
        result = _render(tokenizer, messages, reasoning_effort="high")
        self._test_case(messages, result)
