# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Invariant checks for Llama 3.x chat templates."""

from pathlib import Path
from typing import Any

import pytest
from transformers import AutoTokenizer

from tests.renderers.chat_templates.conversation_builder import create_conversation
from tests.renderers.chat_templates.invariant_checks import (
    BASIC_CASES,
    TOOL_CALL_CASES,
    TestChatTemplateInvariants,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"

LLAMA31_TEMPLATE = (EXAMPLES_DIR / "tool_chat_template_llama3.1_json.jinja").read_text()
LLAMA32_TEMPLATE = (EXAMPLES_DIR / "tool_chat_template_llama3.2_json.jinja").read_text()

# Llama 3.x JSON tool calling does not support parallel tool calls and
# drops assistant content when a tool call is emitted in the same turn.
UNSUPPORTED_TOOL_CALL_CASES = {
    "tool_call_w_content",
    "tool_call_w_content_w_response",
    "tool_turn_w_content",
    "tool_turn_w_content_user_interrupt",
    "multi_tool_call_w_content",
    "multi_tool_call_w_content_w_response",
    "multi_tool_turn_w_content",
    "multi_tool_turn_w_content_user_followup",
}

SUPPORTED_CASES = {
    **BASIC_CASES,
    **{
        name: case
        for name, case in TOOL_CALL_CASES.items()
        if name not in UNSUPPORTED_TOOL_CALL_CASES
    },
}


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            ("meta-llama/Llama-3.1-8B-Instruct", LLAMA31_TEMPLATE),
            id="llama31-8b",
        ),
        pytest.param(
            ("meta-llama/Llama-3.2-1B-Instruct", LLAMA32_TEMPLATE),
            id="llama32-1b",
        ),
        pytest.param(
            ("meta-llama/Llama-3.3-70B-Instruct", None),
            id="llama33-70b",
        ),
    ],
)
def tokenizer_and_template(request):
    model_name, chat_template = request.param
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    return tokenizer, chat_template


def _render(
    tokenizer_and_template: tuple[Any, str | None],
    messages: list[dict[str, object]],
    **kwargs,
) -> str:
    tokenizer, chat_template = tokenizer_and_template
    kwargs.setdefault("tokenize", False)
    kwargs.setdefault("add_generation_prompt", False)
    if chat_template is not None:
        kwargs.setdefault("chat_template", chat_template)
    return tokenizer.apply_chat_template(messages, **kwargs)


class TestLlamaChatTemplateInvariants(TestChatTemplateInvariants):
    eot_marker = "<|eot_id|>"

    @classmethod
    def _check_delimiters(cls, messages: list[dict[str, Any]], result: str):
        pass

    @classmethod
    def _build_markers(cls, messages: list[dict[str, Any]]) -> list[str]:
        markers = []
        for msg in messages:
            content = msg.get("content")
            if content is not None:
                markers.append(content)
            markers.append(cls.eot_marker)
        return markers

    @pytest.mark.parametrize(
        "test_case",
        SUPPORTED_CASES.values(),
        ids=SUPPORTED_CASES.keys(),
    )
    def test_invariants(
        self,
        tokenizer_and_template,
        test_case,
    ):
        messages = create_conversation(*test_case)
        result = _render(tokenizer_and_template, messages)
        self._test_case(messages, result)
