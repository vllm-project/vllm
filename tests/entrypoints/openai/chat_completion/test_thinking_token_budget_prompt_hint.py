# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.render import serving as render_serving


def test_thinking_budget_prompt_hint_prepends_system_message():
    messages = [{"role": "user", "content": "Solve 2+2."}]

    result = render_serving._with_thinking_budget_system_prompt(messages, 50)

    assert result == [
        {
            "role": "system",
            "content": (
                "Think step by step and use fewer than 50 reasoning tokens "
                "before the final answer."
            ),
        },
        {"role": "user", "content": "Solve 2+2."},
    ]
    assert messages == [{"role": "user", "content": "Solve 2+2."}]


def test_thinking_budget_prompt_hint_extends_leading_system_message():
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Solve 2+2."},
    ]

    result = render_serving._with_thinking_budget_system_prompt(messages, 100)

    assert result[0]["role"] == "system"
    assert result[0]["content"] == (
        "You are concise.\n\n"
        "Think step by step and use fewer than 100 reasoning tokens "
        "before the final answer."
    )
    assert result[1] == {"role": "user", "content": "Solve 2+2."}


@pytest.mark.asyncio
async def test_render_chat_injects_thinking_budget_prompt_hint():
    captured_messages = []

    async def preprocess_chat(_request, messages, **_kwargs):
        captured_messages.append(messages)
        return messages, []

    serving = SimpleNamespace(
        renderer=SimpleNamespace(tokenizer=None),
        parser=None,
        use_harmony=False,
        enable_auto_tools=False,
        exclude_tools_when_tool_choice_none=False,
        chat_template=None,
        chat_template_content_format="auto",
        default_chat_template_kwargs={},
        trust_request_chat_template=False,
        validate_chat_template=lambda **_kwargs: None,
        preprocess_chat=preprocess_chat,
    )
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Solve 2+2."}],
        thinking_token_budget=50,
    )

    await render_serving.OpenAIServingRender.render_chat(serving, request)

    assert captured_messages == [
        [
            {
                "role": "system",
                "content": (
                    "Think step by step and use fewer than 50 reasoning tokens "
                    "before the final answer."
                ),
            },
            {"role": "user", "content": "Solve 2+2."},
        ]
    ]
