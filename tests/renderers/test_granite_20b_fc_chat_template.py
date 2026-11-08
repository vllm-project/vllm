# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the granite-20b-functioncalling chat template.

The template lifts the system message out of the conversation and renders it as
a ``SYSTEM:`` preamble, so the message loop has no branch for the system role
and reaching it raises. A conversation carrying a system message, which is the
usual shape from the OpenAI API, must still render.
"""

import json
from pathlib import Path

import jinja2
import jinja2.sandbox
import pytest

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "tool_chat_template_granite_20b_fc.jinja"
)

SYSTEM_PROMPT = "You are a terse assistant."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@pytest.fixture(scope="module")
def granite_template():
    """Compile the template in the environment chat templates render in."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )

    def raise_exception(message: str):
        raise jinja2.exceptions.TemplateError(message)

    env.filters["tojson"] = lambda obj, **kwargs: json.dumps(
        obj, ensure_ascii=False, **kwargs
    )
    env.globals["raise_exception"] = raise_exception
    return env.from_string(TEMPLATE_PATH.read_text())


def test_conversation_with_a_system_message_renders(granite_template):
    """A system message is rendered as the preamble, not fed to the loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is the weather in Paris?"},
    ]

    rendered = granite_template.render(
        messages=messages, tools=TOOLS, add_generation_prompt=True
    )

    assert f"SYSTEM: {SYSTEM_PROMPT}" in rendered
    # Once as the preamble, and never again from the message loop.
    assert rendered.count(SYSTEM_PROMPT) == 1
    assert "USER: What is the weather in Paris?" in rendered
    assert rendered.rstrip().endswith("ASSISTANT:")


def test_generation_prompt_is_emitted_without_a_user_turn(granite_template):
    """The assistant cue does not depend on the message loop running."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    rendered = granite_template.render(
        messages=messages, tools=TOOLS, add_generation_prompt=True
    )

    assert rendered.rstrip().endswith("ASSISTANT:")
