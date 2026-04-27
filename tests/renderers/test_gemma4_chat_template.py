# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Gemma4 chat template rendering."""

from pathlib import Path

import jinja2.sandbox
import pytest

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "tool_chat_template_gemma4.jinja"
)


@pytest.fixture(scope="module")
def gemma4_template():
    """Load and compile the Gemma4 chat template."""
    template_str = TEMPLATE_PATH.read_text()
    env = jinja2.sandbox.ImmutableSandboxedEnvironment()
    return env.from_string(template_str)


def _render(template, messages, **kwargs):
    """Render the template with sensible defaults."""
    kwargs.setdefault("bos_token", "<bos>")
    kwargs.setdefault("add_generation_prompt", False)
    return template.render(messages=messages, **kwargs)


class TestGemma4ChatTemplate:
    def test_basic_multiturn_thinking_disabled(self, gemma4_template):
        """With enable_thinking=False (default), generation prompt ends with
        an empty thought channel to suppress thinking."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _render(gemma4_template, messages, add_generation_prompt=True)
        assert "<|turn>user\n" in result
        assert "<|turn>model\n" in result
        assert "Hello" in result
        assert "Hi there!" in result
        assert "How are you?" in result
        assert result.rstrip("\n").endswith("<|channel>thought\n<channel|>")

    def test_basic_multiturn_thinking_enabled(self, gemma4_template):
        """With enable_thinking=True, generation prompt ends with model
        turn opener (no thought suppression)."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _render(
            gemma4_template,
            messages,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        assert "<|turn>user\n" in result
        assert "<|turn>model\n" in result
        assert "Hello" in result
        assert "Hi there!" in result
        assert "How are you?" in result
        assert result.rstrip("\n").endswith("<|turn>model")

    def test_system_message(self, gemma4_template):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = _render(gemma4_template, messages)
        assert "<|turn>system\n" in result
        assert "You are helpful." in result

    def test_thinking_enabled(self, gemma4_template):
        messages = [{"role": "user", "content": "Think about this"}]
        result = _render(
            gemma4_template,
            messages,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        assert "<|think|>" in result
        assert "<|turn>system\n" in result

    def test_tool_declarations(self, gemma4_template):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "What is the weather?"}]
        result = _render(
            gemma4_template,
            messages,
            tools=tools,
            add_generation_prompt=True,
        )
        assert "<|tool>" in result
        assert "declaration:get_weather" in result
        assert "<tool|>" in result
        assert '<|"|>City name<|"|>' in result

    def test_tool_calls_in_assistant(self, gemma4_template):
        messages = [
            {"role": "user", "content": "Weather in London?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            },
        ]
        result = _render(gemma4_template, messages)
        assert "<|tool_call>call:get_weather{" in result
        assert "}<tool_call|>" in result
        assert '<|"|>London<|"|>' in result

    def test_tool_responses_openai_style(self, gemma4_template):
        """role='tool' messages are formatted as <|tool_response> blocks
        with content dumped as-is."""
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temperature": 15, "condition": "sunny"}',
            },
        ]
        result = _render(gemma4_template, messages, add_generation_prompt=True)
        assert "<|tool_response>" in result
        assert "response:get_weather{" in result
        assert "<tool_response|>" in result
        assert '"temperature": 15' in result

    def test_tool_responses_legacy_style(self, gemma4_template):
        """tool_responses embedded on the assistant message."""
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
                "tool_responses": [
                    {
                        "name": "get_weather",
                        "response": {"temperature": 20},
                    }
                ],
            },
        ]
        result = _render(gemma4_template, messages)
        assert "<|tool_response>" in result
        assert "response:get_weather{" in result
        assert "temperature:" in result

    def test_generation_prompt_not_after_tool_response(self, gemma4_template):
        """add_generation_prompt=True should NOT add <|turn>model when the
        last message type was tool_response (the model turn continues)."""
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "sunny",
            },
        ]
        result = _render(gemma4_template, messages, add_generation_prompt=True)
        assert not result.strip().endswith("<|turn>model\n")

    def test_reasoning_in_tool_chains(self, gemma4_template):
        """reasoning field on assistant with tool_calls after last user
        message emits <|channel>thought\\n...<channel|>."""
        messages = [
            {"role": "user", "content": "Calculate something"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "Let me think about this...",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": {"expr": "2+2"},
                        },
                    }
                ],
            },
        ]
        result = _render(gemma4_template, messages)
        assert "<|channel>thought\n" in result
        assert "Let me think about this..." in result
        assert "<channel|>" in result

    def test_reasoning_not_before_last_user(self, gemma4_template):
        """reasoning on assistant BEFORE the last user message is dropped."""
        messages = [
            {"role": "user", "content": "First"},
            {
                "role": "assistant",
                "content": "Response",
                "reasoning": "Old reasoning that should be dropped",
                "tool_calls": [
                    {
                        "function": {
                            "name": "fn",
                            "arguments": {},
                        },
                    }
                ],
            },
            {"role": "user", "content": "Second"},
        ]
        result = _render(gemma4_template, messages, add_generation_prompt=True)
        assert "Old reasoning" not in result

    def test_strip_thinking_in_model_content(self, gemma4_template):
        """<|channel>...<channel|> in model content is stripped by the
        strip_thinking macro."""
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": ("<|channel>internal thought<channel|>Visible answer"),
            },
        ]
        result = _render(gemma4_template, messages)
        assert "internal thought" not in result
        assert "Visible answer" in result

    def test_multi_turn_tool_chain(self, gemma4_template):
        """assistant->tool->assistant->tool produces exactly one
        <|turn>model (later assistants continue the same turn)."""
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "step1", "arguments": {}},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "function": {"name": "step2", "arguments": {}},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "result2"},
        ]
        result = _render(gemma4_template, messages, add_generation_prompt=True)
        assert result.count("<|turn>model\n") == 1

    def test_format_argument_types(self, gemma4_template):
        """Strings wrapped in <|"|>, booleans as true/false, numbers bare."""
        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "test_fn",
                            "arguments": {
                                "name": "Alice",
                                "active": True,
                                "count": 42,
                            },
                        },
                    }
                ],
            },
        ]
        result = _render(gemma4_template, messages)
        assert '<|"|>Alice<|"|>' in result
        assert "active:true" in result
        assert "count:42" in result
