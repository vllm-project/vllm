# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Gemma4 chat template rendering and invariants."""

from pathlib import Path
from typing import Any

import jinja2.sandbox
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


SUPPORTED_CASES = {
    **BASIC_CASES,
    **TOOL_CALL_CASES,
    **PARALLEL_TOOL_CALL_CASES,
    **REASONING_CASES,
    **TOOL_CALL_W_REASONING_CASES,
    **PARALLEL_TOOL_CALL_W_REASONING_CASES,
}


class TestGemma4ChatTemplateInvariants(TestChatTemplateInvariants):
    turn_delimiter = ("<|turn>", "<turn|>")
    reasoning_delimiter = ("<|channel>", "<channel|>")
    tool_call_delimiter = ("<|tool_call>", "<tool_call|>")
    tool_response_delimiter = ("<|tool_response>", "<tool_response|>")

    @classmethod
    def _build_markers(cls, messages: list[dict[str, Any]]) -> list[str]:
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
            msg_role = msg.get("role")
            if msg_role == "tool":
                markers.append(tool_response_start)
            elif msg_role != "assistant":
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

            if msg_role == "tool":
                markers.append(tool_response_end)
            elif msg_role != "assistant":
                markers.append(turn_end)

        return markers

    @classmethod
    def _check_delimiters(cls, messages: list[dict[str, Any]], result: str):
        turn_state = delimiter_state(result, cls.turn_delimiter)

        last_msg_role = messages[-1].get("role")
        last_msg_tool_calls = messages[-1].get("tool_calls")
        if (
            last_msg_role == "tool"
            or last_msg_role == "assistant"
            and last_msg_tool_calls
        ):
            # Gemma 4 keeps the final model turn open across tool-call flows.
            assert turn_state == 1
        else:
            assert turn_state == 0

        reasoning_state = delimiter_state(result, cls.reasoning_delimiter)
        assert reasoning_state == 0

        tool_call_state = delimiter_state(result, cls.tool_call_delimiter)
        assert tool_call_state == 0

        tool_response_state = delimiter_state(result, cls.tool_response_delimiter)
        if last_msg_tool_calls:
            # Gemma 4 inserts a tool_call end marker after a tool call
            assert tool_response_state == 1
        else:
            assert tool_response_state == 0

    @pytest.mark.skip(
        reason="Temporarily disabled until gemma 4 chat template is updated"
    )
    @pytest.mark.parametrize(
        "test_case",
        SUPPORTED_CASES.values(),
        ids=SUPPORTED_CASES.keys(),
    )
    def test_invariants(
        self,
        gemma4_template,
        test_case,
    ):
        messages = create_conversation(*test_case)
        result = _render(gemma4_template, messages)
        self._test_case(messages, result)
