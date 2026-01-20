# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple standalone tests for Trinity tool parser logic.
These tests verify the core parsing without requiring the full vllm stack.
"""

import json
import re

import pytest


class SimpleTrinityParser:
    """Minimal recreation of Trinity parsing logic for testing."""

    def __init__(self):
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        self.tool_call_regex = re.compile(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
        )

    def _strip_think_tags(self, text: str) -> str:
        return text.replace(self.think_start_token, "").replace(
            self.think_end_token, ""
        )

    def extract_tool_calls(self, model_output: str) -> dict:
        cleaned_output = self._strip_think_tags(model_output)

        if self.tool_call_start_token not in cleaned_output:
            return {
                "tools_called": False,
                "tool_calls": [],
                "content": cleaned_output,
            }

        tool_call_json_list = self.tool_call_regex.findall(cleaned_output)
        tool_calls = []
        for tool_call_json in tool_call_json_list:
            tool_call_dict = json.loads(tool_call_json)
            args_str = json.dumps(
                tool_call_dict.get("arguments", {}), ensure_ascii=False
            )
            tool_calls.append(
                {
                    "name": tool_call_dict.get("name", ""),
                    "arguments": args_str,
                }
            )

        content_idx = cleaned_output.find(self.tool_call_start_token)
        content = cleaned_output[:content_idx].strip()

        return {
            "tools_called": len(tool_calls) > 0,
            "tool_calls": tool_calls,
            "content": content if content else None,
        }


@pytest.fixture
def parser():
    return SimpleTrinityParser()


class TestTrinityParserLogic:
    """Test the core Trinity parsing logic."""

    def test_think_tags_followed_by_tool_call(self, parser):
        """Think tags followed by tool call should extract both."""
        output = """<think>internal reasoning</think>
<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""

        result = parser.extract_tool_calls(output)

        assert result["tools_called"] is True
        assert result["content"] == "internal reasoning"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "final_answer"
        assert result["tool_calls"][0]["arguments"] == '{"trigger": true}'

    def test_tool_call_wrapped_inside_think_tags(self, parser):
        """Tool call inside think tags should still be extracted."""
        output = """<think>internal reasoning
<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>
more reasoning</think>"""

        result = parser.extract_tool_calls(output)

        assert result["tools_called"] is True
        assert result["content"] == "internal reasoning"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "final_answer"

    def test_no_tool_call(self, parser):
        """Output with only think tags and no tool call."""
        output = "<think>internal reasoning</think> done"

        result = parser.extract_tool_calls(output)

        assert result["tools_called"] is False
        assert result["tool_calls"] == []
        assert "internal reasoning" in result["content"]
        assert "done" in result["content"]

    def test_multiple_tool_calls(self, parser):
        """Multiple tool calls should all be extracted."""
        output = """<think>thinking</think>
<tool_call>
{"name": "search", "arguments": {"query": "test"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"expr": "2+2"}}
</tool_call>"""

        result = parser.extract_tool_calls(output)

        assert result["tools_called"] is True
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["name"] == "search"
        assert result["tool_calls"][1]["name"] == "calculate"

    def test_plain_tool_call_without_think_tags(self, parser):
        """Tool call without think tags should work."""
        output = """Some content
<tool_call>
{"name": "get_weather", "arguments": {"city": "NYC"}}
</tool_call>"""

        result = parser.extract_tool_calls(output)

        assert result["tools_called"] is True
        assert result["content"] == "Some content"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "get_weather"

    def test_empty_arguments(self, parser):
        """Tool call with empty arguments."""
        output = """<tool_call>
{"name": "no_args", "arguments": {}}
</tool_call>"""

        result = parser.extract_tool_calls(output)

        assert result["tools_called"] is True
        assert result["tool_calls"][0]["name"] == "no_args"
        assert result["tool_calls"][0]["arguments"] == "{}"

    def test_nested_json_arguments(self, parser):
        """Tool call with nested JSON arguments."""
        output = """<tool_call>
{"name": "complex", "arguments": {"nested": {"a": 1, "b": [1, 2, 3]}}}
</tool_call>"""

        result = parser.extract_tool_calls(output)

        assert result["tools_called"] is True
        assert result["tool_calls"][0]["name"] == "complex"
        args = json.loads(result["tool_calls"][0]["arguments"])
        assert args["nested"]["a"] == 1
        assert args["nested"]["b"] == [1, 2, 3]

    def test_plain_text_no_tags(self, parser):
        """Plain text with no tags at all."""
        output = "Just some regular text without any special tags."

        result = parser.extract_tool_calls(output)

        assert result["tools_called"] is False
        assert result["content"] == output
        assert result["tool_calls"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
