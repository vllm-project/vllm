# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM4 tool-only parser."""

import unittest
from unittest.mock import Mock

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers.glm4_tool_only_parser import (
    Glm4ToolOnlyParser)


class TestGlm4ToolOnlyParser(unittest.TestCase):

    def setUp(self):
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.vocab = {}

        self.parser = Glm4ToolOnlyParser(mock_tokenizer)
        self.request = ChatCompletionRequest(
            model="glm-4",
            messages=[],
            tools=[],
        )

    def test_multiple_tool_calls_streaming(self):
        """Test streaming multiple tool calls chunk by chunk."""
        # Simulate the streaming chunks from the example
        chunks = [
            "search",
            "_web",
            "\n",
            "<arg_key>",
            "query",
            "</arg_key>",
            "\n",
            "<arg_value>",
            "hot",
            "els",
            " near",
            " Union",
            " Square",
            " New",
            " York",
            " recommendations",
            " ",
            "202",
            "4",
            "</arg_value>",
            "\n",
            "</tool_call>",
            # Second tool call
            "<tool_call>",
            "get",
            "_weather",
            "\n",
            "<arg_key>",
            "location",
            "</arg_key>",
            "\n",
            "<arg_value>",
            "San Francisco",
            "</arg_value>",
            "\n",
            "</tool_call>",
        ]

        results = []
        previous_text = ""
        current_text = ""

        for chunk in chunks:
            previous_text = current_text
            current_text += chunk

            result = self.parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=self.request,
            )

            if result and result.tool_calls:
                for tool_call in result.tool_calls:
                    results.append(tool_call.model_dump())

        # Assert exact structure for all streaming results
        self.assertEqual(len(results), 7,
                         "Unexpected number of streaming results")

        # First tool call results
        self.assertDictEqual(
            results[0], {
                "id": "0",
                "type": "function",
                "index": 0,
                "function": {
                    "name": "search",
                    "arguments": None
                }
            })
        self.assertDictEqual(
            results[1], {
                "id": "0",
                "type": "function",
                "index": 0,
                "function": {
                    "name": "_web",
                    "arguments": None
                }
            })
        self.assertDictEqual(
            results[2],
            {
                "id": "0",
                "type": "function",
                "index": 0,
                "function": {
                    "name":
                    None,
                    "arguments":
                    ('{"query": "hotels near Union Square New York '
                     'recommendations 2024"}'),
                },
            },
        )

        # Empty chunk at beginning for second tool call
        self.assertDictEqual(
            results[3], {
                "id": "1",
                "type": "function",
                "index": 1,
                "function": {
                    "name": None,
                    "arguments": None
                }
            })
        # Second tool call results
        self.assertDictEqual(
            results[4], {
                "id": "1",
                "type": "function",
                "index": 1,
                "function": {
                    "name": "get",
                    "arguments": None
                }
            })
        self.assertDictEqual(
            results[5], {
                "id": "1",
                "type": "function",
                "index": 1,
                "function": {
                    "name": "_weather",
                    "arguments": None
                }
            })
        self.assertDictEqual(
            results[6],
            {
                "id": "1",
                "type": "function",
                "index": 1,
                "function": {
                    "name": None,
                    "arguments": '{"location": "San Francisco"}',
                },
            },
        )

    def test_single_tool_call_non_streaming(self):
        """Test non-streaming extraction of a single tool call."""
        # Model output without <tool_call> wrapper (template adds it)
        model_output = """search_web
<arg_key>query</arg_key>
<arg_value>hotels near Union Square New York recommendations 2024</arg_value>
</tool_call>"""

        result = self.parser.extract_tool_calls(model_output, self.request)

        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.content,
                         "")  # Tool-only mode never returns content

        tool_call = result.tool_calls[0]
        self.assertEqual(tool_call.function.name, "search_web")

        import json
        args = json.loads(tool_call.function.arguments)
        self.assertEqual(
            args["query"],
            "hotels near Union Square New York recommendations 2024")

    def test_multiple_tool_calls_non_streaming(self):
        """Test non-streaming extraction of multiple tool calls."""
        model_output = """search_web
<arg_key>query</arg_key>
<arg_value>restaurants NYC</arg_value>
</tool_call>
<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>San Francisco</arg_value>
</tool_call>"""

        result = self.parser.extract_tool_calls(model_output, self.request)

        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.content, "")

        import json

        # Check first tool call
        self.assertEqual(result.tool_calls[0].function.name, "search_web")
        args1 = json.loads(result.tool_calls[0].function.arguments)
        self.assertEqual(args1["query"], "restaurants NYC")

        # Check second tool call
        self.assertEqual(result.tool_calls[1].function.name, "get_weather")
        args2 = json.loads(result.tool_calls[1].function.arguments)
        self.assertEqual(args2["location"], "San Francisco")


if __name__ == "__main__":
    unittest.main()
