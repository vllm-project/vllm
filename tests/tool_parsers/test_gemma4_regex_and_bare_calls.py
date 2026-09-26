# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Behavioral tests for the offline Gemma-4 tool call parser.

Exercises ``vllm.tool_parsers.gemma4_utils.parse_tool_calls`` directly so
regressions in the production tiered regex are caught. Covers the known
Gemma-4 output variations:

    * canonical ``<|tool_call>call:name{...}<tool_call|>``
    * ``<turn|>`` used as the closing delimiter
    * bare ``call:name{...}`` right after a ``<channel|>`` transition (no
      whitespace)
    * colon-prefixed ``<|tool_call>:call:name{...}``
    * hyphen- and dot-containing tool names in the bare fallback

References:
    1. Google Gemma-4 canonical chat template / transformers reference.
    2. vLLM offline parser: ``vllm.tool_parsers.gemma4_utils.parse_tool_calls``.
"""

from vllm.tool_parsers.gemma4_utils import parse_tool_calls


class TestGemma4CanonicalFormats:
    """Tier 1: standard ``<|tool_call>`` delimited calls."""

    def test_canonical_tag_wrapped_call(self):
        text = '<|tool_call>call:bash{command:<|"|>ls -la /work<|"|>}<tool_call|>'

        tool_calls = parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "bash"
        assert tool_calls[0]["arguments"] == {"command": "ls -la /work"}

    def test_turn_end_closure_instead_of_tool_call_end(self):
        """Gemma-4 sometimes closes the call with ``<turn|>``."""
        text = '<|tool_call>call:gdb{args:<|"|>-batch -ex run /tmp/poc<|"|>}<turn|>'

        tool_calls = parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "gdb"
        assert tool_calls[0]["arguments"] == {"args": "-batch -ex run /tmp/poc"}

    def test_multiple_sequential_canonical_calls(self):
        text = (
            '<|tool_call>call:read_file{path:<|"|>/work/Makefile<|"|>}<tool_call|>\n'
            '<|tool_call>call:bash{command:<|"|>make -j8<|"|>}<tool_call|>'
        )

        tool_calls = parse_tool_calls(text)

        assert [tc["name"] for tc in tool_calls] == ["read_file", "bash"]

    def test_strict_mode_ignores_fallback_formats(self):
        text = '<channel|>call:bash{command:<|"|>ls<|"|>}'

        assert parse_tool_calls(text, strict=True) == []
        assert len(parse_tool_calls(text)) == 1


class TestGemma4FallbackFormats:
    """Tier 2: bare ``call:`` variations emitted when the model drops
    ``<|tool_call>``."""

    def test_bare_call_after_thought_channel(self):
        """Bare ``call:`` glued to ``<channel|>`` with no whitespace.

        This is the regression the fallback tier exists for: the legacy
        strict regex produced 0 tool calls here and aborted the stream.
        """
        text = (
            "<|channel>thought\n"
            "I need to read the entry source file to inspect the crash point.\n"
            "<channel|>"
            'call:bash{command:<|"|>cat /work/src/entry.c<|"|>}'
        )

        tool_calls = parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "bash"
        assert tool_calls[0]["arguments"] == {"command": "cat /work/src/entry.c"}

    def test_colon_prefixed_tool_call(self):
        """``<|tool_call>:call:...`` — stray colon after the open tag."""
        text = '<|tool_call>:call:bash{command:<|"|>pytest<|"|>}'

        tool_calls = parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "bash"
        assert tool_calls[0]["arguments"] == {"command": "pytest"}

    def test_bare_call_with_leading_whitespace(self):
        text = 'some prose\ncall:bash{command:<|"|>make -j8<|"|>}'

        tool_calls = parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "bash"
        assert tool_calls[0]["arguments"] == {"command": "make -j8"}

    def test_fragmented_call_tag_from_multimodal(self):
        """``<call>name{...}`` — fragmented special token from MM inputs."""
        text = '<call>get_weather{city:<|"|>Paris<|"|>}'

        tool_calls = parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["arguments"] == {"city": "Paris"}

    def test_hyphenated_tool_name(self):
        text = 'call:web-search{query:<|"|>gemma 4 release<|"|>}'

        tool_calls = parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "web-search"
        assert tool_calls[0]["arguments"] == {"query": "gemma 4 release"}

    def test_dotted_tool_name(self):
        text = 'call:fs.read_file{path:<|"|>/work/Makefile<|"|>}'

        tool_calls = parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "fs.read_file"
        assert tool_calls[0]["arguments"] == {"path": "/work/Makefile"}

    def test_no_tool_call_returns_empty(self):
        text = "Just some reasoning text with no call in it."

        assert parse_tool_calls(text) == []

    def test_nested_object_argument_not_truncated(self):
        """A nested ``{...}`` argument must not stop at its own closing brace."""
        text = 'call:tool{config:{mode:<|"|>x<|"|>}}'

        tool_calls = parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "tool"
        # _parse_tool_arguments stringifies nested values; the important
        # thing is "mode" survives instead of being truncated to "{}".
        assert tool_calls[0]["arguments"] == {"config": "{'mode': 'x'}"}

    def test_truncated_call_is_rejected(self):
        """A call cut off before its closing brace must not be emitted."""
        text = "call:add{a:1,b:2"

        assert parse_tool_calls(text) == []

    def test_nested_call_like_text_is_not_a_second_call(self):
        """Argument text that looks like ``call:name{...}`` isn't a real call."""
        text = 'call:outer{payload:call:inner{x:<|"|>y<|"|>}}'

        tool_calls = parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "outer"


class TestGemma4EngineBareCallIntegration:
    """Validate that the engine and offline utils properly extract bare tool calls."""

    def test_offline_utils_bare_call_after_channel(self):
        """parse_tool_calls extracts bare calls immediately following <channel|>."""
        text = (
            "<|channel>thought\n"
            "I need to read the entry source file.\n"
            "<channel|>"
            'call:bash{command:<|"|>cat /work/src/entry.c<|"|>}'
        )
        tool_calls = parse_tool_calls(text)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "bash"
        assert tool_calls[0]["arguments"] == {"command": "cat /work/src/entry.c"}

    def test_engine_tool_parser_bare_call_extraction(self):
        """Gemma4EngineToolParser extracts bare calls without dropping them."""
        import json
        from unittest.mock import MagicMock

        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
            ChatCompletionToolsParam,
        )
        from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            )
        ]
        vocab = {
            "<|tool_call>": 48,
            "<tool_call|>": 49,
            "<|channel>": 50,
            "<channel|>": 51,
        }
        decode_map = {v: k for k, v in vocab.items()}
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = vocab
        mock_tokenizer.decode.side_effect = lambda ids: decode_map.get(
            ids[0], f"tok{ids[0]}"
        )

        parser = Gemma4EngineToolParser(mock_tokenizer, tools=tools)
        request = MagicMock(spec=ChatCompletionRequest)
        request.tools = tools
        request.tool_choice = "auto"

        # Model output containing channel thought and immediate bare call
        model_output = (
            "<|channel>thought\n"
            "Listing directory files.\n"
            "<channel|>"
            'call:bash{command:<|"|>ls -la /work<|"|>}'
        )

        result = parser.extract_tool_calls(model_output, request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "bash"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"command": "ls -la /work"}
