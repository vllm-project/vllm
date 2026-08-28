# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Test suite comparing legacy vLLM Gemma-4 tool call regex vs Google reference parser.

References:
    1. Google Gemma-4 Canonical Chat Template & Transformers Reference:
       ``transformers.models.gemma4`` & ``examples/tool_chat_template_gemma4.jinja``
       Specifies optional ``<|tool_call>`` and ``<tool_call|>`` delimiters:
       ``(?:<\|tool_call>)?call:(\w+)\{(.*?)\}(?:<tool_call\|>)?``
    2. vLLM Offline Parser Reference:
       ``vllm.tool_parsers.gemma4_utils.parse_tool_calls``
    3. Failure Trigger in Live Generation:
       When transitioning out of the reasoning channel (``<channel|>``), Gemma-4
       frequently emits bare ``call:func{...}`` directly without whitespace or
       ``<|tool_call>``. The legacy vLLM regex misses it, producing 0 tool calls
       and triggering runtime stream errors.
"""

import re
import pytest

# ---------------------------------------------------------------------------
# Regex Definitions
# ---------------------------------------------------------------------------

# Legacy vLLM parser regex (strictly requires <|tool_call> and <tool_call|>)
LEGACY_VLLM_REGEX = re.compile(
    r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)

# Google Canonical Specification (transformers.models.gemma4 / chat_template.jinja)
GOOGLE_CANONICAL_REGEX = re.compile(
    r"(?:<\|tool_call>)?call:([\w\-\.]+)\{(.*?)\}(?:<tool_call\|>)?",
    re.DOTALL,
)

# vLLM Offline gemma4_utils.py Tier 2 Fallback Pattern
VLLM_OFFLINE_TIER2_FALLBACK = re.compile(
    r"(?:<call>|(?:^|\s)call:)(\w+)\{(.*?)\}",
    re.DOTALL,
)

# Patched Unified Regex (handles canonical, bare call, turn boundary, and colon prefix)
PATCHED_UNIFIED_REGEX = re.compile(
    r"(?:<\|tool_call>:?)?call:([\w\-\.]+)\{(.*?)\}(?:<tool_call\|>|<turn\|>)?",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestGemma4RegexComparison:
    """Compare regex parsing behavior across standard and edge-case Gemma-4 outputs."""

    def test_canonical_tag_wrapped_call(self):
        """Case 1: Standard canonical format. All robust regexes succeed."""
        text = '<|tool_call>call:bash{command:<|"|>ls -la /work<|"|>}<tool_call|>'

        legacy_matches = LEGACY_VLLM_REGEX.findall(text)
        assert len(legacy_matches) == 1
        assert legacy_matches[0][0] == "bash"

        google_matches = GOOGLE_CANONICAL_REGEX.findall(text)
        assert len(google_matches) == 1
        assert google_matches[0][0] == "bash"

        patched_matches = PATCHED_UNIFIED_REGEX.findall(text)
        assert len(patched_matches) == 1
        assert patched_matches[0][0] == "bash"

    def test_bare_call_after_thought_channel(self):
        """Case 2: Bare call: immediately after thought channel exit (<channel|>call:...).
        
        Gemma-4 frequently drops <|tool_call> when transitioning from <channel|>.
        - Legacy vLLM regex: FAILS (0 matches -> fatal harness stream abort).
        - vLLM gemma4_utils Tier 2: FAILS (expects whitespace or start of string before 'call:').
        - Google Canonical regex: PASSES.
        - Patched Unified regex: PASSES.
        """
        text = (
            "<|channel>thought\n"
            "I need to read the entry source file to inspect the crash point.\n"
            "<channel|>"
            'call:bash{command:<|"|>cat /work/src/entry.c<|"|>}'
        )

        # 1. Legacy vLLM regex fails completely
        legacy_matches = LEGACY_VLLM_REGEX.findall(text)
        assert len(legacy_matches) == 0, "Legacy regex should fail on bare call"

        # 2. Offline tier 2 fails because <channel|> is attached without whitespace
        offline_matches = VLLM_OFFLINE_TIER2_FALLBACK.findall(text)
        assert len(offline_matches) == 0, "Offline regex requires whitespace before call:"

        # 3. Google canonical regex succeeds
        google_matches = GOOGLE_CANONICAL_REGEX.findall(text)
        assert len(google_matches) == 1
        assert google_matches[0][0] == "bash"
        assert '<|"|>cat /work/src/entry.c<|"|>' in google_matches[0][1]

        # 4. Patched unified regex succeeds directly
        patched_matches = PATCHED_UNIFIED_REGEX.findall(text)
        assert len(patched_matches) == 1
        assert patched_matches[0][0] == "bash"
        assert '<|"|>cat /work/src/entry.c<|"|>' in patched_matches[0][1]

    def test_turn_end_closure_instead_of_tool_call_end(self):
        """Case 3: Model terminates tool call with <turn|> rather than <tool_call|>.
        
        - Legacy vLLM regex: FAILS.
        - Google canonical regex: PASSES.
        - Patched unified regex: PASSES.
        """
        text = '<|tool_call>call:gdb{args:<|"|>-batch -ex run /tmp/poc<|"|>}<turn|>'

        legacy_matches = LEGACY_VLLM_REGEX.findall(text)
        assert len(legacy_matches) == 0

        google_matches = GOOGLE_CANONICAL_REGEX.findall(text)
        assert len(google_matches) == 1
        assert google_matches[0][0] == "gdb"

        patched_matches = PATCHED_UNIFIED_REGEX.findall(text)
        assert len(patched_matches) == 1
        assert patched_matches[0][0] == "gdb"

    def test_colon_prefixed_tool_call(self):
        """Case 4: Model emits <|tool_call>:call:... under attention stress.
        
        - Legacy vLLM regex: FAILS.
        - Google canonical regex: PASSES (bare call match).
        - Patched unified regex: PASSES.
        """
        text = '<|tool_call>:call:bash{command:<|"|>pytest<|"|>}'

        legacy_matches = LEGACY_VLLM_REGEX.findall(text)
        assert len(legacy_matches) == 0

        google_matches = GOOGLE_CANONICAL_REGEX.findall(text)
        assert len(google_matches) == 1
        assert google_matches[0][0] == "bash"

        patched_matches = PATCHED_UNIFIED_REGEX.findall(text)
        assert len(patched_matches) == 1
        assert patched_matches[0][0] == "bash"

    def test_multiple_sequential_tool_calls_mixed_syntax(self):
        """Case 5: Multi-tool call payload with mixed canonical and bare syntax."""
        text = (
            '<|tool_call>call:read_file{path:<|"|>/work/Makefile<|"|>}<tool_call|>\n'
            'call:bash{command:<|"|>make -j8<|"|>}'
        )

        legacy_matches = LEGACY_VLLM_REGEX.findall(text)
        # Legacy only catches the first one, dropping the second
        assert len(legacy_matches) == 1
        assert legacy_matches[0][0] == "read_file"

        # Patched catches both
        patched_matches = PATCHED_UNIFIED_REGEX.findall(text)
        assert len(patched_matches) == 2
        assert patched_matches[0][0] == "read_file"
        assert patched_matches[1][0] == "bash"


class TestGemma4EngineBareCallIntegration:
    """Validate that the engine and offline utils properly extract bare tool calls."""

    def test_offline_utils_bare_call_after_channel(self):
        """Offline parse_tool_calls should extract bare calls immediately following <channel|>."""
        from vllm.tool_parsers.gemma4_utils import parse_tool_calls

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
        """Gemma4EngineToolParser should transition and extract bare calls without dropping them."""
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
        mock_tokenizer.decode.side_effect = lambda ids: decode_map.get(ids[0], f"tok{ids[0]}")

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
