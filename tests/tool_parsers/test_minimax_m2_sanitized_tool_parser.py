# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.tool_parsers.minimax_m2_sanitized_tool_parser import (
    MinimaxM2SanitizedToolParser,
)


class FakeTokenizer:
    """Minimal fake tokenizer for unit tests."""

    def __init__(self):
        self.model_tokenizer = True
        self.vocab = {
            "<minimax:tool_call>": 1,
            "</minimax:tool_call>": 2,
        }

    def get_vocab(self):
        return self.vocab


def test_sanitized_tool_parser_normalizes_tool_arguments():
    parser = MinimaxM2SanitizedToolParser(FakeTokenizer())
    output = (
        '<minimax:tool_call><invoke name="read_file">'
        '<parameter name="path">scripts/monkey_character. gd</parameter>'
        "</invoke></minimax:tool_call>"
    )
    info = parser.extract_tool_calls(output, request=None)
    assert info.tools_called is True
    assert len(info.tool_calls) == 1
    assert json.loads(info.tool_calls[0].function.arguments) == {
        "path": "scripts/monkey_character.gd"
    }


def test_sanitized_tool_parser_normalizes_visible_content():
    parser = MinimaxM2SanitizedToolParser(FakeTokenizer())
    output = (
        "Reading scripts/monkey_character. gd\n"
        '<minimax:tool_call><invoke name="read_file">'
        '<parameter name="path">scripts/monkey_character. gd</parameter>'
        "</invoke></minimax:tool_call>"
    )
    info = parser.extract_tool_calls(output, request=None)
    assert info.content == "Reading scripts/monkey_character.gd\n"
