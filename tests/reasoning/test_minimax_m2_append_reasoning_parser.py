# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "minimax_m2_append_think"
end_token = "</think>"

# MiniMax M2 model path
REASONING_MODEL_NAME = "MiniMaxAI/MiniMax-M2"


@pytest.fixture(scope="module")
def minimax_m2_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


# =============================================================================
# MiniMaxM2AppendThinkReasoningParser behavior:
# - Prepends <think> to the beginning of the output
# - Does NOT separate reasoning and content
# - Returns everything as content (with <think> prepended)
# - reasoning is always None
#
# This parser is used when you want to keep the raw output with <think> added
# =============================================================================

# Case: simple output with end token
SIMPLE_OUTPUT = {
    "output": "This is reasoning</think>This is response",
    "reasoning": None,
    "content": "<think>This is reasoning</think>This is response",
    "is_reasoning_end": True,
}

# Case: output without end token (reasoning in progress)
NO_END_TOKEN = {
    "output": "This is reasoning in progress",
    "reasoning": None,
    "content": "<think>This is reasoning in progress",
    "is_reasoning_end": False,
}

# Case: only end token
ONLY_END_TOKEN = {
    "output": "</think>This is response",
    "reasoning": None,
    "content": "<think></think>This is response",
    "is_reasoning_end": True,
}

# Case: multiple lines
MULTIPLE_LINES = {
    "output": "Line 1\nLine 2</think>Response 1\nResponse 2",
    "reasoning": None,
    "content": "<think>Line 1\nLine 2</think>Response 1\nResponse 2",
    "is_reasoning_end": True,
}

# Case: empty output (non-streaming prepends <think>)
EMPTY = {
    "output": "",
    "reasoning": None,
    "content": "<think>",
    "is_reasoning_end": False,
}

# Case: empty output streaming (no tokens = no output)
EMPTY_STREAMING = {
    "output": "",
    "reasoning": None,
    "content": None,
    "is_reasoning_end": False,
}

# Case: special characters
SPECIAL_CHARS = {
    "output": "Let me think... 1+1=2</think>Yes!",
    "reasoning": None,
    "content": "<think>Let me think... 1+1=2</think>Yes!",
    "is_reasoning_end": True,
}

# Case: code in output
CODE_OUTPUT = {
    "output": "```python\nprint('hi')\n```</think>Here's the code.",
    "reasoning": None,
    "content": "<think>```python\nprint('hi')\n```</think>Here's the code.",
    "is_reasoning_end": True,
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_OUTPUT,
        id="simple_output",
    ),
    pytest.param(
        True,
        SIMPLE_OUTPUT,
        id="simple_output_streaming",
    ),
    pytest.param(
        False,
        NO_END_TOKEN,
        id="no_end_token",
    ),
    pytest.param(
        True,
        NO_END_TOKEN,
        id="no_end_token_streaming",
    ),
    pytest.param(
        False,
        ONLY_END_TOKEN,
        id="only_end_token",
    ),
    pytest.param(
        True,
        ONLY_END_TOKEN,
        id="only_end_token_streaming",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES,
        id="multiple_lines",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES,
        id="multiple_lines_streaming",
    ),
    pytest.param(
        False,
        EMPTY,
        id="empty",
    ),
    pytest.param(
        True,
        EMPTY_STREAMING,
        id="empty_streaming",
    ),
    pytest.param(
        False,
        SPECIAL_CHARS,
        id="special_chars",
    ),
    pytest.param(
        True,
        SPECIAL_CHARS,
        id="special_chars_streaming",
    ),
    pytest.param(
        False,
        CODE_OUTPUT,
        id="code_output",
    ),
    pytest.param(
        True,
        CODE_OUTPUT,
        id="code_output_streaming",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    minimax_m2_tokenizer,
):
    output = minimax_m2_tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        minimax_m2_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        minimax_m2_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = minimax_m2_tokenizer.convert_tokens_to_ids(output)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]


# ---------------------------------------------------------------------------
# Composition with the minimax_m2 tool parser (#58486): append mode keeps
# reasoning markup inside content, so the composed tool parser must pass
# </think> through instead of eating it as a reasoning boundary.
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Offline stand-in carrying the special tokens both parsers lex."""

    def __init__(self):
        self.vocab = {
            "<think>": 1,
            "</think>": 2,
            "<minimax:tool_call>": 3,
            "</minimax:tool_call>": 4,
        }

    def get_vocab(self):
        return self.vocab

    def decode(self, token_ids):
        id_to_token = {v: k for k, v in self.vocab.items()}
        return "".join(id_to_token.get(token_id, "") for token_id in token_ids)


def test_append_think_tool_composition_keeps_closing_delimiter():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.parser_manager import ParserManager

    model_output = (
        "I should add them.</think><minimax:tool_call>"
        '<invoke name="add"><parameter name="a">3</parameter>'
        '<parameter name="b">5</parameter></invoke></minimax:tool_call>'
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            },
        }
    ]
    parser_cls = ParserManager.get_parser(
        tool_parser_name="minimax_m2",
        reasoning_parser_name=parser_name,
        enable_auto_tools=True,
    )
    parser = parser_cls(FakeTokenizer(), tools)
    request = ChatCompletionRequest.model_validate(
        {
            "model": "MiniMaxAI/MiniMax-M2",
            "messages": [{"role": "user", "content": "3 + 5?"}],
            "tools": tools,
            "tool_choice": "auto",
        }
    )

    reasoning, content = parser.extract_reasoning(model_output, request)
    assert reasoning is None
    tool_calls, content = parser._extract_tool_calls(
        content, request, enable_auto_tools=True
    )

    assert content == "<think>I should add them.</think>"
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].name == "add"
    assert json.loads(tool_calls[0].arguments) == {"a": 3, "b": 5}
