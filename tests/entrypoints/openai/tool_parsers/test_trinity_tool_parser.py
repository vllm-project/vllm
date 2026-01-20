# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.entrypoints.openai.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tool_parsers import ToolParser, ToolParserManager


@pytest.fixture
def trinity_tokenizer() -> TokenizerLike:
    return get_tokenizer("arcee-ai/Trinity-Large-Preview")


@pytest.fixture
def trinity_parser(trinity_tokenizer: TokenizerLike) -> ToolParser:
    return ToolParserManager.get_tool_parser("trinity")(trinity_tokenizer)


THINK_TOOL_CALL_OUTPUT = """<think>internal reasoning</think>
<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""

THINK_WRAPPED_TOOL_CALL_OUTPUT = """<think>internal reasoning
<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>
more reasoning</think>"""

THINK_NO_TOOL_CALL_OUTPUT = "<think>internal reasoning</think> done"

FINAL_ANSWER_FUNCTION_CALL = FunctionCall(
    name="final_answer",
    arguments='{"trigger": true}',
)


@pytest.mark.parametrize("streaming", [True, False])
def test_trinity_parser_think_tool_call(
    streaming: bool,
    trinity_parser: ToolParser,
) -> None:
    content, tool_calls = run_tool_extraction(
        trinity_parser, THINK_TOOL_CALL_OUTPUT, streaming=streaming
    )

    assert content == "internal reasoning"
    assert len(tool_calls) == 1
    assert tool_calls[0].function == FINAL_ANSWER_FUNCTION_CALL


@pytest.mark.parametrize("streaming", [True, False])
def test_trinity_parser_tool_call_within_think_tags(
    streaming: bool,
    trinity_parser: ToolParser,
) -> None:
    content, tool_calls = run_tool_extraction(
        trinity_parser, THINK_WRAPPED_TOOL_CALL_OUTPUT, streaming=streaming
    )

    assert content == "internal reasoning"
    assert len(tool_calls) == 1
    assert tool_calls[0].function == FINAL_ANSWER_FUNCTION_CALL


@pytest.mark.parametrize("streaming", [True, False])
def test_trinity_parser_think_no_tool_call(
    streaming: bool,
    trinity_parser: ToolParser,
) -> None:
    content, tool_calls = run_tool_extraction(
        trinity_parser, THINK_NO_TOOL_CALL_OUTPUT, streaming=streaming
    )

    assert content == "internal reasoning done"
    assert tool_calls == []
