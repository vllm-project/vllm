# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.hunyuan_a13b_reasoning_parser import HunyuanA13BReasoningParser

parser_name = "hunyuan_a13b"
START_REASONING = "<think>\n"
START_RESPONSE = "\n</think>\n<answer>\n"
END_RESPONSE = "\n</answer>"

NO_REASONING_QUICK_THOUGHT = {
    "output": f"{START_REASONING}{START_RESPONSE}This is the rest{END_RESPONSE}",  # noqa: E501
    "reasoning": None,
    "content": "This is the rest",
}

SIMPLE_REASONING = {
    "output": f"{START_REASONING}This is a reasoning section{START_RESPONSE}This is the rest{END_RESPONSE}",  # noqa: E501
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
COMPLETE_REASONING = {
    "output": f"{START_REASONING}This is a reasoning section{START_RESPONSE}",
    "reasoning": "This is a reasoning section",
    "content": None,
}

COMPLETE_REASONING_WITH_SYMBOL = {
    "output": f"{START_REASONING}This is a reasoning section!{START_RESPONSE}",
    "reasoning": "This is a reasoning section!",
    "content": None,
}
NO_REASONING = {
    "output": "This is content",
    "reasoning": None,
    "content": "This is content",
}
MULTIPLE_LINES = {
    "output": f"{START_REASONING}This\nThat{START_RESPONSE}This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
}
REASONING_WITH_THINK = {
    "output": f"{START_REASONING}This is a reasoning section{START_RESPONSE}This is the rest",  # noqa: E501
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
COMPLETE_REASONING_WITH_THINK = {
    "output": f"{START_REASONING}This is a reasoning section{START_RESPONSE}",
    "reasoning": "This is a reasoning section",
    "content": None,
}
MULTIPLE_LINES_WITH_THINK = {
    "output": f"{START_REASONING}This\nThat{START_RESPONSE}This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING_WITH_SYMBOL,
        id="complete_reasoning_with_symbol",
    ),
    pytest.param(
        False,
        NO_REASONING,
        id="no_reasoning",
    ),
    pytest.param(False, NO_REASONING_QUICK_THOUGHT, id="no_reasoning_quick"),
    pytest.param(
        False,
        MULTIPLE_LINES,
        id="multiple_lines",
    ),
    pytest.param(
        False,
        REASONING_WITH_THINK,
        id="reasoning_with_think",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think",
    ),
    pytest.param(
        True,
        SIMPLE_REASONING,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_streaming",
    ),
    pytest.param(
        True,
        NO_REASONING,
        id="no_reasoning_streaming",
    ),
    pytest.param(True, NO_REASONING_QUICK_THOUGHT, id="no_reasoning_quick_stream"),
    pytest.param(
        True,
        MULTIPLE_LINES,
        id="multiple_lines_streaming",
    ),
    pytest.param(
        True,
        REASONING_WITH_THINK,
        id="reasoning_with_think_streaming",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think_streaming",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think_streaming",
    ),
]

# Global tokenizer initialization to avoid repeated loading
tokenizer = AutoTokenizer.from_pretrained(
    "tencent/Hunyuan-A13B-Instruct", trust_remote_code=True
)


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
):
    output = tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


class _HunyuanReasoningOnlyParser(DelegatingParser):
    reasoning_parser_cls = HunyuanA13BReasoningParser
    tool_parser_cls = None


def test_delegating_streaming_strips_answer_end():
    """Regression for #58127: DelegatingParser must not leak </answer>.

    ``is_reasoning_end`` flips True when the answer section starts so structured
    output can constrain content, but the closing ``\\n</answer>`` must still be
    consumed by the Hunyuan state machine instead of passthrough.
    """
    text = f"{START_REASONING}ink{START_RESPONSE}answer{END_RESPONSE}"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    parser = _HunyuanReasoningOnlyParser(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for tid in token_ids:
        delta_text = tokenizer.decode([tid])
        delta = parser.parse_delta(
            delta_text,
            [tid],
            request,
            finished=False,
        )
        if delta and delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta and delta.content:
            content_parts.append(delta.content)

    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)
    assert reasoning == "ink"
    assert content == "answer"
    assert "</answer>" not in content
    assert END_RESPONSE not in content
