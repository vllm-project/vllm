# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

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

def test_streaming_does_not_leak_closing_answer_marker():
    """Regression test for issue #58127.

    The outer streaming loop only keeps calling
    ``extract_reasoning_streaming`` while ``is_reasoning_end_streaming``
    reports reasoning is still open. For the Hunyuan A13B three-state envelope
    (``<think>...</think>\\n<answer>...</answer>``), the reasoning hand-off used
    to fire as soon as the state machine entered the ``response`` state (right
    after ``<answer>``). That stopped feeding the parser before it could consume
    the closing ``\\n</answer>`` marker, which was then passed through verbatim
    into the streamed ``content``.
    """
    # Exact token id sequence reported in the issue, decoding to:
    #   <think>\nink\n</think>\n<answer>\n</answer>
    token_ids = [
        14023, 771, 397, 771, 198, 524, 27963, 397, 27, 9399, 397,
        198, 524, 9399, 29,
    ]
    parser = ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer)

    # Mirror DelegatingParser.parse_delta(): only feed the reasoning parser
    # while reasoning is still open; afterwards deltas pass straight to content.
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    reasoning_ended = False
    current_ids: list[int] = []
    for tid in token_ids:
        delta_text = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens([tid])
        )
        current_ids.append(tid)
        if not reasoning_ended:
            delta_message = parser.extract_reasoning_streaming(
                previous_text="",
                current_text="".join(content_parts),
                delta_text=delta_text,
                previous_token_ids=current_ids[:-1],
                current_token_ids=current_ids,
                delta_token_ids=[tid],
            )
            if delta_message is not None:
                if delta_message.reasoning:
                    reasoning_parts.append(delta_message.reasoning)
                if delta_message.content:
                    content_parts.append(delta_message.content)
            if parser.is_reasoning_end_streaming(current_ids, [tid]):
                reasoning_ended = True
        else:
            content_parts.append(delta_text)

    content = "".join(content_parts)
    assert "</answer>" not in content, (
        f"Closing </answer> marker leaked into streamed content: {content!r}"
    )
