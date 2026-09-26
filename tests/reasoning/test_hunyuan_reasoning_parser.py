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


def _feed_streaming(parser: ReasoningParser, token_ids: list[int], step: int):
    """Feed token_ids in slices of `step`; return the concatenated output."""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    previous: list[int] = []

    for start in range(0, len(token_ids), step):
        current = token_ids[: start + step]
        delta = token_ids[start : start + step]
        delta_message = parser.extract_reasoning_streaming(
            tokenizer.decode(previous),
            tokenizer.decode(current),
            tokenizer.decode(delta),
            previous,
            current,
            delta,
        )
        previous = current
        if delta_message is not None:
            if delta_message.reasoning:
                reasoning_parts.append(delta_message.reasoning)
            if delta_message.content:
                content_parts.append(delta_message.content)

    return "".join(reasoning_parts), "".join(content_parts)


@pytest.mark.parametrize("step", [2, 3, 7, 100])
def test_streaming_multi_token_delta_matches_single_token_feed(step: int):
    """A streaming step may carry several token ids.

    The engine passes one generation step per call
    (``as_list(output.token_ids)``), which holds several tokens when speculative
    decoding or MTP is enabled. The parser asserted a single token per delta and
    raised AssertionError for such steps. Feeding the same ids in larger slices
    must produce exactly the same output as feeding them one at a time.
    """
    parser = ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer)
    token_ids = (
        list(parser.think_start_ids)
        + tokenizer.encode(" reasoning here")
        + list(parser.response_start_ids)
        + tokenizer.encode(" the answer")
        + list(parser.response_end_ids)
    )

    single_token = _feed_streaming(
        ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer),
        token_ids,
        1,
    )
    assert single_token[0] == " reasoning here"
    assert single_token[1] == " the answer"

    multi_token = _feed_streaming(
        ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer),
        token_ids,
        step,
    )
    assert multi_token == single_token


def test_empty_delta_returns_none():
    """A step carrying no tokens has nothing to emit.

    ``extract_reasoning_streaming`` is part of the parser framework, which does
    pass ``delta_token_ids=[]`` (vllm/parser/engine/parser_engine.py), and before
    this change an empty delta tripped ``assert len(delta_token_ids) == 1``
    instead of being ignored.
    """
    parser = ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer)
    assert parser.extract_reasoning_streaming("", "", "", [], [], []) is None


@pytest.mark.parametrize(
    "payload",
    [
        " reasoning here",
        " reasoning 中文测试",
        " reasoning café naïve résumé",
        " reasoning rocket 🚀 done",
        " reasoning 中文 🚀 café",
    ],
)
def test_per_token_text_reconstructs_the_delta(payload: str):
    """Per-token text must reconstruct the delta exactly.

    A character can span several tokens under byte-fallback tokenization. The
    helper decodes growing prefixes of the delta, so an unfinished trailing byte
    sequence (U+FFFD) must be withheld rather than emitted: otherwise the slice
    that derives each token's text is taken against a prefix that later changes,
    and astral-plane characters (emoji) come out corrupted.
    """
    parser = ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer)
    token_ids = (
        list(parser.think_start_ids)
        + tokenizer.encode(payload)
        + list(parser.response_start_ids)
        + tokenizer.encode(" answer")
        + list(parser.response_end_ids)
    )

    reconstructed = "".join(
        text for _, text in parser._iter_delta_tokens([], token_ids, token_ids, "")
    )
    assert reconstructed == tokenizer.decode(token_ids, skip_special_tokens=False)
