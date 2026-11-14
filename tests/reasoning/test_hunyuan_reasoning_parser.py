# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
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


def _new_parser() -> HunyuanA13BReasoningParser:
    return HunyuanA13BReasoningParser(tokenizer)


END_DETECTION_OUTPUTS = [
    pytest.param(SIMPLE_REASONING["output"], id="simple_reasoning"),
    pytest.param(NO_REASONING_QUICK_THOUGHT["output"], id="quick_thought"),
    pytest.param(MULTIPLE_LINES["output"], id="multiple_lines"),
    pytest.param(COMPLETE_REASONING_WITH_SYMBOL["output"], id="with_symbol"),
]


@pytest.mark.parametrize("output", END_DETECTION_OUTPUTS)
def test_reasoning_end_gate_matches_streaming_extraction(output: str):
    """The structured-output manager uses its own parser instance and never
    calls `extract_reasoning_streaming`, so end detection must work from token
    ids alone and fire on the same token where extraction switches to the
    answer."""
    ids = tokenizer.encode(output, add_special_tokens=False)

    extractor = _new_parser()
    extraction_end = None
    for i in range(1, len(ids) + 1):
        extractor.extract_reasoning_streaming(
            tokenizer.decode(ids[: i - 1]),
            tokenizer.decode(ids[:i]),
            tokenizer.decode(ids[i - 1 : i]),
            ids[: i - 1],
            ids[:i],
            ids[i - 1 : i],
        )
        if extractor.current_state == "response":
            extraction_end = i
            break
    assert extraction_end is not None

    gate = _new_parser()
    gate_steps = [
        i
        for i in range(1, len(ids) + 1)
        if gate.is_reasoning_end_streaming(ids[:i], ids[i - 1 : i])
    ]
    assert gate_steps == [extraction_end]
    assert not gate.is_reasoning_end(ids[: extraction_end - 1])
    assert gate.is_reasoning_end(ids[:extraction_end])


def test_is_reasoning_end_streaming_marker_across_steps():
    ids = tokenizer.encode(SIMPLE_REASONING["output"], add_special_tokens=False)
    parser = _new_parser()
    end = len(tokenizer.encode(COMPLETE_REASONING["output"], add_special_tokens=False))

    # Several tokens accepted at once, e.g. speculative decoding.
    assert parser.is_reasoning_end_streaming(ids[:end], iter(ids[end - 3 : end]))
    assert not parser.is_reasoning_end_streaming(ids[: end + 2], ids[end : end + 2])


def test_is_reasoning_end_ignores_prior_turns_in_prompt():
    """Earlier assistant turns keep `<think>...</answer>` in the prompt; they
    must not start the grammar for the current turn."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": SIMPLE_REASONING["output"]},
        {"role": "user", "content": "again"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    parser = _new_parser()

    assert not parser.is_reasoning_end(prompt_ids)
    assert not parser.is_reasoning_end(
        prompt_ids + tokenizer.encode(START_REASONING, add_special_tokens=False)
    )
    assert parser.is_reasoning_end(
        prompt_ids
        + tokenizer.encode(COMPLETE_REASONING["output"], add_special_tokens=False)
    )
