# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tokenizers import get_tokenizer

parser_name = "step3p5"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAME = "stepfun-ai/Step-3.5-Flash"


@pytest.fixture(scope="module")
def step3p5_tokenizer():
    return get_tokenizer(tokenizer_name=REASONING_MODEL_NAME)


SIMPLE_REASONING = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
# need to get into parser again to remove newline after </think>
COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
NO_CONTENT = {
    "output": "This is content",
    "reasoning": "This is content",
    "content": None,
    "is_reasoning_end": False,
}
NO_REASONING_STREAMING = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
MULTIPLE_LINES = {
    "output": "This\nThat</think>This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_NO_STREAMING = {
    "output": "</think>This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING = {
    "output": "</think>This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
REASONING_WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
COMPLETE_REASONING_WITH_THINK = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
MULTIPLE_LINES_WITH_THINK = {
    "output": "<think>This\nThat</think>This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_NO_STREAMING_WITH_THINK = {
    "output": "</think>This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_WITH_THINK = {
    "output": "</think>This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
THINK_NO_END = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
EMPTY = {
    "output": "",
    "reasoning": None,
    "content": None,
    "is_reasoning_end": False,
}
EMPTY_STREAMING = {
    "output": "",
    "reasoning": None,
    "content": None,
    "is_reasoning_end": False,
}
NEW_LINE = {
    "output": "\n<think>This is a reasoning section</think>\nThis is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

NEW_LINE_STREAMING = {
    "output": "\n<think>This is a reasoning section\n</think>\nThis is the rest",
    "reasoning": "\nThis is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

NEW_LINE_STREAMING_COMPLEX_CONTENT = {
    "output": "\n This is a \n reasoning section\n\n\n</think>\n\nThis is the rest",
    "reasoning": "\n This is a \n reasoning section\n\n",
    "content": "\nThis is the rest",
    "is_reasoning_end": True,
}

MULTI_TURN_PROMPT_CONTENT = {
    "output": "<think> This is last turn's reasoning section </think> hello <think>",
    "reasoning": "",
    "content": "",
    "is_reasoning_end": False,
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        True,
        SIMPLE_REASONING,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_streaming",
    ),
    pytest.param(
        False,
        NO_CONTENT,
        id="no_content_token",
    ),
    pytest.param(
        True,
        NO_REASONING_STREAMING,
        id="no_reasoning_token_streaming",
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
        True,
        SHORTEST_REASONING,
        id="shortest",
    ),
    pytest.param(
        False,
        SHORTEST_REASONING_NO_STREAMING,
        id="shortest_streaming",
    ),
    pytest.param(
        False,
        REASONING_WITH_THINK,
        id="reasoning_with_think",
    ),
    pytest.param(
        True,
        REASONING_WITH_THINK,
        id="reasoning_with_think_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think_streaming",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think_streaming",
    ),
    pytest.param(
        False,
        SHORTEST_REASONING_NO_STREAMING_WITH_THINK,
        id="shortest_with_think",
    ),
    pytest.param(
        True,
        SHORTEST_REASONING_WITH_THINK,
        id="shortest_with_think_streaming",
    ),
    pytest.param(
        False,
        THINK_NO_END,
        id="think_no_end",
    ),
    pytest.param(
        True,
        THINK_NO_END,
        id="think_no_end_streaming",
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
        NEW_LINE,
        id="new_line",
    ),
    pytest.param(
        True,
        NEW_LINE_STREAMING,
        id="new_line_streaming",
    ),
    pytest.param(
        True,
        NEW_LINE_STREAMING_COMPLEX_CONTENT,
        id="new_line_streaming_complex_content",
    ),
    pytest.param(
        True,
        MULTI_TURN_PROMPT_CONTENT,
        id="multi_turn_prompt_content",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    step3p5_tokenizer,
    request,
):
    output = step3p5_tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        step3p5_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        step3p5_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    print(f"reasoning: {reasoning}")
    print(f"content: {content}")
    test_id = request.node.callspec.id if hasattr(request.node, "callspec") else None
    if request.node.callspec.id != "multi_turn_prompt_content":
        assert reasoning == param_dict["reasoning"]
        assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = step3p5_tokenizer.convert_tokens_to_ids(output)
    if streaming:
        is_reasoning_end = parser.is_reasoning_end(output_ids)
        assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content
    if param_dict["content"] is not None:
        content = parser.extract_content_ids(output_ids)
        # Fixed expected token ids for specific test cases
        test_id = (
            request.node.callspec.id if hasattr(request.node, "callspec") else None
        )
        # Match most specific first
        if test_id not in [
            "new_line_streaming_complex_content",
            "new_line_streaming",
            "new_line",
            "multi_turn_prompt_content",
        ]:
            expected_content_ids = step3p5_tokenizer.convert_tokens_to_ids(
                step3p5_tokenizer.tokenize(param_dict["content"])
            )
            assert content == expected_content_ids
    else:
        content = parser.extract_content_ids(output)
        assert content == []


def test_step3p5_streaming_drops_leading_newline(step3p5_tokenizer):
    parser_cls = ReasoningParserManager.get_reasoning_parser("step3p5")
    parser = parser_cls(step3p5_tokenizer)
    output = "<think>calc</think>\nAnswer"
    tokens = step3p5_tokenizer.tokenize(output)
    output_tokens = [
        step3p5_tokenizer.convert_tokens_to_string([token]) for token in tokens
    ]

    _, content = run_reasoning_extraction(parser, output_tokens, streaming=True)
    assert content == "Answer"


def test_is_reasoning_end_streaming_and_nonstreaming_isolated(step3p5_tokenizer):
    """Verify that is_reasoning_end (non-streaming) is stateless and does not
    clobber the _end_token_pending accumulator used by is_reasoning_end_streaming.

    Before the fix, both paths shared _is_reasoning_end_from_ids which mutated
    _end_token_pending, causing cross-contamination when both paths were called
    on the same parser instance.
    """
    parser_cls = ReasoningParserManager.get_reasoning_parser("step3p5")

    start_id = parser_cls(step3p5_tokenizer).start_token_id
    end_id = parser_cls(step3p5_tokenizer).end_token_id

    # Pick a normal token id that is neither start nor end special token.
    normal_id = 1 if (start_id != 1 and end_id != 1) else start_id + 7

    # ------------------------------------------------------------------
    # Scenario A: non-streaming call must not clobber streaming pending state
    # ------------------------------------------------------------------
    parser = parser_cls(step3p5_tokenizer)

    # Feed a streaming delta ending exactly on </think> — streaming sets
    # _end_token_pending=True and returns False (still waiting for next token).
    full_ids_at_end: list[int] = [normal_id, end_id]
    result = parser.is_reasoning_end_streaming(full_ids_at_end, [end_id])
    assert result is False, (
        "Streaming: </think> at tail of delta should return False (pending)"
    )
    assert parser._end_token_pending is True, (
        "Streaming: _end_token_pending should be True after seeing </think> tail"
    )

    # Now call the non-streaming path with the full sequence ending at </think>.
    # Post-fix: should return False (no token follows </think>).
    # Post-fix: must NOT change _end_token_pending.
    ns_result = parser.is_reasoning_end(full_ids_at_end)
    assert ns_result is False, (
        "Non-streaming: sequence ending at </think> with no following token should "
        "return False"
    )
    assert parser._end_token_pending is True, (
        "Non-streaming is_reasoning_end must not mutate _end_token_pending "
        "(cross-contamination bug)"
    )

    # Now feed the confirming streaming delta (a single normal token).
    # Because _end_token_pending was preserved, streaming should return True.
    full_ids_after: list[int] = [normal_id, end_id, normal_id]
    streaming_confirm = parser.is_reasoning_end_streaming(full_ids_after, [normal_id])
    assert streaming_confirm is True, (
        "Streaming: normal token after pending </think> should confirm end (True). "
        "Pre-fix this returns False because _end_token_pending was clobbered."
    )

    # ------------------------------------------------------------------
    # Scenario B: is_reasoning_end correctness on various full sequences
    # ------------------------------------------------------------------
    fresh = parser_cls(step3p5_tokenizer)

    # Only </think>, nothing after — reasoning not yet ended.
    assert fresh.is_reasoning_end([end_id]) is False, (
        "Sequence [</think>] with nothing after should be False"
    )

    # </think> followed by a normal token — reasoning ended.
    assert fresh.is_reasoning_end([end_id, normal_id]) is True, (
        "Sequence [</think>, normal] should be True"
    )

    # <think> somewhere — still inside reasoning.
    assert fresh.is_reasoning_end([start_id, normal_id]) is False, (
        "Sequence ending after <think> should be False (still in reasoning)"
    )

    # No special tokens at all — not ended.
    assert fresh.is_reasoning_end([normal_id]) is False, (
        "Sequence with no special tokens should be False"
    )

    # </think> then start_id — start_id seen last, so still inside reasoning.
    assert fresh.is_reasoning_end([end_id, normal_id, start_id]) is False, (
        "Sequence whose last special token is <think> should be False"
    )

    # ------------------------------------------------------------------
    # Scenario C: is_reasoning_end must NOT mutate _end_token_pending at all
    # ------------------------------------------------------------------
    stateless = parser_cls(step3p5_tokenizer)
    assert stateless._end_token_pending is False

    # Call with a sequence that would trigger the old code to set pending.
    stateless.is_reasoning_end([end_id])
    assert stateless._end_token_pending is False, (
        "is_reasoning_end must not set _end_token_pending=True (stateless contract)"
    )

    stateless.is_reasoning_end([end_id, normal_id])
    assert stateless._end_token_pending is False, (
        "is_reasoning_end must not clear _end_token_pending (stateless contract)"
    )
