# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction_streaming
from vllm.reasoning import ReasoningParserManager

# Issue #35221: When max_tokens is hit, the model output is truncated.
# If this happens inside a reasoning block, the parser should still
# identify the emitted tokens as reasoning, not content.
TRUNCATED_REASONING_CASE = [
    pytest.param(
        # Simulates a stream where <think> was in the prompt (Qwen3.5 style),
        # so we just see reasoning text, but it gets cut off without </think>.
        ["This is some reasoning", " that gets truncated"],
        "This is some reasoning that gets truncated",
        None, # Should NOT be content
        id="truncated_reasoning_no_end_tag",
    ),
    pytest.param(
        # Simulates old style where <think> is generated, then truncated.
        ["<think>", "This is reasoning", " truncated"],
        "This is reasoning truncated",
        None,
        id="truncated_reasoning_with_start_tag",
    )
]

@pytest.mark.parametrize(
    "deltas, expected_reasoning, expected_content", TRUNCATED_REASONING_CASE
)
def test_qwen3_reasoning_truncation(
    deltas, expected_reasoning, expected_content
):
    model_name = "Qwen/Qwen3-0.6B" # Or any Qwen3 model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        pytest.skip(f"Skipping test, could not load tokenizer for {model_name}")

    parser_cls = ReasoningParserManager.get_reasoning_parser("qwen3")
    parser = parser_cls(tokenizer)

    reconstructor = run_reasoning_extraction_streaming(parser, deltas)

    assert reconstructor.reasoning == expected_reasoning
    # This assertion fails if the parser defaults to content when no tags are seen
    assert (reconstructor.other_content or None) == expected_content
