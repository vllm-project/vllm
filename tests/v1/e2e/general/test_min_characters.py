# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive end-to-end tests for `min_characters` in the V1 engine.

Covers:
1) Basic functionality
2) Stop strings with `min_characters`
3) Edge cases (min_characters == 0, large min_characters)
4) Interaction with min_tokens

Note: Unlike min_tokens which blocks EOS/stop tokens at the logits level,
min_characters only affects stop STRING matching in the detokenizer.
EOS tokens are NOT blocked by min_characters.
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

# Test configuration
TEST_MODEL = "facebook/opt-125m"  # Small model for fast CI execution
GREEDY = 0.0  # Deterministic generation for consistent testing


class MinCharactersTestCase:
    """Data class for min_characters test scenarios"""

    def __init__(
        self,
        name: str,
        min_characters: int,
        max_tokens: int,
        stop: str | list[str] | None = None,
    ):
        self.name = name
        self.min_characters = min_characters
        self.max_tokens = max_tokens
        self.stop = stop

    def __str__(self):
        return (
            f"{self.name}: min_chars={self.min_characters}, "
            f"max_tokens={self.max_tokens}, stop={self.stop}"
        )


# Test scenarios covering all critical cases
MIN_CHARACTERS_TEST_CASES = [
    # === BASIC FUNCTIONALITY ===
    MinCharactersTestCase(
        name="basic_min_characters_no_stop",
        min_characters=20,
        max_tokens=50,
        stop=None,
    ),
    MinCharactersTestCase(
        name="min_characters_zero",
        min_characters=0,
        max_tokens=20,
        stop=None,
    ),
    # === STOP STRINGS WITH MIN_CHARACTERS ===
    MinCharactersTestCase(
        name="min_characters_with_stop_string",
        min_characters=15,
        max_tokens=50,
        stop=["e"],  # Common letter that will appear early
    ),
    MinCharactersTestCase(
        name="min_characters_with_multiple_stops",
        min_characters=20,
        max_tokens=50,
        stop=["a", "e", "i", "o", "u"],  # Vowels - will appear early
    ),
    MinCharactersTestCase(
        name="min_characters_with_space_stop",
        min_characters=10,
        max_tokens=30,
        stop=[" "],  # Space will appear very early
    ),
    # === EDGE CASES ===
    MinCharactersTestCase(
        name="large_min_characters",
        min_characters=100,
        max_tokens=100,
        stop=None,
    ),
    MinCharactersTestCase(
        name="min_characters_with_empty_stop_list",
        min_characters=15,
        max_tokens=30,
        stop=[],  # Empty stop list
    ),
]


@pytest.fixture(scope="module")
def llm_v1():
    """Create V1 LLM instance for testing"""
    llm = LLM(
        model=TEST_MODEL,
        tensor_parallel_size=1,
        max_model_len=1024,  # Small context for fast testing
        enforce_eager=True,  # Avoid graph compilation overhead
    )
    return llm


def get_output_text(output: RequestOutput) -> str:
    """Extract generated text from LLM output"""
    if not output.outputs:
        return ""
    return output.outputs[0].text


def get_token_count(output: RequestOutput) -> int:
    """Extract token count from LLM output"""
    if not output.outputs:
        return 0
    return len(output.outputs[0].token_ids)


def assert_min_characters_satisfied(
    output: RequestOutput, test_case: MinCharactersTestCase
) -> None:
    """Assert that min_characters requirement is satisfied"""
    generated_text = get_output_text(output)
    char_count = len(generated_text)
    stop_reason = output.outputs[0].stop_reason if output.outputs else "no output"

    assert char_count >= test_case.min_characters, (
        f"Expected at least {test_case.min_characters} characters, "
        f"got {char_count} characters. "
        f"Stop reason: {stop_reason}. "
        f"Text: {repr(generated_text)}"
    )


@pytest.mark.parametrize(
    "test_case",
    MIN_CHARACTERS_TEST_CASES,
    ids=lambda tc: tc.name,
)
def test_min_characters_comprehensive(llm_v1: LLM, test_case: MinCharactersTestCase):
    """
    Comprehensive test for min_characters functionality in V1 engine.

    This test covers all critical scenarios for min_characters:
    - Basic functionality
    - Stop strings with min_characters
    - Edge cases

    Args:
        llm_v1: V1 LLM instance
        test_case: Test scenario parameters
    """
    # Create sampling parameters
    sampling_params = SamplingParams(
        min_characters=test_case.min_characters,
        max_tokens=test_case.max_tokens,
        stop=test_case.stop,
        temperature=GREEDY,
        include_stop_str_in_output=True,  # Include stop strings for debugging
    )

    # Use simple prompt
    prompt = "Hello"

    # Generate output
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1, "Expected exactly one output"
    output = outputs[0]

    # Debug information
    generated_text = get_output_text(output)
    char_count = len(generated_text)
    token_count = get_token_count(output)
    stop_reason = output.outputs[0].stop_reason if output.outputs else "unknown"

    print(f"\nTest: {test_case.name}")
    print(f"Generated {char_count} characters, {token_count} tokens")
    print(f"Stop reason: {repr(stop_reason)}")
    print(f"Generated text: {repr(generated_text)}")
    print(f"Expected min chars: {test_case.min_characters}")

    # Validate min_characters requirement
    assert_min_characters_satisfied(output, test_case)


def test_min_characters_vs_min_tokens_independence(llm_v1: LLM):
    """
    Test that min_characters and min_tokens work independently.

    min_tokens blocks EOS/stop tokens at logit level.
    min_characters blocks stop strings at detokenizer level.
    Both constraints should be respected.
    """
    # Use both min_tokens and min_characters
    sampling_params = SamplingParams(
        min_tokens=5,
        min_characters=30,
        max_tokens=50,
        stop=["e"],  # Common letter
        temperature=GREEDY,
        include_stop_str_in_output=True,
    )

    prompt = "Hello world"
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1
    generated_text = get_output_text(outputs[0])
    token_count = get_token_count(outputs[0])
    char_count = len(generated_text)

    print(f"Generated text: {repr(generated_text)}")
    print(f"Token count: {token_count}, Character count: {char_count}")

    # Both constraints should be satisfied
    assert token_count >= 5, f"min_tokens not satisfied: {token_count} < 5"
    assert char_count >= 20, f"min_characters not satisfied: {char_count} < 20"


def test_min_characters_does_not_block_eos(llm_v1: LLM):
    """
    Test that min_characters does NOT block EOS tokens.

    Unlike min_tokens, min_characters only affects stop STRING matching.
    EOS tokens should still terminate generation even if min_characters
    is not reached.

    This is expected behavior - use min_tokens to block EOS.
    """
    tokenizer = llm_v1.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    min_characters = 1000
    # High min_characters but no min_tokens
    sampling_params = SamplingParams(
        min_characters=min_characters,  # Very high - won't be reached
        min_tokens=0,  # Don't block EOS
        max_tokens=500,
        temperature=GREEDY,
    )

    # Prompt that might generate EOS early
    prompt = "Give a file extension."
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1
    output = outputs[0]
    generated_text = get_output_text(output)
    token_ids = output.outputs[0].token_ids if output.outputs else []
    finish_reason = output.outputs[0].finish_reason if output.outputs else "unknown"

    print(f"Generated text: {repr(generated_text)}")
    print(f"Finish reason: {finish_reason}")
    print(f"Character count: {len(generated_text)}")

    # If EOS was generated, it should have stopped generation
    # even though min_characters wasn't reached
    if eos_token_id in token_ids or finish_reason == "stop":
        # This is expected - min_characters doesn't block EOS
        assert len(generated_text) < min_characters, (
            "EOS should terminate generation regardless of min_characters"
        )


def test_min_characters_validation():
    """
    Test that SamplingParams correctly validates min_characters parameters.
    """
    # Valid cases
    SamplingParams(min_characters=0)
    SamplingParams(min_characters=50)
    SamplingParams(min_characters=100)

    # Invalid case: negative min_characters
    with pytest.raises(
        ValueError,
        match="min_characters must be greater than or equal to 0",
    ):
        SamplingParams(min_characters=-1)


if __name__ == "__main__":
    """
    Run tests locally for development.

    Usage:
        cd vllm/
        python -m pytest tests/v1/e2e/general/test_min_characters.py -v
    """
    pytest.main([__file__, "-v"])
