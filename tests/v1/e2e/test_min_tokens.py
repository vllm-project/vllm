# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive end-to-end tests for `min_tokens` in the V1 engine.

Addresses #21950: verify and add CI coverage.

Covers:
1) Basic functionality
2) Stop strings with `min_tokens` (bug #21987; fix in PR #22014)
3) EOS behavior with `min_tokens` (potential logits-processor bug)
4) Edge cases (min_tokens == max_tokens, min_tokens == 0)
5) Multiple stop conditions
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

# Test configuration
TEST_MODEL = "facebook/opt-125m"  # Small model for fast CI execution
GREEDY = 0.0  # Deterministic generation for consistent testing


class MinTokensTestCase:
    """Data class for min_tokens test scenarios"""

    def __init__(
        self,
        name: str,
        min_tokens: int,
        max_tokens: int,
        stop: str | list[str] | None = None,
        expected_min_len: int | None = None,
        expected_exact_len: int | None = None,
    ):
        self.name = name
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.stop = stop
        self.expected_min_len = expected_min_len or min_tokens
        self.expected_exact_len = expected_exact_len

    def __str__(self):
        return (
            f"{self.name}: min={self.min_tokens}, "
            f"max={self.max_tokens}, stop={self.stop}"
        )


# Test scenarios covering all critical cases
MIN_TOKENS_TEST_CASES = [
    # === BASIC FUNCTIONALITY (should work) ===
    MinTokensTestCase(
        name="basic_min_tokens_no_stop",
        min_tokens=8,
        max_tokens=20,
        stop=None,
        expected_min_len=8,
    ),
    MinTokensTestCase(
        name="min_tokens_zero",
        min_tokens=0,
        max_tokens=10,
        stop=None,
        expected_min_len=0,
    ),
    MinTokensTestCase(
        name="min_equals_max_no_stop",
        min_tokens=15,
        max_tokens=15,
        stop=None,
        expected_exact_len=15,
    ),
    # === STOP STRINGS WITH MIN_TOKENS ===
    # These tests expose the detokenizer bug where stop strings
    # bypass min_tokens
    # Using mathematically guaranteed approach with wide stop nets
    pytest.param(
        MinTokensTestCase(
            name="min_tokens_with_comprehensive_stops",
            min_tokens=5,
            max_tokens=20,
            stop=[
                "a",
                "e",
                "i",
                "o",
                "u",
                "t",
                "n",
                "s",
                "r",
                "l",
                " ",
            ],
            expected_min_len=5,
        ),
        marks=pytest.mark.xfail(
            reason=(
                "Known bug #21987: stop strings bypass min_tokens (fixed by PR #22014)"
            ),
            strict=False,
        ),
        id="min_tokens_with_comprehensive_stops",
    ),
    pytest.param(
        MinTokensTestCase(
            name="min_tokens_with_simple_char_stop",
            min_tokens=3,
            max_tokens=15,
            stop=["e", "a", " "],
            expected_min_len=3,
        ),
        marks=pytest.mark.xfail(
            reason=(
                "Known bug #21987: stop strings bypass min_tokens (fixed by PR #22014)"
            ),
            strict=False,
        ),
        id="min_tokens_with_simple_char_stop",
    ),
    # === EOS TOKEN WITH MIN_TOKENS (potential LogitsProcessor bug) ===
    # These test the MinTokensLogitsProcessor handling of EOS tokens
    pytest.param(
        MinTokensTestCase(
            name="min_equals_max_eos_only",
            min_tokens=20,
            max_tokens=20,
            stop=None,  # Relies on default EOS token behavior
            expected_exact_len=20,
        ),
        marks=pytest.mark.xfail(
            reason=("Potential logits-processor bug: EOS tokens may bypass min_tokens"),
            strict=False,
        ),
        id="min_equals_max_eos_only",
    ),
    # === EDGE CASES ===
    MinTokensTestCase(
        name="large_min_tokens",
        min_tokens=50,
        max_tokens=60,
        stop=None,
        expected_min_len=50,
    ),
    MinTokensTestCase(
        name="min_tokens_with_empty_stop_list",
        min_tokens=5,
        max_tokens=15,
        stop=[],  # Empty stop list
        expected_min_len=5,
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


def get_token_count(output: RequestOutput) -> int:
    """Extract token count from LLM output"""
    if not output.outputs:
        return 0
    return len(output.outputs[0].token_ids)


def assert_min_tokens_satisfied(
    output: RequestOutput, test_case: MinTokensTestCase
) -> None:
    """Assert that min_tokens requirement is satisfied"""
    token_count = get_token_count(output)
    stop_reason = output.outputs[0].stop_reason if output.outputs else "no output"

    if test_case.expected_exact_len is not None:
        # Exact length requirement
        assert token_count == test_case.expected_exact_len, (
            f"Expected exactly {test_case.expected_exact_len} tokens, "
            f"got {token_count} tokens. "
            f"Stop reason: {stop_reason}"
        )
    else:
        # Minimum length requirement
        assert token_count >= (test_case.expected_min_len or 0), (
            f"Expected at least {test_case.expected_min_len} tokens, "
            f"got {token_count} tokens. "
            f"Stop reason: {stop_reason}"
        )


@pytest.mark.parametrize(
    "test_case",
    MIN_TOKENS_TEST_CASES,
    ids=lambda tc: tc.name,
)
def test_min_tokens_comprehensive(llm_v1: LLM, test_case: MinTokensTestCase):
    """
    Comprehensive test for min_tokens functionality in V1 engine.

    This test covers all critical scenarios for min_tokens:
    - Basic functionality (should work)
    - Stop strings with min_tokens (known bug)
    - EOS tokens with min_tokens (potential bug)
    - Edge cases

    Args:
        llm_v1: V1 LLM instance
        test_case: Test scenario parameters
    """
    # Known failing cases are handled via param-level xfail marks above.

    # Create sampling parameters
    sampling_params = SamplingParams(
        min_tokens=test_case.min_tokens,
        max_tokens=test_case.max_tokens,
        stop=test_case.stop,
        temperature=GREEDY,
        include_stop_str_in_output=True,  # Include stop strings for debugging
    )

    # Use simple prompt. Comprehensive stop lists should catch any generation
    prompt = "Hello"

    # Generate output
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1, "Expected exactly one output"
    output = outputs[0]

    # Debug information
    token_count = get_token_count(output)
    generated_text = output.outputs[0].text if output.outputs else ""
    stop_reason = output.outputs[0].stop_reason if output.outputs else "unknown"

    print(f"\nTest: {test_case.name}")
    print(f"Generated {token_count} tokens")
    print(f"Stop reason: {stop_reason}")
    print(f"Generated text: {repr(generated_text)}")
    print(f"Expected min: {test_case.expected_min_len}")
    if test_case.expected_exact_len:
        print(f"Expected exact: {test_case.expected_exact_len}")

    # Validate min_tokens requirement
    assert_min_tokens_satisfied(output, test_case)


def test_min_tokens_basic_functionality(llm_v1: LLM):
    """
    Test basic min_tokens functionality without stop conditions.

    This is a baseline test that should always pass and validates
    that min_tokens works correctly in the simple case.
    """
    sampling_params = SamplingParams(min_tokens=10, max_tokens=20, temperature=GREEDY)

    prompt = "Once upon a time"
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1
    token_count = get_token_count(outputs[0])

    assert token_count >= 10, f"Expected at least 10 tokens, got {token_count}"
    assert token_count <= 20, f"Expected at most 20 tokens, got {token_count}"


@pytest.mark.xfail(
    reason=("Known bug #21987: stop strings bypass min_tokens (fixed by PR #22014)"),
    strict=False,
)
def test_min_tokens_stop_strings_bug(llm_v1: LLM):
    """
    Test the specific bug where stop strings bypass min_tokens.

    This test specifically reproduces the bug Calvin is fixing in PR #22014.
    It should fail until that fix is merged.

    Strategy: Use guaranteed stop characters that will appear
    in any generated text.
    """
    # If the bug is fixed upstream, this test will XPASS

    sampling_params = SamplingParams(
        min_tokens=15,
        max_tokens=50,
        # Common letter; likely appears early
        stop=["e"],
        temperature=GREEDY,
        include_stop_str_in_output=True,
    )

    # Simple prompt that will generate text containing "e"
    prompt = "The quick brown fox"
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1
    token_count = get_token_count(outputs[0])
    generated_text = outputs[0].outputs[0].text if outputs[0].outputs else ""

    # Debug info to understand what happened
    print(f"Generated text: {repr(generated_text)}")
    print(f"Token count: {token_count}")
    print(f"Contains 'e': {'e' in generated_text}")

    # This assertion should fail due to the bug - if stop string is found early,
    # the model should still continue generating until min_tokens is reached
    stop_reason = (
        outputs[0].outputs[0].stop_reason if outputs[0].outputs else "no output"
    )
    assert token_count >= 15, (
        "Bug confirmed: "
        f"{token_count} tokens < min_tokens=15. "
        f"Reason: {stop_reason}. "
        f"Text: {repr(generated_text)}"
    )


@pytest.mark.xfail(
    reason=("Known bug #21987: stop strings bypass min_tokens (fixed by PR #22014)"),
    strict=False,
)
def test_min_tokens_stop_strings_guaranteed_early_trigger(llm_v1: LLM):
    """
    Guaranteed test for stop strings bypassing min_tokens bug.

    Strategy: Use very low temperature and multiple common stop strings
    to virtually guarantee early detection, combined with long min_tokens
    to ensure the bug is exposed regardless of model behavior.
    """
    # If the bug is fixed upstream, this test will XPASS

    sampling_params = SamplingParams(
        min_tokens=50,  # Set high min_tokens to ensure bug detection
        max_tokens=200,
        # Use multiple very common patterns - at least one will appear
        stop=["e", "a", "i", "o", "u", " ", "t", "n", "s", "r"],
        temperature=GREEDY,
        include_stop_str_in_output=True,
    )

    # Simple prompt that will generate some text
    prompt = "The cat"
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1
    token_count = get_token_count(outputs[0])
    generated_text = outputs[0].outputs[0].text if outputs[0].outputs else ""
    stop_reason = outputs[0].outputs[0].stop_reason if outputs[0].outputs else "unknown"

    print(f"Generated text: {repr(generated_text)}")
    print(f"Token count: {token_count}")
    print(f"Stop reason: {stop_reason}")

    # With the bug, this will fail because ANY of the common characters
    # will trigger early termination before min_tokens=50 is reached
    # It's virtually impossible to generate 50 tokens without hitting
    # at least one of: e, a, i, o, u, space, t, n, s, r
    finish_reason = (
        outputs[0].outputs[0].finish_reason if outputs[0].outputs else "unknown"
    )

    print(f"Finish reason: {finish_reason}")

    if finish_reason == "stop":
        assert token_count >= 50, (
            "Bug confirmed: "
            f"{token_count} tokens < min_tokens=50. "
            f"Reason: {finish_reason}. "
            f"Text: {repr(generated_text)}"
        )


@pytest.mark.xfail(
    reason=("Potential logits-processor bug: EOS tokens may bypass min_tokens"),
    strict=False,
)
def test_min_tokens_eos_behavior(llm_v1: LLM):
    """
    Verify EOS handling with and without min_tokens.

    - Without min_tokens: expect early EOS -> finish_reason == "stop",
      stop_reason is None, and generated tokens < max_tokens (25).
    - With min_tokens: EOS should be blocked until min_tokens is reached
      (finish_reason == "length"); verify that eos_token_id does not appear
      in generated token_ids.
    """
    # tokenizer + eos id
    tokenizer = llm_v1.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    prompt = "Give a file extension."
    max_toks = 32

    # Case 1: WITHOUT min_tokens
    sp_no_min = SamplingParams(
        max_tokens=max_toks,
        temperature=GREEDY,
    )
    out_no_min = llm_v1.generate([prompt], sp_no_min)
    assert len(out_no_min) == 1
    choice_no_min = out_no_min[0].outputs[0]

    ids_no_min = choice_no_min.token_ids or []
    finish_no_min = choice_no_min.finish_reason
    stop_no_min = choice_no_min.stop_reason

    print(
        "[no-min] tokens=",
        len(ids_no_min),
        " finish=",
        finish_no_min,
        " stop_reason=",
        stop_no_min,
    )

    assert finish_no_min == "stop", (
        f"Expected finish_reason 'stop' without min_tokens, got {finish_no_min}"
    )
    assert stop_no_min is None, (
        "For EOS-based stop (no user stop strings), stop_reason should be None."
    )
    assert len(ids_no_min) < max_toks, (
        f"Expected early EOS with < {max_toks} tokens, got {len(ids_no_min)}"
    )

    # Case 2: WITH min_tokens
    sp_with_min = SamplingParams(
        min_tokens=max_toks,
        max_tokens=max_toks,
        temperature=GREEDY,
    )
    out_with_min = llm_v1.generate([prompt], sp_with_min)
    assert len(out_with_min) == 1
    choice_with_min = out_with_min[0].outputs[0]

    ids_with_min = choice_with_min.token_ids or []
    finish_with_min = choice_with_min.finish_reason
    stop_with_min = choice_with_min.stop_reason

    print(
        "[with-min] tokens=",
        len(ids_with_min),
        " finish=",
        finish_with_min,
        " stop_reason=",
        stop_with_min,
    )

    # Exact length reached; EOS should have been blocked
    assert len(ids_with_min) == max_toks, (
        f"Expected exactly {max_toks} tokens with min_tokens; got {len(ids_with_min)}"
    )
    assert finish_with_min == "length", (
        f"Expected finish_reason 'length'; got {finish_with_min}"
    )
    assert eos_token_id not in ids_with_min, (
        "EOS token id should not appear when min_tokens prevents early EOS."
    )


def test_min_tokens_validation():
    """
    Test that SamplingParams correctly validates min_tokens parameters.

    This tests the parameter validation logic in SamplingParams.
    """
    # Valid cases
    SamplingParams(min_tokens=0, max_tokens=10)
    SamplingParams(min_tokens=5, max_tokens=10)
    SamplingParams(min_tokens=10, max_tokens=10)

    # Invalid cases
    with pytest.raises(
        ValueError,
        match="min_tokens must be greater than or equal to 0",
    ):
        SamplingParams(min_tokens=-1, max_tokens=10)

    with pytest.raises(
        ValueError,
        match="min_tokens must be less than or equal to max_tokens",
    ):
        SamplingParams(min_tokens=15, max_tokens=10)


if __name__ == "__main__":
    """
    Run tests locally for development.
    
    Usage:
        cd vllm/
        python -m pytest tests/v1/e2e/test_min_tokens.py -v
    """
    pytest.main([__file__, "-v"])
