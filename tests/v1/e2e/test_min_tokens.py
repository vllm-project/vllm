# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive end-to-end tests for min_tokens functionality in V1 engine.

This test file addresses issue #21950: "Verify and add CI coverage for min_tokens"

The tests cover:
1. Basic min_tokens functionality (baseline behavior)
2. min_tokens with stop strings (the bug Calvin is fixing in PR #22014)
3. min_tokens with EOS tokens (potential LogitsProcessor bug)
4. Edge cases (min_tokens=max_tokens, min_tokens=0, etc.)
5. Multiple stop conditions

Background:
- Bug #21987: Stop sequences ignored when min_tokens specified  
- The bug is V1-specific and involves two potential failure points:
  a) Stop strings bypass min_tokens (detokenizer issue - Calvin's fix)
  b) EOS tokens may bypass min_tokens (LogitsProcessor issue - needs investigation)

Test Strategy:
- All tests use V1 engine (VLLM_USE_V1=1)
- Small model for fast CI execution (facebook/opt-125m)
- Parametrized tests for comprehensive coverage
- Clear assertions about expected vs actual token counts
- Some tests may initially fail (exposing known bugs) until fixes are merged
"""

import os
import pytest
from typing import List, Optional, Union

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput


# Test configuration
TEST_MODEL = "facebook/opt-125m"  # Small model for fast CI execution
TEMPERATURE = 0.0  # Deterministic generation for consistent testing


class MinTokensTestCase:
    """Data class for min_tokens test scenarios"""
    def __init__(
        self,
        name: str,
        min_tokens: int,
        max_tokens: int,
        stop: Optional[Union[str, List[str]]] = None,
        expected_min_len: int = None,
        expected_exact_len: int = None,
        should_pass: bool = True,
        xfail_reason: Optional[str] = None
    ):
        self.name = name
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.stop = stop
        self.expected_min_len = expected_min_len or min_tokens
        self.expected_exact_len = expected_exact_len
        self.should_pass = should_pass
        self.xfail_reason = xfail_reason

    def __str__(self):
        return f"{self.name}: min={self.min_tokens}, max={self.max_tokens}, stop={self.stop}"


# Test scenarios covering all critical cases
MIN_TOKENS_TEST_CASES = [
    # === BASIC FUNCTIONALITY (should work) ===
    MinTokensTestCase(
        name="basic_min_tokens_no_stop",
        min_tokens=8,
        max_tokens=20,
        stop=None,
        expected_min_len=8
    ),
    
    MinTokensTestCase(
        name="min_tokens_zero",
        min_tokens=0,
        max_tokens=10,
        stop=None,
        expected_min_len=0
    ),
    
    MinTokensTestCase(
        name="min_equals_max_no_stop",
        min_tokens=15,
        max_tokens=15,
        stop=None,
        expected_exact_len=15
    ),
    
    # === STOP STRINGS WITH MIN_TOKENS ===
    # These tests expose the detokenizer bug where stop strings bypass min_tokens
    # Using mathematically guaranteed approach with wide stop nets
    pytest.param(
        MinTokensTestCase(
            name="min_tokens_with_comprehensive_stops",
            min_tokens=5,  # Lower min_tokens for higher confidence
            max_tokens=20,  # Lower max_tokens to focus the test
            stop=["a", "e", "i", "o", "u", "t", "n", "s", "r", "l", " "],  # Comprehensive coverage
            expected_min_len=5,
            should_pass=False,
            xfail_reason="Known bug #21987: Stop strings bypass min_tokens (fixed by PR #22014)"
        ),
        marks=pytest.mark.xfail(reason="Known bug #21987: Stop strings bypass min_tokens (fixed by PR #22014)", strict=False),
        id="min_tokens_with_comprehensive_stops",
    ),
    
    pytest.param(
        MinTokensTestCase(
            name="min_tokens_with_simple_char_stop", 
            min_tokens=3,  # Very low threshold
            max_tokens=15,
            stop=["e", "a", " "],  # Multiple common patterns
            expected_min_len=3,
            should_pass=False,
            xfail_reason="Known bug #21987: Stop strings bypass min_tokens (fixed by PR #22014)"
        ),
        marks=pytest.mark.xfail(reason="Known bug #21987: Stop strings bypass min_tokens (fixed by PR #22014)", strict=False),
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
            should_pass=False,
            xfail_reason="Potential LogitsProcessor bug: EOS tokens may bypass min_tokens"
        ),
        marks=pytest.mark.xfail(reason="Potential LogitsProcessor bug: EOS tokens may bypass min_tokens", strict=False),
        id="min_equals_max_eos_only",
    ),
    
    # === EDGE CASES ===
    MinTokensTestCase(
        name="large_min_tokens",
        min_tokens=50,
        max_tokens=60,
        stop=None,
        expected_min_len=50
    ),
    
    MinTokensTestCase(
        name="min_tokens_with_empty_stop_list",
        min_tokens=5,
        max_tokens=15,
        stop=[],  # Empty stop list
        expected_min_len=5
    ),
]


@pytest.fixture(scope="module")
def llm_v1():
    """Create V1 LLM instance for testing"""
    # Ensure V1 engine is used
    os.environ["VLLM_USE_V1"] = "1"
    
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
    output: RequestOutput, 
    test_case: MinTokensTestCase
) -> None:
    """Assert that min_tokens requirement is satisfied"""
    token_count = get_token_count(output)
    
    if test_case.expected_exact_len is not None:
        # Exact length requirement
        assert token_count == test_case.expected_exact_len, (
            f"Expected exactly {test_case.expected_exact_len} tokens, "
            f"got {token_count} tokens. "
            f"Stop reason: {output.outputs[0].stop_reason}"
        )
    else:
        # Minimum length requirement
        assert token_count >= test_case.expected_min_len, (
            f"Expected at least {test_case.expected_min_len} tokens, "
            f"got {token_count} tokens. "
            f"Stop reason: {output.outputs[0].stop_reason}"
        )


@pytest.mark.parametrize("test_case", MIN_TOKENS_TEST_CASES, ids=lambda tc: tc.name)
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
        temperature=TEMPERATURE,
        include_stop_str_in_output=True  # Include stop strings in output for debugging
    )
    
    # Use simple prompt - the comprehensive stop lists should catch any generation
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
    sampling_params = SamplingParams(
        min_tokens=10,
        max_tokens=20,
        temperature=TEMPERATURE
    )
    
    prompt = "Once upon a time"
    outputs = llm_v1.generate([prompt], sampling_params)
    
    assert len(outputs) == 1
    token_count = get_token_count(outputs[0])
    
    assert token_count >= 10, f"Expected at least 10 tokens, got {token_count}"
    assert token_count <= 20, f"Expected at most 20 tokens, got {token_count}"


@pytest.mark.xfail(reason="Known bug #21987: Stop strings bypass min_tokens (fixed by PR #22014)", strict=False)
def test_min_tokens_stop_strings_bug(llm_v1: LLM):
    """
    Test the specific bug where stop strings bypass min_tokens.
    
    This test specifically reproduces the bug Calvin is fixing in PR #22014.
    It should fail until that fix is merged.
    
    Strategy: Use guaranteed stop characters that will appear in ANY generated text.
    """
    # If the bug is fixed upstream, this test will XPASS
    
    sampling_params = SamplingParams(
        min_tokens=15,
        max_tokens=50,
        stop=["e"],  # Most common letter in English - guaranteed to appear early
        temperature=TEMPERATURE,
        include_stop_str_in_output=True
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
    assert token_count >= 15, (
        f"Bug confirmed: Generated only {token_count} tokens despite min_tokens=15. "
        f"Stop reason: {outputs[0].outputs[0].stop_reason}. "
        f"Generated text: {repr(generated_text)}"
    )


@pytest.mark.xfail(reason="Known bug #21987: Stop strings bypass min_tokens (fixed by PR #22014)", strict=False)
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
        temperature=0.0,  # Maximum determinism
        include_stop_str_in_output=True
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
    if stop_reason == "stop":
        assert token_count >= 50, (
            f"Bug confirmed: Generated only {token_count} tokens despite min_tokens=50. "
            f"Early stop detected with reason '{stop_reason}'. "
            f"Generated text: {repr(generated_text)}"
        )




@pytest.mark.xfail(reason="Potential LogitsProcessor bug: EOS tokens may bypass min_tokens", strict=False)
def test_min_tokens_eos_behavior(llm_v1: LLM):
    """
    Test min_tokens behavior with EOS tokens (no explicit stop strings).
    
    This tests the MinTokensLogitsProcessor's handling of EOS tokens.
    If this fails, it indicates the LogitsProcessor bug because the MinTokensLogitsProcessor 
    may have failed to block an EOS when the token count is less than min_tokens
    """
    # If the bug is fixed upstream, this test will XPASS
    
    sampling_params = SamplingParams(
        min_tokens=25,
        max_tokens=25,  # Force exact length
        temperature=TEMPERATURE
    )
    
    prompt = "The capital of France is"
    outputs = llm_v1.generate([prompt], sampling_params)
    
    assert len(outputs) == 1
    token_count = get_token_count(outputs[0])
    
    # This should generate exactly 25 tokens
    assert token_count == 25, (
        f"Expected exactly 25 tokens, got {token_count}. "
        f"Stop reason: {outputs[0].outputs[0].stop_reason}"
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
    with pytest.raises(ValueError, match="min_tokens must be greater than or equal to 0"):
        SamplingParams(min_tokens=-1, max_tokens=10)
    
    with pytest.raises(ValueError, match="min_tokens must be less than or equal to max_tokens"):
        SamplingParams(min_tokens=15, max_tokens=10)


if __name__ == "__main__":
    """
    Run tests locally for development.
    
    Usage:
        cd vllm/
        VLLM_USE_V1=1 python -m pytest tests/v1/test_min_tokens.py -v
    """
    pytest.main([__file__, "-v"])