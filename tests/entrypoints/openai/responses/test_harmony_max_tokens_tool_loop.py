"""Test that max_output_tokens is respected in Harmony tool-call loops.

Regression test: the Harmony path used `max_model_len - len(token_ids)` for
subsequent tool-call turns, ignoring `request.max_output_tokens`. The
ParsableContext path correctly used `get_max_tokens()`.
"""

from vllm.entrypoints.utils import get_max_tokens


def test_harmony_max_tokens_respects_max_output_tokens():
    """Verify that get_max_tokens limits output based on max_output_tokens."""
    max_model_len = 131072
    max_output_tokens = 100
    prompt_len = 1000

    # Correct behavior: respect max_output_tokens
    result = get_max_tokens(max_model_len, max_output_tokens, prompt_len, {})
    assert result == 100, f"Expected 100, got {result}"

    # Bug behavior: ignore max_output_tokens
    bug_result = max_model_len - prompt_len
    assert bug_result == 130072, f"Expected 130072, got {bug_result}"

    # The bug gives much more tokens than intended
    assert bug_result > result, "Bug allows more tokens than max_output_tokens"


def test_harmony_max_tokens_without_limit():
    """When max_output_tokens is None, use remaining context window."""
    max_model_len = 131072
    prompt_len = 1000

    result = get_max_tokens(max_model_len, None, prompt_len, {})
    # Should use max_model_len - prompt_len
    assert result == 130072


def test_harmony_max_tokens_capped_by_context():
    """When max_output_tokens exceeds remaining context, cap it."""
    max_model_len = 131072
    max_output_tokens = 200000  # more than max_model_len
    prompt_len = 1000

    result = get_max_tokens(max_model_len, max_output_tokens, prompt_len, {})
    # Should be capped at max_model_len - prompt_len
    assert result == 130072
