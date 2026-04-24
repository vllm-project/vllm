#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple unit test for _adjust_logprobs_for_force_end_thinking function.
Run with: python test_adjust_logprobs_simple.py
"""

import sys
from typing import Any

import numpy as np

from vllm.cohere.utils import _adjust_logprobs_for_force_end_thinking
from vllm.v1.outputs import LogprobsLists


def create_test_logprobs(token_ids, logprobs, ranks):
    """Helper to create LogprobsLists for testing."""
    return LogprobsLists(
        logprob_token_ids=np.array(token_ids, dtype=np.int32),
        logprobs=np.array(logprobs, dtype=np.float32),
        sampled_token_ranks=np.array(ranks, dtype=np.int32),
        cu_num_generated_tokens=[len(token_ids)],
    )


def test_no_adjustments():
    """Test that function returns unchanged data when no adjustments."""
    print("Test 1: No adjustments - should return original data")

    # Original data: 3 requests with 2, 3, 2 tokens respectively
    original_token_ids = [100, 101, 200, 201, 202, 300, 301]
    original_logprobs = [-1.0, -2.0, -1.5, -2.5, -3.0, -1.8, -2.2]
    original_ranks = [0, 1, 0, 1, 2, 0, 1]

    logprobs_lists = create_test_logprobs(
        original_token_ids, original_logprobs, original_ranks
    )
    valid_sampled_token_ids = [[100, 101], [200, 201, 202], [300, 301]]
    token_adjustments: list[dict[str, Any]] = []

    result = _adjust_logprobs_for_force_end_thinking(
        logprobs_lists, token_adjustments, valid_sampled_token_ids
    )

    # Should be unchanged
    assert np.array_equal(result.logprob_token_ids, np.array(original_token_ids)), (
        f"Token IDs mismatch: {result.logprob_token_ids} vs {original_token_ids}"
    )
    assert np.allclose(result.logprobs, np.array(original_logprobs)), (
        f"Logprobs mismatch: {result.logprobs} vs {original_logprobs}"
    )
    assert np.array_equal(result.sampled_token_ranks, original_ranks), (
        f"Ranks mismatch: {result.sampled_token_ranks} vs {original_ranks}"
    )
    print("✓ PASS: No adjustments test")


def test_truncate_single_request():
    """Test truncating tokens from one request."""
    print("Test 2: Truncate tokens from request 1 (middle request)")

    # Original: req0:[100,101], req1:[200,201,202], req2:[300,301]
    # After: req0:[100,101], req1:[200], req2:[300,301] (remove last 2 from req1)
    original_token_ids = [100, 101, 200, 201, 202, 300, 301]
    original_logprobs = [-1.0, -2.0, -1.5, -2.5, -3.0, -1.8, -2.2]
    original_ranks = [0, 1, 0, 1, 2, 0, 1]

    logprobs_lists = create_test_logprobs(
        original_token_ids, original_logprobs, original_ranks
    )

    # Request 1 (index 1) had 3 tokens, keep only 1
    valid_sampled_token_ids = [[100, 101], [200], [300, 301]]  # After truncation
    token_adjustments = [
        {
            "req_index": 1,  # Request at position 1 in valid_sampled_token_ids
            "action": "truncate",
            "original_length": 3,
            "final_length": 1,
        }
    ]

    result = _adjust_logprobs_for_force_end_thinking(
        logprobs_lists, token_adjustments, valid_sampled_token_ids
    )

    # Expected: remove tokens at indices 3,4 (201,202)
    expected_token_ids = [100, 101, 200, 300, 301]
    expected_logprobs = [-1.0, -2.0, -1.5, -1.8, -2.2]
    expected_ranks = [0, 1, 0, 0, 1]

    assert np.array_equal(result.logprob_token_ids, expected_token_ids), (
        f"Token IDs mismatch: {result.logprob_token_ids} vs {expected_token_ids}"
    )
    assert np.allclose(result.logprobs, expected_logprobs), (
        f"Logprobs mismatch: {result.logprobs} vs {expected_logprobs}"
    )
    assert np.array_equal(result.sampled_token_ranks, expected_ranks), (
        f"Ranks mismatch: {result.sampled_token_ranks} vs {expected_ranks}"
    )
    print("✓ PASS: Truncate single request test")


def test_end_thinking_token():
    """Test setting logprobs to 0.0 for end thinking tokens."""
    print("Test 3: Set logprobs to 0.0 for end thinking tokens")

    END_THINKING_TOKEN_ID = 999

    # Original: req0:[999], req1:[200,201], req2:[301] (999 is end thinking)
    original_token_ids = [999, 200, 201, 301]
    original_logprobs = [-2.0, -1.5, -2.5, -1.8]
    original_ranks = [0, 0, 1, 0]

    logprobs_lists = create_test_logprobs(
        original_token_ids, original_logprobs, original_ranks
    )
    valid_sampled_token_ids = [[999], [200, 201], [301]]
    token_adjustments: list[dict[str, Any]] = []

    result = _adjust_logprobs_for_force_end_thinking(
        logprobs_lists,
        token_adjustments,
        valid_sampled_token_ids,
        end_thinking_token_id=END_THINKING_TOKEN_ID,
    )

    # Expected: logprobs for token 999 (at index 0) should be 0.0 and rank should be 0
    expected_logprobs = [0.0, -1.5, -2.5, -1.8]
    expected_ranks = [0, 0, 1, 0]  # Rank for 999 changed from 1 to 0

    assert np.array_equal(result.logprob_token_ids, original_token_ids), (
        f"Token IDs mismatch: {result.logprob_token_ids} vs {original_token_ids}"
    )
    assert np.allclose(result.logprobs, expected_logprobs), (
        f"Logprobs mismatch: {result.logprobs} vs {expected_logprobs}"
    )
    assert np.array_equal(result.sampled_token_ranks, expected_ranks), (
        f"Ranks mismatch: {result.sampled_token_ranks} vs {expected_ranks}"
    )
    print("✓ PASS: End thinking token test")


def test_truncate_and_end_thinking_combined():
    """Test both truncation and end thinking token adjustment."""
    print("Test 4: Combined truncation and end thinking token")

    END_THINKING_TOKEN_ID = 999

    # Original: req0:[100,101], req1:[200,201,202,203], req2:[300,999]
    original_token_ids = [100, 101, 200, 201, 202, 203, 999]
    original_logprobs = [-1.0, -2.0, -1.5, -2.5, -3.0, -3.5, -2.2]
    original_ranks = [0, 1, 0, 1, 2, 3, 0]

    logprobs_lists = create_test_logprobs(
        original_token_ids, original_logprobs, original_ranks
    )

    # Truncate req1 from 4 tokens to 2, keep req2 with end thinking token
    valid_sampled_token_ids = [[100, 101], [200, 201], [999]]
    token_adjustments = [
        {"req_index": 1, "action": "truncate", "original_length": 4, "final_length": 2}
    ]

    result = _adjust_logprobs_for_force_end_thinking(
        logprobs_lists,
        token_adjustments,
        valid_sampled_token_ids,
        end_thinking_token_id=END_THINKING_TOKEN_ID,
    )

    # After truncation: remove indices 4,5 (202,203)
    # Remaining: [100,101,200,201,300,999] at indices [0,1,2,3,4,5]
    # Then set logprob for 999 (at final index 5) to 0.0 and rank to 0
    expected_token_ids = [100, 101, 200, 201, 999]
    expected_logprobs = [-1.0, -2.0, -1.5, -2.5, 0.0]  # Last one set to 0.0
    expected_ranks = [0, 1, 0, 1, 0]  # Last rank changed from 1 to 0

    assert np.array_equal(result.logprob_token_ids, expected_token_ids), (
        f"Token IDs mismatch: {result.logprob_token_ids} vs {expected_token_ids}"
    )
    assert np.allclose(result.logprobs, expected_logprobs), (
        f"Logprobs mismatch: {result.logprobs} vs {expected_logprobs}"
    )
    assert np.array_equal(result.sampled_token_ranks, expected_ranks), (
        f"Ranks mismatch: {result.sampled_token_ranks} vs {expected_ranks}"
    )
    print("✓ PASS: Combined truncation and end thinking test")


def test_edge_case_empty_requests():
    """Test edge case with empty data."""
    print("Test 5: Edge case - empty requests")

    logprobs_lists = create_test_logprobs([], [], [])
    valid_sampled_token_ids: list[list[int]] = []
    token_adjustments: list[dict[str, Any]] = []

    result = _adjust_logprobs_for_force_end_thinking(
        logprobs_lists, token_adjustments, valid_sampled_token_ids
    )

    assert len(result.logprob_token_ids) == 0
    assert len(result.logprobs) == 0
    assert len(result.sampled_token_ranks) == 0
    print("✓ PASS: Empty requests edge case test")


def test_multiple_truncate():
    """Test multiple truncation and end thinking token adjustment."""
    print("Test 4: Combined truncation and end thinking token")

    END_THINKING_TOKEN_ID = 999

    # Original: req0:[100,101], req1:[200,201,202,203], req2:[300,999]
    original_token_ids = [100, 101, 200, 201, 202, 203, 204, 205]
    original_logprobs = [-1.0, -2.0, -1.5, -2.5, -3.0, -3.5, -2.2, -2.1]
    original_ranks = [0, 1, 0, 1, 2, 3, 0, 1]

    logprobs_lists = create_test_logprobs(
        original_token_ids, original_logprobs, original_ranks
    )

    # Truncate req1 from 4 tokens to 2, keep req2 with end thinking token
    valid_sampled_token_ids = [[100, 101], [200, 201], [204]]
    token_adjustments = [
        {"req_index": 1, "action": "truncate", "original_length": 4, "final_length": 2},
        {"req_index": 2, "action": "truncate", "original_length": 2, "final_length": 1},
    ]

    result = _adjust_logprobs_for_force_end_thinking(
        logprobs_lists,
        token_adjustments,
        valid_sampled_token_ids,
        end_thinking_token_id=END_THINKING_TOKEN_ID,
    )

    expected_token_ids = [100, 101, 200, 201, 204]
    expected_logprobs = [-1.0, -2.0, -1.5, -2.5, -2.2]
    expected_ranks = [0, 1, 0, 1, 0]

    assert np.array_equal(result.logprob_token_ids, expected_token_ids), (
        f"Token IDs mismatch: {result.logprob_token_ids} vs {expected_token_ids}"
    )
    assert np.allclose(result.logprobs, expected_logprobs), (
        f"Logprobs mismatch: {result.logprobs} vs {expected_logprobs}"
    )
    assert np.array_equal(result.sampled_token_ranks, expected_ranks), (
        f"Ranks mismatch: {result.sampled_token_ranks} vs {expected_ranks}"
    )
    print("✓ PASS: Combined truncation and end thinking test")


def run_all_tests():
    """Run all tests and report results."""
    print("Running _adjust_logprobs_for_force_end_thinking unit tests...")
    print("=" * 60)

    tests = [
        test_no_adjustments,
        test_truncate_single_request,
        test_end_thinking_token,
        test_truncate_and_end_thinking_combined,
        test_edge_case_empty_requests,
        test_multiple_truncate,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAIL: {test_func.__name__} - {str(e)}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
