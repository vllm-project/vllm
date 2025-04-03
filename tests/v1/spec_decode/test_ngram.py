# SPDX-License-Identifier: Apache-2.0

import numpy as np

from vllm.v1.spec_decode.ngram_proposer import (NgramProposer,
                                                _find_subarray_kmp,
                                                _kmp_lps_array)


def test_kmp_lps_array():
    np.testing.assert_array_equal(_kmp_lps_array(np.array([])), np.array([]))
    np.testing.assert_array_equal(_kmp_lps_array(np.array([1])), np.array([0]))
    np.testing.assert_array_equal(_kmp_lps_array(np.array([1, 1, 1])),
                                  np.array([0, 1, 2]))
    np.testing.assert_array_equal(_kmp_lps_array(np.array([1, 2, 3, 4])),
                                  np.array([0, 0, 0, 0]))
    np.testing.assert_array_equal(_kmp_lps_array(np.array([1, 2, 1, 2, 3])),
                                  np.array([0, 0, 1, 2, 0]))


def test_find_subarray_kmp():
    X = np.array([1, 2, 3, 4, 1, 2, 3, 5, 6])
    assert _find_subarray_kmp(X, 2, 2) is None
    X = np.array([1, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(_find_subarray_kmp(X, 2, 3),
                                  np.array([4, 1, 2]))
    np.testing.assert_array_equal(_find_subarray_kmp(X, 2, 2), np.array([4,
                                                                         1]))
    np.testing.assert_array_equal(_find_subarray_kmp(X, 1, 3),
                                  np.array([4, 1, 2]))
    np.testing.assert_array_equal(_find_subarray_kmp(X, 1, 2), np.array([4,
                                                                         1]))
    X = np.array([1, 3, 6, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(_find_subarray_kmp(X, 2, 3),
                                  np.array([4, 1, 2]))
    # Return on the first match
    np.testing.assert_array_equal(_find_subarray_kmp(X, 1, 3),
                                  np.array([6, 2, 3]))


def test_ngram_proposer():
    proposer = NgramProposer()

    # No match.
    result = proposer.propose(
        context_token_ids=np.array([1, 2, 3, 4, 5]),
        min_n=2,
        max_n=2,
        k=2,
    )
    assert result is None

    # No match for 4-gram.
    result = proposer.propose(
        context_token_ids=np.array([1, 2, 3, 4, 1, 2, 3]),
        min_n=4,
        max_n=4,
        k=2,
    )
    assert result is None

    # No match for 4-gram but match for 3-gram.
    result = proposer.propose(
        context_token_ids=np.array([1, 2, 3, 4, 1, 2, 3]),
        min_n=3,
        max_n=4,
        k=2,
    )
    assert np.array_equal(result, np.array([4, 1]))

    # Match for both 4-gram and 3-gram.
    # In this case, the proposer should return the 4-gram match.
    result = proposer.propose(
        context_token_ids=np.array([2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]),
        min_n=3,
        max_n=4,
        k=2,
    )
    assert np.array_equal(result, np.array([1, 2]))  # Not [5, 1]

    # Match for 2-gram and 3-gram, but not 4-gram.
    result = proposer.propose(
        context_token_ids=np.array([3, 4, 5, 2, 3, 4, 1, 2, 3, 4]),
        min_n=2,
        max_n=4,
        k=2,
    )
    assert np.array_equal(result, np.array([1, 2]))  # Not [5, 2]
