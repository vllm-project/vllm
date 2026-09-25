# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.v1.spec_decode.ngram_hint_proposer import find_hint_match

HINT = [100, 101, 102, 103, 104, 105]


def _context(*token_ids):
    return np.array(token_ids, dtype=np.int32)


def test_match_proposes_the_continuation():
    assert find_hint_match([HINT], _context(1, 2, 100, 101, 102), 3, 16) == [
        103,
        104,
        105,
    ]
    # A shorter suffix also matches.
    assert find_hint_match([HINT], _context(7, 103, 104), 3, 16) == [105]


def test_single_token_only_matches_at_the_start_of_a_hint():
    # 100 is the first token of the hint.
    assert find_hint_match([HINT], _context(1, 2, 100), 1, 16) == [
        101,
        102,
        103,
        104,
        105,
    ]
    # 103 is in the middle of the hint, so a lone 103 does not match.
    assert find_hint_match([HINT], _context(1, 2, 103), 1, 16) == []


def test_no_match():
    assert find_hint_match([HINT], _context(1, 2, 3, 4), 3, 16) == []
    # A match at the end of the hint has nothing to propose.
    assert find_hint_match([HINT], _context(103, 104, 105), 3, 16) == []


def test_k_limits_the_proposal():
    assert find_hint_match([HINT], _context(100, 101), 3, 2) == [102, 103]
    assert find_hint_match([HINT], _context(100, 101, 102), 3, 0) == []


def test_longest_suffix_wins_across_hints():
    other = [200, 101, 102, 201]
    assert find_hint_match([HINT, other], _context(1, 100, 101, 102), 3, 16) == [
        103,
        104,
        105,
    ]
    assert find_hint_match([HINT, other], _context(1, 200, 101, 102), 3, 16) == [201]


def test_empty_inputs():
    assert find_hint_match([], _context(100, 101, 102), 3, 16) == []
    assert find_hint_match([[]], _context(100, 101, 102), 3, 16) == []
    assert find_hint_match([HINT], _context(), 3, 16) == []
