# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact top-1 acceptance for fresh, same-prefix sampling scores."""

import math


def check_top1(
    reference_token: int,
    candidate_token: int,
    reference: dict[int, float],
    candidate: dict[int, float],
) -> bool:
    """Require a shared maximum; differing choices must tie exactly in both maps.

    Returns whether the candidate selected a different tied maximum. The caller
    must establish identical input prefixes and run the shared top-5 checker.
    """
    for scores in (reference, candidate):
        assert len(scores) == 5 and all(math.isfinite(x) for x in scores.values())
        assert reference_token in scores and candidate_token in scores
        assert (
            scores[reference_token] == scores[candidate_token] == max(scores.values())
        ), (reference_token, candidate_token, scores)
    return reference_token != candidate_token
