# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.ncp_olmo.sampler_reference import check_top1


@pytest.mark.parametrize("tied", [False, True])
def test_same_prefix_top1_accepts_only_maxima(tied):
    reference = {1: -0.5, 2: -0.5 if tied else -1.0, 3: -2.0, 4: -3.0, 5: -4.0}
    candidate = {token: score - 0.125 for token, score in reference.items()}
    assert check_top1(1, 2 if tied else 1, reference, candidate) is tied


@pytest.mark.parametrize(
    "invalid",
    ["near-reference", "near-candidate", "nonmaximum", "missing", "nonfinite"],
)
def test_same_prefix_top1_rejects_invalid_choices(invalid):
    reference = {1: -0.5, 2: -0.5, 3: -2.0, 4: -3.0, 5: -4.0}
    candidate = reference.copy()
    if invalid == "near-reference":
        reference[2] -= 1e-12
    elif invalid == "near-candidate":
        candidate[1] -= 1e-12
    elif invalid == "nonmaximum":
        reference[3] = candidate[3] = -0.25
    elif invalid == "missing":
        candidate[6] = candidate.pop(2)
    else:
        candidate[3] = float("nan")
    with pytest.raises(AssertionError):
        check_top1(1, 2, reference, candidate)
