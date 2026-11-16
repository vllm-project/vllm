# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic tests pinning per-request acceptance totals."""

import pytest

from vllm.v1.metrics.acceptance import acceptance_totals

# Pure-CPU arithmetic: skip the repository-wide torch/GPU cleanup fixture.
pytestmark = pytest.mark.skip_global_cleanup


def test_empty_histogram_uses_neutral_defaults():
    t = acceptance_totals([0, 0, 0, 0], 0)
    assert t.num_spec_steps == 0
    assert t.num_accepted_draft_tokens == 0
    assert t.mean_acceptance_length == 1.0
    assert t.draft_acceptance_rate == 0.0


def test_all_rejected_steps():
    t = acceptance_totals([5, 0, 0, 0], 15)
    assert t.mean_acceptance_length == 1.0
    assert t.draft_acceptance_rate == 0.0


def test_all_accepted_steps():
    t = acceptance_totals([0, 0, 0, 4], 12)
    assert t.mean_acceptance_length == 4.0
    assert t.draft_acceptance_rate == 1.0


def test_zero_draft_tokens_keeps_rate_neutral():
    t = acceptance_totals([2, 1], 0)
    assert t.draft_acceptance_rate == 0.0
    assert t.mean_acceptance_length == pytest.approx(1.0 + 1 / 3)


def test_mixed_histogram_settles_to_literal_values():
    t = acceptance_totals([3, 2, 1, 0], 20)
    assert t.num_spec_steps == 6
    assert t.num_accepted_draft_tokens == 4
    assert t.mean_acceptance_length == pytest.approx(1.6666666666666667)
    assert t.draft_acceptance_rate == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("hist", "drafted", "steps", "accepted", "mean", "rate"),
    [
        ([0, 0], 0, 0, 0, 1.0, 0.0),
        ([1, 0], 2, 1, 0, 1.0, 0.0),
        ([0, 2], 4, 2, 2, 2.0, 0.5),
        ([2, 3, 1], 12, 6, 5, 1.8333333333333333, 0.4166666666666667),
        ([7, 0, 0, 5], 40, 12, 15, 2.25, 0.375),
    ],
)
def test_totals_are_pinned_to_literal_values(
    hist, drafted, steps, accepted, mean, rate
):
    """Literal expectations keep the totals verifiable without the engine."""
    t = acceptance_totals(hist, drafted)
    assert t.num_spec_steps == steps
    assert t.num_accepted_draft_tokens == accepted
    assert t.mean_acceptance_length == pytest.approx(mean)
    assert t.draft_acceptance_rate == pytest.approx(rate)
