# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic arithmetic for vLLM's per-request acceptance metrics.

The fields under ``metrics.speculative_decoding`` are designed to be
recomputable from the accepted-count histogram alone, without an engine.
Keeping that arithmetic in one dependency-free function makes the documented
formula, the response payload and the tests share a single implementation.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptanceTotals:
    num_spec_steps: int
    num_accepted_draft_tokens: int
    mean_acceptance_length: float
    draft_acceptance_rate: float


def acceptance_totals(
    histogram: Sequence[int], num_draft_tokens: int
) -> AcceptanceTotals:
    """Compute per-request acceptance totals from a dense accepted-count histogram.

    ``histogram[j]`` counts verification steps that accepted exactly ``j`` draft
    tokens. ``mean_acceptance_length`` is the headline field (including the
    bonus token); ``draft_acceptance_rate`` is draft-only at full precision. An
    empty histogram or zero drafted tokens resolves to the neutral values
    carried by the response payload.
    """
    num_spec_steps = sum(histogram)
    num_accepted = sum(j * count for j, count in enumerate(histogram))
    mean_al = 1.0 + num_accepted / num_spec_steps if num_spec_steps else 1.0
    rate = num_accepted / num_draft_tokens if num_draft_tokens else 0.0
    return AcceptanceTotals(
        num_spec_steps=num_spec_steps,
        num_accepted_draft_tokens=num_accepted,
        mean_acceptance_length=mean_al,
        draft_acceptance_rate=rate,
    )
