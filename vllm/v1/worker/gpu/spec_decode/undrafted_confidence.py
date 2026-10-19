# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confidence estimates for draft steps skipped by an early drafting stop."""

import numpy as np

# Pseudo-observations of the population mean each request starts with: until a
# request has drafted a step this many times, its estimate leans on the
# population's.
_PRIOR_COUNT = 20.0
# Most observations kept per request and step, and across all requests per step:
# older ones are scaled down so the means follow recent confidences.
_MAX_COUNT = 500.0
_MAX_POPULATION_COUNT = 10_000.0


class UndraftedConfidenceEstimator:
    """Per-request, per-step confidence estimates for steps past an early stop.

    A step's estimate is the running mean of that request's confidences in
    proposals that drafted the step, shrunk towards the same mean over all
    requests.
    """

    def __init__(self, max_num_reqs: int, num_steps: int):
        self.sums = np.zeros((max_num_reqs, num_steps))
        self.counts = np.zeros((max_num_reqs, num_steps))
        self.population_sums = np.zeros(num_steps)
        self.population_counts = np.zeros(num_steps)

    def update(
        self,
        confidences: np.ndarray,
        depths: np.ndarray,
        idx_mapping: np.ndarray,
    ) -> None:
        """Fold in one proposal: each request's first `depth` steps.

        `confidences` and `depths` are indexed by slot; `idx_mapping` holds the
        slots of the requests in the proposal.
        """
        if len(idx_mapping) == 0:
            return
        num_steps = self.sums.shape[1]
        drafted = np.arange(num_steps)[None, :] < depths[idx_mapping][:, None]
        observed = np.where(drafted, confidences[idx_mapping], 0.0)
        self.sums[idx_mapping] += observed
        self.counts[idx_mapping] += drafted
        self.population_sums += observed.sum(axis=0)
        self.population_counts += drafted.sum(axis=0)
        _cap(self.sums, self.counts, _MAX_COUNT)
        _cap(self.population_sums, self.population_counts, _MAX_POPULATION_COUNT)

    def reset(self, slot: int) -> None:
        """Forget a slot's counts, so a new request doesn't inherit them."""
        self.sums[slot] = 0.0
        self.counts[slot] = 0.0

    def mean(self, idx_mapping: np.ndarray) -> np.ndarray:
        # Steps no request has drafted yet start at 0.5.
        population = (self.population_sums + 0.5) / (self.population_counts + 1.0)
        return (self.sums[idx_mapping] + _PRIOR_COUNT * population) / (
            self.counts[idx_mapping] + _PRIOR_COUNT
        )


def _cap(sums: np.ndarray, counts: np.ndarray, limit: float) -> None:
    """Rescale to at most `limit` observations, keeping each mean."""
    over = counts > limit
    scale = limit / counts[over]
    sums[over] *= scale
    counts[over] *= scale
