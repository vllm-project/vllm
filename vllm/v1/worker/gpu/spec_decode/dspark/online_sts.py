# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np


class DSparkOnlineSTS:
    """Online Sequential Temperature Scaling for DSpark confidences.

    Per-position temperatures minimize binned calibration error. Sparse
    positions blend toward a pooled fit; cold start is the identity.
    """

    DECAY = 0.999
    PRIOR_WEIGHT = 64.0
    NUM_BINS = 16
    LOGIT_RANGE = 8.0
    # Log-spaced temperature grid; 1.0 is on the grid so a well-calibrated
    # head fits the identity exactly.
    TEMP_GRID_MIN = 0.125
    TEMP_GRID_MAX = 8.0
    TEMP_GRID_SIZE = 49

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        # EMA counters per (position, logit bin).
        self.bin_trials = np.zeros((num_steps, self.NUM_BINS), dtype=np.float64)
        self.bin_hits = np.zeros_like(self.bin_trials)
        # Per-position temperatures; identity until observations accumulate.
        self.temperatures = np.ones(num_steps, dtype=np.float64)

        bin_width = 2 * self.LOGIT_RANGE / self.NUM_BINS
        self._bin_mids = (
            -self.LOGIT_RANGE + (np.arange(self.NUM_BINS) + 0.5) * bin_width
        )
        self._temp_grid = np.logspace(
            np.log10(self.TEMP_GRID_MIN),
            np.log10(self.TEMP_GRID_MAX),
            self.TEMP_GRID_SIZE,
        )
        # sigmoid(mid_b / T) for every (T, bin) pair, fixed for the run.
        self._grid_probs = 1.0 / (
            1.0 + np.exp(-self._bin_mids[None, :] / self._temp_grid[:, None])
        )
        self._log_temp_grid = np.log(self._temp_grid)

    def reset(self) -> None:
        """Discard accumulated observations (e.g. from warmup/profiling
        steps) and return to the cold-start identity temperatures."""
        self.bin_trials.fill(0.0)
        self.bin_hits.fill(0.0)
        self.temperatures.fill(1.0)

    def update(
        self,
        confidence_logits: np.ndarray,
        num_accepted: np.ndarray,
        num_verified: np.ndarray,
    ) -> None:
        """Fold one verification step's outcomes into the calibration.

        Args:
            confidence_logits: [num_reqs, num_steps] raw head logits of the
                drafts that were verified.
            num_accepted: [num_reqs] accepted draft tokens.
            num_verified: [num_reqs] draft tokens that were verified
                (post-capacity), zero for rows without drafts.
        """
        if confidence_logits.shape[0] == 0:
            return
        num_cells = self.num_steps * self.NUM_BINS
        bin_width = 2 * self.LOGIT_RANGE / self.NUM_BINS
        bin_idx = ((confidence_logits + self.LOGIT_RANGE) / bin_width).astype(np.int64)
        np.clip(bin_idx, 0, self.NUM_BINS - 1, out=bin_idx)

        k = np.arange(self.num_steps)[None, :]
        # Position k (0-based) is evaluated iff the k-token prefix before it
        # was accepted and it was inside the verified capacity.
        trial = k < np.minimum(num_accepted + 1, num_verified)[:, None]
        hit = (k < num_accepted[:, None]) & trial

        cell = k * self.NUM_BINS + bin_idx  # [reqs, steps]
        shape = (self.num_steps, self.NUM_BINS)
        self.bin_trials *= self.DECAY
        self.bin_trials += np.bincount(cell[trial], minlength=num_cells).reshape(shape)
        self.bin_hits *= self.DECAY
        self.bin_hits += np.bincount(cell[hit], minlength=num_cells).reshape(shape)

        # Per-position grid search: T_k minimizing trial-weighted ECE of
        # sigmoid(mid_b / T) against the empirical bin acceptance.
        emp = self.bin_hits / np.maximum(self.bin_trials, 1e-6)
        err = np.abs(self._grid_probs[:, None, :] - emp[None])  # [T, steps, bins]
        ece = (err * self.bin_trials[None]).sum(-1)  # [T, steps]
        log_t = self._log_temp_grid[ece.argmin(0)]
        # Blend data-starved positions toward a pooled fit.
        pooled_trials = self.bin_trials.sum(0)
        pooled_emp = self.bin_hits.sum(0) / np.maximum(pooled_trials, 1e-6)
        pooled_err = np.abs(self._grid_probs - pooled_emp[None])  # [T, bins]
        pooled_ece = (pooled_err * pooled_trials[None]).sum(-1)  # [T]
        pooled_log_t = self._log_temp_grid[int(pooled_ece.argmin())]
        pooled_total = pooled_trials.sum()
        pooled_log_t *= pooled_total / (pooled_total + self.PRIOR_WEIGHT)
        # Per-position blend toward the pooled fit by observation count.
        total = self.bin_trials.sum(-1)
        weight = total / (total + self.PRIOR_WEIGHT)
        self.temperatures = np.exp(weight * log_t + (1.0 - weight) * pooled_log_t)
