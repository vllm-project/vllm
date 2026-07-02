# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark hardware-aware confidence scheduler policy.

Owns all scheduler state for the DSpark speculator: the shape-aware cost table,
the online engine-overhead regression, the survival EMA + calibration, the
hysteresis incumbent, and per-request width allocation. The speculator keeps
only the GPU-graph-written survival buffer and the width plumbing the runner
consumes, driving this policy through a thin method surface.
"""

import torch

from vllm.config import SpeculativeConfig
from vllm.v1.spec_decode.dynamic.utils import (
    DEFAULT_SURVIVAL_PRIOR,
    derive_dynamic_sd_schedule,
)

# Survival EMA: slow blend so the per-step length choice does not chase jitter.
SURVIVAL_EMA_DECAY = 0.9
SURVIVAL_EMA_GAIN = 0.1
# Consume/launch the host survival readout every N steps (staleness-tolerant).
READOUT_CADENCE = 4
# Online calibration ratio EMA (realized/predicted survival).
CALIBRATION_DECAY = 0.8
CALIBRATION_GAIN = 0.2
# Minimum observations in an interval before trusting a calibration update.
CALIBRATION_MIN_OBS = 64.0
# Clamp the realized/predicted ratio to reject noisy intervals.
CALIBRATION_RATIO_MIN = 0.25
CALIBRATION_RATIO_MAX = 4.0
# Overhead regression ring: bounded sample history for the (c0, c1) fit.
OVERHEAD_RING_SIZE = 129
# Discard overhead residuals above this many seconds as idle-gap outliers.
OVERHEAD_SAMPLE_GATE = 0.25
# Minimum ring occupancy before the regression is meaningful.
OVERHEAD_MIN_SAMPLES = 16
# Cap the per-token overhead slope c1 (a physical sampler/detok upper bound).
OVERHEAD_C1_CLAMP = 1e-3
# Keep the incumbent length unless a challenger clears this relative margin.
HYSTERESIS_MARGIN = 0.05
# Prefix-survival prior (measured V4-Flash/Qwen3) seeding the dynamic-SD table;
# online calibration corrects the runtime estimate, not this table.
SURVIVAL_PRIOR = DEFAULT_SURVIVAL_PRIOR
# Engine-overhead prior (per-step, per-token seconds) for the dynamic-SD table.
DERIVATION_OVERHEAD_C0 = 0.003
DERIVATION_OVERHEAD_C1 = 5e-5


def schedule_uniform_length(
    accept: list[float],
    num_reqs: int,
    gamma: int,
    r_grid: list[int] | None,
    times_by_l: list[list[float]] | None,
    overhead: tuple[float, float] = (0.0, 0.0),
    current: int | None = None,
    hysteresis: float = HYSTERESIS_MARGIN,
) -> tuple[int, float]:
    """Hardware-aware per-step *uniform* verify length L in [0, gamma].

    Picks the batch-uniform L maximizing accepted-tokens / step-time from
    ``accept`` (``accept[L]`` = summed prefix-survival over the first L
    positions) and the shape-aware cost table ``times_by_l[L][i]`` = step time
    at ``r_grid[i]`` requests for verify query length ``1+L``. Each L is a
    fixed CUDA-graph-captured shape; L=0 skips drafting; falls back to gamma
    without a table. Shape-awareness matters: full-gamma verify replays the
    fast FULL decode graph while a trimmed L may fall to PIECEWISE, and the
    eager cutoff depends on (R, L) -- a 1-D T(num_tokens) expresses neither.

    ``overhead`` is (c0, c1): per-step constant + per-generated-token cost.
    Folding c1 into c0 would wrongly pin L at gamma.

    Returns (best_l, predicted step time of best_l including overhead).
    """
    if not r_grid or not times_by_l:
        return gamma, 0.0

    def cost(length: int) -> float:
        # Ceiling-bucket lookup, NOT interpolation: dispatch pads R up to the next
        # captured bucket, so cost is a step function of R; interpolating smooths
        # across the FULL-graph/eager cliff and underestimates eager shapes.
        row = times_by_l[length]
        for i, r in enumerate(r_grid):
            if num_reqs <= r:
                return row[i]
        return row[-1] * (num_reqs / r_grid[-1])

    c0, c1 = overhead
    best_l, best_tput, best_cost = gamma, -1.0, 0.0
    cur_tput, cur_cost = -1.0, 0.0
    for length in range(gamma + 1):
        c = cost(length)
        if c <= 0.0:
            continue
        tokens = num_reqs * (1.0 + accept[length])
        c += c0 + c1 * tokens
        tput = tokens / c
        if tput > best_tput:
            best_tput, best_l, best_cost = tput, length, c
        if length == current:
            cur_tput, cur_cost = tput, c
    # Hysteresis: re-evaluated at the current num_reqs so genuine load shifts
    # still switch immediately, but survival-EMA jitter cannot flap the length.
    if (
        current is not None
        and cur_tput > 0.0
        and best_tput <= cur_tput * (1.0 + hysteresis)
    ):
        return current, cur_cost
    return best_l, best_cost


def derive_dynamic_sd_table(
    r_grid: list[int],
    times_by_l: list[list[float]],
    gamma: int,
    max_num_reqs: int,
) -> list[tuple[int, int, int]]:
    """Derive the dynamic-SD batch-size table from the startup profile:
    per-bucket throughput-optimal K under the survival/overhead priors,
    adjacent equal-K buckets collapsed (ceiling-bucket semantics matching
    dispatch, so every scheduled K runs on a captured shape).
    """
    return derive_dynamic_sd_schedule(
        r_grid,
        times_by_l,
        list(SURVIVAL_PRIOR[:gamma]),
        max_num_reqs,
        overhead_c0=DERIVATION_OVERHEAD_C0,
        overhead_c1=DERIVATION_OVERHEAD_C1,
    )


def allocate_widths(
    survival_view: torch.Tensor,
    cal_t: torch.Tensor,
    num_reqs: int,
    length: int,
    tau: float,
    budget_frac: float,
) -> torch.Tensor:
    """Per-request keep lengths (paper Algorithm 1) via a global survival top-k.

    With ``tau > 0`` returns a static confidence-threshold width; otherwise
    distributes ``num_reqs*length*budget_frac`` tokens over the freshest
    calibrated survivals. Survival is a cumprod, so thresholding yields
    valid prefixes.
    """
    a = survival_view * cal_t
    if tau > 0.0:
        return (a >= tau).sum(dim=1, dtype=torch.int32)
    a = a[:, :length]
    budget = min(int(num_reqs * length * budget_frac) + 1, num_reqs * length)
    flat = a.reshape(-1)
    if budget < flat.numel():
        th = torch.topk(flat, budget, sorted=False).values.min()
        return (a >= th).sum(dim=1, dtype=torch.int32)
    return torch.full((num_reqs,), length, dtype=torch.int32, device=a.device)


class DSparkScheduler:
    """Confidence-scheduler policy for the DSpark speculator (see module docstring)."""

    def __init__(
        self,
        spec_config: SpeculativeConfig,
        gamma: int,
        max_num_reqs: int,
        device: torch.device,
    ):
        self.gamma = gamma
        self.device = device
        # CPU-testability seam: on CPU use a pageable host buffer and an
        # always-ready readout (copies are synchronous) so the stateful core is
        # unit-testable without a GPU; the CUDA path is byte-identical.
        self._is_cuda = device.type == "cuda"
        self.perreq = spec_config.dspark_per_request
        self.tau = spec_config.dspark_confidence_threshold
        self._budget_frac = spec_config.dspark_budget_frac

        # Shape-aware cost table T(R, L), installed after CUDA graph capture.
        self._cost_r_grid: list[int] | None = None
        self._cost_times_by_l: list[list[float]] | None = None

        # Survival EMA: on-GPU accumulator + stale CPU readout consumed by decide.
        self._surv_ema_t: torch.Tensor | None = None
        self._surv_ema: list[float] | None = None
        self._ema_ct = 0
        # Host readout rows: [survival EMA; realized-acceptance counts; observation
        # counts; predicted-survival sum]; the latter three feed online calibration.
        self._surv_host = torch.empty(
            4, gamma, dtype=torch.float32, pin_memory=self._is_cuda
        )
        self._surv_evt = torch.cuda.Event() if self._is_cuda else None
        self._surv_inflight = False

        # Calibration state (realized vs predicted survival over the same set).
        self._acc_counts = torch.zeros(gamma, dtype=torch.float32, device=device)
        self._obs_counts = torch.zeros(gamma, dtype=torch.float32, device=device)
        self._pred_counts = torch.zeros(gamma, dtype=torch.float32, device=device)
        self._surv_state = torch.zeros(
            max_num_reqs, gamma, dtype=torch.float32, device=device
        )
        self._pos_arange = torch.arange(1, gamma + 1, dtype=torch.int32, device=device)
        self._cal = [1.0] * gamma
        self._cal_prev_acc = [0.0] * gamma
        self._cal_prev_obs = [0.0] * gamma
        self._cal_prev_pred = [0.0] * gamma
        self._cal_t = torch.ones(gamma, dtype=torch.float32, device=device)

        # Online engine-overhead regression O = c0 + c1 * generated_tokens over a
        # ring of (tokens, residual) samples (median-free least squares).
        self._o_c0 = 0.0
        self._o_c1 = 0.0
        self._o_samples: list[tuple[float, float]] = []
        self._last_t: float | None = None
        self._last_pred = 0.0
        self._last_gpu_pred = 0.0
        self._last_tokens = 0.0
        # Hysteresis incumbent and last committed verify length.
        self._last_l: int | None = None
        self._prev_sched_l = 0

    def set_cost_table(self, r_grid: list[int], times_by_l: list[list[float]]) -> None:
        """Install the shape-aware cost table T(R, L) [seconds]."""
        if r_grid and times_by_l and all(len(row) == len(r_grid) for row in times_by_l):
            self._cost_r_grid = list(r_grid)
            self._cost_times_by_l = [list(row) for row in times_by_l]

    def observe_verified(
        self,
        num_reqs: int,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        prev_widths: torch.Tensor,
        idx_mapping: torch.Tensor,
    ) -> None:
        # Accumulate realized prefix survival from the step just verified: request
        # r survived position j iff it accepted >= j draft tokens; position j was
        # observed iff verified at its full scheduled width and j <= width_r.
        if self._prev_sched_l <= 0:
            return
        prev = prev_widths[idx_mapping]
        full = (num_sampled[:num_reqs] + num_rejected[:num_reqs]) == (1 + prev)
        accepted = (num_sampled[:num_reqs] - 1).clamp_(min=0)
        observed_j = (prev.unsqueeze(1) >= self._pos_arange) & full.unsqueeze(1)
        surv_j = (accepted.unsqueeze(1) >= self._pos_arange) & observed_j
        self._acc_counts += surv_j.sum(dim=0, dtype=torch.float32)
        self._obs_counts += observed_j.sum(dim=0, dtype=torch.float32)
        self._pred_counts += (self._surv_state[idx_mapping] * observed_j).sum(dim=0)

    def begin_step(self, now: float, num_reqs: int) -> int:
        """Observe engine overhead, then choose this step's uniform verify length."""
        # Overhead observation: wall dt minus the prior step's GPU-only prediction,
        # regressed against generated tokens (the O the GPU table cannot see).
        if self._last_t is not None and self._last_gpu_pred > 0.0:
            o = (now - self._last_t) - self._last_gpu_pred
            if 0.0 <= o < OVERHEAD_SAMPLE_GATE:
                self._o_samples.append((self._last_tokens, o))
                if len(self._o_samples) > OVERHEAD_RING_SIZE:
                    del self._o_samples[0]
                if len(self._o_samples) >= OVERHEAD_MIN_SAMPLES:
                    n = len(self._o_samples)
                    mx = sum(x for x, _ in self._o_samples) / n
                    my = sum(y for _, y in self._o_samples) / n
                    var = sum((x - mx) ** 2 for x, _ in self._o_samples)
                    if var > 1e-6:
                        cov = sum((x - mx) * (y - my) for x, y in self._o_samples)
                        self._o_c1 = min(max(cov / var, 0.0), OVERHEAD_C1_CLAMP)
                    self._o_c0 = max(my - self._o_c1 * mx, 0.0)
        self._last_t = now

        length = self.gamma
        if self._surv_ema is not None:
            # accept[L] = expected accepted draft tokens/req (cumulative CALIBRATED
            # prefix-survival over the first L positions).
            accept = [0.0]
            run = 0.0
            for s, k in zip(self._surv_ema, self._cal):
                run += min(s * k, 1.0)
                accept.append(run)
            length, self._last_pred = schedule_uniform_length(
                accept,
                num_reqs,
                self.gamma,
                self._cost_r_grid,
                self._cost_times_by_l,
                overhead=(self._o_c0, self._o_c1),
                current=self._last_l,
            )
            self._last_l = length
            self._last_tokens = num_reqs * (1.0 + accept[length])
            self._last_gpu_pred = self._last_pred - (
                self._o_c0 + self._o_c1 * self._last_tokens
            )
        return length

    def skip_next_overhead_sample(self) -> None:
        # Engine-forced width (dynamic SD): no GPU-cost prediction was made, so the
        # next overhead observation has nothing to regress against.
        self._last_gpu_pred = 0.0

    def update_survival(
        self, survival_batch_view: torch.Tensor, idx_mapping: torch.Tensor
    ) -> None:
        # EMA the drafted step's survival, consume any landed host readout (+ online
        # calibration), launch the next readout on cadence, and persist predictions
        # req-state-indexed for the next step's realized-vs-predicted accumulation.
        m = survival_batch_view.mean(dim=0)
        if self._surv_ema_t is None:
            self._surv_ema_t = m.clone()
        else:
            self._surv_ema_t.mul_(SURVIVAL_EMA_DECAY).add_(m, alpha=SURVIVAL_EMA_GAIN)
        self._ema_ct += 1
        # On CPU the host copies are synchronous, so a launched readout is always
        # ready; on CUDA this is exactly self._surv_evt.query() (byte-identical).
        if self._surv_inflight and (not self._is_cuda or self._surv_evt.query()):
            rows = self._surv_host.tolist()
            self._surv_ema = rows[0]
            cal_moved = False
            for j in range(self.gamma):
                d_obs = rows[2][j] - self._cal_prev_obs[j]
                d_acc = rows[1][j] - self._cal_prev_acc[j]
                d_pred = rows[3][j] - self._cal_prev_pred[j]
                if d_obs >= CALIBRATION_MIN_OBS and d_pred > 1e-3:
                    ratio = min(
                        max(d_acc / d_pred, CALIBRATION_RATIO_MIN),
                        CALIBRATION_RATIO_MAX,
                    )
                    self._cal[j] = (
                        CALIBRATION_DECAY * self._cal[j] + CALIBRATION_GAIN * ratio
                    )
                    self._cal_prev_obs[j] = rows[2][j]
                    self._cal_prev_acc[j] = rows[1][j]
                    self._cal_prev_pred[j] = rows[3][j]
                    cal_moved = True
            if cal_moved:
                self._cal_t.copy_(torch.tensor(self._cal, dtype=torch.float32))
            self._surv_inflight = False
        if not self._surv_inflight and self._ema_ct % READOUT_CADENCE == 0:
            self._surv_host[0].copy_(self._surv_ema_t, non_blocking=True)
            self._surv_host[1].copy_(self._acc_counts, non_blocking=True)
            self._surv_host[2].copy_(self._obs_counts, non_blocking=True)
            self._surv_host[3].copy_(self._pred_counts, non_blocking=True)
            if self._is_cuda:
                self._surv_evt.record()
            self._surv_inflight = True
        self._surv_state[idx_mapping] = survival_batch_view

    def allocate(
        self, survival_batch_view: torch.Tensor, num_reqs: int, length: int
    ) -> torch.Tensor:
        return allocate_widths(
            survival_batch_view,
            self._cal_t,
            num_reqs,
            length,
            self.tau,
            self._budget_frac,
        )

    def confidence_widths(
        self, survival_batch_view: torch.Tensor, length: int
    ) -> torch.Tensor:
        # Static confidence-threshold prefix width (survival is a cumprod).
        return (survival_batch_view[:, :length] >= self.tau).sum(
            dim=1, dtype=torch.int32
        )

    def commit_length(self, length: int) -> None:
        self._prev_sched_l = length
