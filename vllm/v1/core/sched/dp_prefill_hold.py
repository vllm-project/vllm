# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost-based prefill hold, across data parallel ranks where they step together.

Each step, every rank reports a compact `_RankRow` (`PrefillHold.local_row`)
and runs the deterministic `PrefillHold.update` on all ranks' rows. With DP
attention + EP, all ranks step in lockstep, so a prefill on any rank slows the
step for every rank's running decodes: `_RankRow` is piggybacked on the DP
state sync all-reduce so all ranks hold or release together without extra RPCs.
Without DP (and on independent dense DP ranks, which run with DP size 1), the
scheduler feeds its own row to `PrefillHold.update` before each step.

The hold fits an online non-negative 3-parameter mixed-step overhead model
`extra_ms(M, S) = F + a * M + b * S` (fixed overhead, slowest-rank tokens `M`,
total tokens `S`; reducing to 2D `F + a * M` on a single rank), then applies
Online Ski-Rental against the amortizable waste `(F, a)` while bounding each
request's hold time by the TTFT target `T`.
"""

import math
import time
from collections import deque
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Pure-decode step time EWMA time constant, in seconds.
_DEC_TAU_S = 10.0
# Mixed-step exponential forgetting window, in steps.
_FIT_WINDOW = 200.0
# Step periods beyond this multiple of the current estimate are treated as
# stalls until `_MAX_STALL_REJECTS` consecutive samples indicate a regime shift.
_STALL_FACTOR = 3.0
_MAX_STALL_REJECTS = 5
# Cold-start pure-decode samples collected before switching to EWMA.
_DEC_INIT_SAMPLES = 3
# Upper bound on startable waiting requests scanned per step.
_MAX_QUEUE_SCAN = 256
# Periodic summary log interval on DP rank 0, in seconds.
_LOG_INTERVAL_S = 30.0


class _RankRow(NamedTuple):
    """Per-rank state all-gathered each step in `sync_dp_state`."""

    done: float
    period: float
    done_tokens: float
    work: float
    x: float
    decodes: float
    queued: float
    slack: float


ROW_SIZE = len(_RankRow._fields)


def _solve_3x3(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve a regularized 3x3 linear system with partial pivoting.

    Returns None if the system is ill-conditioned.
    """
    col_norms = [max(abs(a[i][k]) for i in range(3)) for k in range(3)]
    m = [
        [a[i][j] + (1e-6 * max(1.0, a[i][i]) if i == j else 0.0) for j in range(3)]
        + [b[i]]
        for i in range(3)
    ]
    for k in range(3):
        if col_norms[k] < 1e-12:
            return None
        p = max(range(k, 3), key=lambda i: abs(m[i][k]))
        piv = m[p][k]
        if abs(piv) < 1e-4 * col_norms[k]:
            return None
        if p != k:
            m[k], m[p] = m[p], m[k]
        for j in range(k, 4):
            m[k][j] /= piv
        for i in range(3):
            if i != k and m[i][k] != 0.0:
                factor = m[i][k]
                for j in range(k, 4):
                    m[i][j] -= factor * m[k][j]
    return [m[0][3], m[1][3], m[2][3]]


class _Cost3D:
    """Online non-negative WLS for `extra_ms(M, S) = F + a * M + b * S`.

    `M = max_r x_r / 1e3` (busiest-rank ktok) and `S = sum_r x_r / 1e3`
    (total ktok across ranks). Uses Huber 3-sigma downweighting and an
    exponential forgetting window of `_FIT_WINDOW` mixed steps.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.a_mat = [[0.0] * 3 for _ in range(3)]
        self.b_vec = [0.0, 0.0, 0.0]
        self.w = 0.0
        self.mean_y = 0.0
        self.f = 0.0
        self.a = 0.0
        self.b = 0.0
        self.sig: float | None = None
        self.s_minus_m_sq = 0.0
        self.has_2d = False
        self.has_3d = False

    def predict_ms(self, m_ktok: float, s_ktok: float) -> float:
        return self.f + self.a * m_ktok + self.b * s_ktok

    def _rescale(self, m_bar: float, s_bar: float) -> None:
        pred_bar = self.predict_ms(m_bar, s_bar)
        if pred_bar > 1e-6 and self.mean_y > 0.0:
            scale = self.mean_y / pred_bar
            self.f *= scale
            self.a *= scale
            self.b *= scale

    def update(self, m_ktok: float, s_ktok: float, y_ms: float) -> None:
        y_ms = max(0.0, y_ms)
        err = abs(y_ms - self.predict_ms(m_ktok, s_ktok))
        wt = (
            3.0 * self.sig / err
            if self.sig is not None and err > 3.0 * self.sig and err > 0.0
            else 1.0
        )
        self.sig = err if self.sig is None else 0.95 * self.sig + 0.05 * err

        keep = 1.0 - 1.0 / _FIT_WINDOW
        self.w = keep * self.w + wt
        self.s_minus_m_sq = keep * self.s_minus_m_sq + wt * (s_ktok - m_ktok) ** 2
        phi = (1.0, m_ktok, s_ktok)
        for i in range(3):
            self.b_vec[i] = keep * self.b_vec[i] + wt * phi[i] * y_ms
            for j in range(3):
                self.a_mat[i][j] = keep * self.a_mat[i][j] + wt * phi[i] * phi[j]
        self.mean_y = self.b_vec[0] / self.w

        sol = None
        if self.w >= 4.0 and self.s_minus_m_sq / self.w > 1e-3:
            sol = _solve_3x3(self.a_mat, self.b_vec)

        if sol is not None:
            f, a, b = sol
            if f < 0.0 or a < 0.0 or b < 0.0:
                f, a, b = self._nnls_active_set()
            self.f, self.a, self.b = f, a, b
            self.has_3d = True
            return

        m_bar = self.a_mat[0][1] / self.w
        s_bar = self.a_mat[0][2] / self.w
        if self.has_3d:
            # Steady-state collinearity (e.g. repeated full batches): preserve
            # the learned (F, a, b) ratios and rescale gently to mean_y.
            self._rescale(m_bar, s_bar)
            return

        # 2D fallback (single rank or before multi-rank variation): F + a * M.
        var_m = self.a_mat[1][1] / self.w - m_bar * m_bar
        if self.w < 2.0 or var_m <= 1e-4:
            if self.has_2d:
                self._rescale(m_bar, s_bar)
            else:
                self.f, self.a, self.b = max(0.0, self.mean_y), 0.0, 0.0
            return
        cov_my = self.b_vec[1] / self.w - m_bar * self.mean_y
        a = max(0.0, cov_my / var_m)
        f = self.mean_y - a * m_bar
        if f < 0.0:
            f = 0.0
            a = max(0.0, self.b_vec[1] / max(self.a_mat[1][1], 1e-12))
        self.f, self.a, self.b = f, a, 0.0
        self.has_2d = True

    def _nnls_active_set(self) -> tuple[float, float, float]:
        """Exact non-negative least squares over 1- and 2-variable subsets."""
        best_obj: float | None = None
        best_c: tuple[float, float, float] | None = None
        for mask in range(1, 7):
            idx = [i for i in range(3) if mask & (1 << i)]
            if len(idx) == 1:
                i0 = idx[0]
                diag = self.a_mat[i0][i0] + 1e-6 * max(1.0, self.a_mat[i0][i0])
                vals = [self.b_vec[i0] / diag]
            else:
                i0, i1 = idx
                a00 = self.a_mat[i0][i0] + 1e-6 * max(1.0, self.a_mat[i0][i0])
                a11 = self.a_mat[i1][i1] + 1e-6 * max(1.0, self.a_mat[i1][i1])
                a01 = self.a_mat[i0][i1]
                det = a00 * a11 - a01 * a01
                if abs(det) < 1e-8 * a00 * a11:
                    continue
                b0, b1 = self.b_vec[i0], self.b_vec[i1]
                vals = [(b0 * a11 - b1 * a01) / det, (a00 * b1 - a01 * b0) / det]
            if any(v < 0.0 for v in vals):
                continue
            cand = [0.0, 0.0, 0.0]
            for i, v in zip(idx, vals):
                cand[i] = v
            obj = sum(
                cand[i]
                * (
                    sum(self.a_mat[i][j] * cand[j] for j in range(3))
                    - 2.0 * self.b_vec[i]
                )
                for i in range(3)
            )
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_c = (cand[0], cand[1], cand[2])
        if best_c is not None:
            return best_c
        return max(0.0, self.mean_y), 0.0, 0.0


class PrefillQueueState(NamedTuple):
    """A read-only view of the scheduler's queues for the hold."""

    running: Sequence["Request"]
    # Running requests with prefill left.
    prefill_chunks: Collection["Request"]
    # Waiting requests that can start now, in the order the scheduler admits
    # them.
    waiting: Callable[[], Iterator["Request"]]
    # Requests in the waiting queues, whether or not they can start.
    num_waiting: int
    max_active_reqs: int
    num_cached_tokens: Callable[["Request"], int]
    # Free KV blocks if the last release stopped for lack of them, else None.
    kv_free_blocks: int | None
    # Estimated KV blocks needed to admit `(request, cached_tokens)`.
    num_blocks_needed: Callable[["Request", int], int]
    # Effective per-request prefill token cap for the next step (0 = uncapped).
    long_prefill_token_threshold: int = 0


class PrefillHold:
    """Decides, once per step, whether to hold prefills on the next step.

    With MoE DP, the DP engine core all-gathers every rank's `local_row` and
    feeds them to `update`, so all ranks decide identically. Without DP (or on
    independent dense DP ranks), the scheduler calls `update([local_row()])`
    before each step.

    Args:
        queue_state: Returns a read-only view of the scheduler's queues.
        ttft_target_s: TTFT target `T` in seconds.
        ttft_weight: Weight on waiting requests' TTFT against running decodes'
            stall time. `1.0` targets end-to-end latency; higher values hold
            less (favouring TTFT), lower values hold longer (favouring ITL).
        token_budget: Max scheduled tokens per step per rank.
        input_budget: Max model-input tokens per step per rank. Defaults to
            `token_budget`.
        decode_tokens_per_req: Scheduled tokens consumed per running decode
            request (`1 + num_spec_tokens`).
        draft_slots_per_req: Extra input-budget slots reserved per request for
            drafting.
        lag_steps: Steps between a decision and the step it takes effect on
            the GPU beyond the next one (`max_concurrent_batches - 1`).
        log_stats: Whether this rank logs a periodic summary; DP rank 0 does.

    """

    def __init__(
        self,
        queue_state: Callable[[], PrefillQueueState],
        ttft_target_s: float,
        ttft_weight: float,
        token_budget: int,
        input_budget: int | None = None,
        decode_tokens_per_req: int = 1,
        draft_slots_per_req: int = 0,
        lag_steps: int = 0,
        log_stats: bool = False,
    ) -> None:
        self._queue_state = queue_state
        self.ttft_target = ttft_target_s
        self.ttft_weight = ttft_weight
        self.token_budget = token_budget
        self.input_budget = token_budget if input_budget is None else input_budget
        self.decode_tokens_per_req = decode_tokens_per_req
        self.draft_slots_per_req = draft_slots_per_req
        self.lag_steps = lag_steps
        self.log_stats = log_stats
        self.hold = False

        # Global state: identical on every rank, updated only from the rows.
        self._cost = _Cost3D()
        self._t_dec: float | None = None
        self._t_dec_init: list[float] = []
        self._t_dec_warm = False
        self._t_dec_at = 0.0
        self._t_dec_rejects = 0
        self._mix_rejects = 0
        self._num_ranks: int | None = None
        self._clock = 0.0
        self._hold_start: float | None = None
        self._rent = 0.0
        self._prev_queued = 0.0
        self._skip_sample = True
        self._reasons: dict[str, int] = {}
        self._logged_at = 0.0

        # Local state.
        self._last_sync: float | None = None
        self._cached_tokens: dict[str, int] = {}
        # `(output, prefill_tokens)` of scheduled steps awaiting their output,
        # and the prefill tokens of the last finished step not yet reported.
        self._pending: deque[tuple[SchedulerOutput, int]] = deque()
        self._done_tokens: int | None = None

    def _clear_hold(self) -> None:
        self.hold = False
        self._hold_start = None
        self._rent = 0.0
        self._prev_queued = 0.0

    def reset(self) -> None:
        """Called when all ranks go idle."""
        self._clear_hold()
        self._skip_sample = True
        self._pending.clear()
        self._done_tokens = None

    def on_lookup(self, request: "Request", num_computed_tokens: int) -> None:
        """Records the prefix hit (local + external) looked up by the scheduler."""
        self._cached_tokens[request.request_id] = num_computed_tokens

    def on_scheduled(
        self, output: "SchedulerOutput", requests: Mapping[str, "Request"]
    ) -> None:
        """Records a scheduled step's prefill tokens and trims cached hits.

        Must run before the scheduler advances `num_computed_tokens`, so that
        `is_prefill_chunk` reflects the pre-step state.
        """
        if output.preempted_req_ids:
            for req_id in output.preempted_req_ids:
                self._cached_tokens.pop(req_id, None)
        if output.total_num_scheduled_tokens == 0:
            return
        num_scheduled = output.num_scheduled_tokens
        tokens = 0
        for new in output.scheduled_new_reqs:
            self._cached_tokens.pop(new.req_id, None)
            tokens += num_scheduled[new.req_id]
        for req_id in output.scheduled_cached_reqs.req_ids:
            if requests[req_id].is_prefill_chunk:
                tokens += num_scheduled[req_id]
        self._pending.append((output, tokens))

    def on_step_finished(self, output: "SchedulerOutput") -> None:
        """Marks a step recorded by `on_scheduled` as finished, to time it."""
        for i, (pending, tokens) in enumerate(self._pending):
            if pending is output:
                for _ in range(i + 1):
                    self._pending.popleft()
                self._done_tokens = tokens
                return

    def _prefill_budget(self, decodes: float) -> int:
        """Token budget left for prefill after `decodes` running decode requests."""
        d = int(decodes)
        tok_left = self.token_budget - d * self.decode_tokens_per_req
        inp_left = (
            self.input_budget
            - d * (self.decode_tokens_per_req + self.draft_slots_per_req)
            - self.draft_slots_per_req
        )
        return max(0, min(tok_left, inp_left))

    def local_row(self) -> _RankRow:
        """This rank's row for the next all-gather."""
        state = self._queue_state()
        now = time.monotonic()
        held = self._held(now)
        period = 0.0 if self._last_sync is None else now - self._last_sync
        self._last_sync = now

        done = 0.0
        done_tokens = 0.0
        if self._done_tokens is not None:
            done = 1.0
            done_tokens = float(self._done_tokens)
            self._done_tokens = None

        chunks = state.prefill_chunks
        if len(self._cached_tokens) > 2 * state.num_waiting + 256:
            self._cached_tokens = {
                r.request_id: self._cached_tokens[r.request_id]
                for r in state.waiting()
                if r.request_id in self._cached_tokens
            }
        decodes = float(max(0, len(state.running) - len(chunks)))
        if not chunks and next(state.waiting(), None) is None:
            return _RankRow(
                done, period, done_tokens, 0.0, 0.0, decodes, 0.0, self.ttft_target
            )

        budget = self._prefill_budget(decodes)
        if budget <= 0:
            return _RankRow(
                done, period, done_tokens, 1.0, 0.0, decodes, 0.0, self.ttft_target
            )

        x, queued, slack = self._scan_queue(state, budget, held)
        return _RankRow(
            done, period, done_tokens, 1.0, float(x), decodes, float(queued), slack
        )

    def _held(self, at: float) -> float:
        """How long the current hold has lasted at monotonic time `at`."""
        if self._hold_start is None:
            return 0.0
        since_sync = 0.0 if self._last_sync is None else at - self._last_sync
        return self._clock - self._hold_start + since_sync

    def _scan_queue(
        self, state: PrefillQueueState, budget: int, held: float
    ) -> tuple[int, int, float]:
        wall = time.time()
        target = self.ttft_target
        cap = state.long_prefill_token_threshold
        x = queued = queue_tokens = 0
        slack = target
        for arrival, remaining, in_progress in self._prefill_queue(state):
            if x < budget:
                step_cap = min(remaining, cap) if cap > 0 else remaining
                x += min(step_cap, budget - x)
            queued += 1
            queue_tokens += remaining
            k = -(-queue_tokens // budget)
            if cap > 0:
                k = max(k, -(-remaining // cap))
            age = wall - arrival
            if not in_progress and age >= target:
                age = min(age, held)
            slack = min(slack, (target - age - (k - 1) * min(held, age)) / k)
            if slack <= 0.0:
                break
        return x, queued, slack

    def _prefill_queue(
        self, state: PrefillQueueState
    ) -> Iterator[tuple[float, int, bool]]:
        """Yield `(arrival_time, uncached_tokens, in_progress)` in admission order."""
        for req in sorted(state.prefill_chunks, key=lambda r: r.arrival_time):
            remaining = req.num_tokens - req.num_computed_tokens
            if remaining > 0:
                yield req.arrival_time, remaining, True
        cached_tokens = self._cached_tokens
        probe = state.num_cached_tokens
        free = state.kv_free_blocks
        slots = min(state.max_active_reqs - len(state.running), _MAX_QUEUE_SCAN)
        for req in state.waiting():
            slots -= 1
            if slots < 0:
                return
            cached = req.num_computed_tokens or cached_tokens.get(req.request_id)
            if cached is None:
                cached = cached_tokens[req.request_id] = probe(req)
            if free is not None:
                free -= state.num_blocks_needed(req, cached)
                if free < 0:
                    return
            # Full prefix hits leave 1 token, which the scheduler never defers.
            if req.num_tokens - cached > 1:
                yield req.arrival_time, req.num_tokens - cached, False

    def update(self, rows: Sequence[Sequence[float]]) -> str:
        """Advance global state with all ranks' rows and set `hold`.

        Returns:
            Reason string (`wait` or `idle` hold; all other reasons release).

        """
        if not rows or any(
            len(r) != ROW_SIZE or any(not math.isfinite(v) for v in r) for r in rows
        ):
            self._clear_hold()
            self._reasons["invalid"] = self._reasons.get("invalid", 0) + 1
            return "invalid"

        num_ranks = len(rows)
        if self._num_ranks is not None and num_ranks != self._num_ranks:
            self._cost.reset()
            self._t_dec = None
            self._t_dec_init.clear()
            self._t_dec_warm = False
            self._t_dec_rejects = 0
            self._mix_rejects = 0
        self._num_ranks = num_ranks

        parsed = [_RankRow(*r) for r in rows]
        period = sum(r.period for r in parsed) / num_ranks
        self._clock += period
        self._learn(parsed, period)

        if self._hold_start is not None:
            self._rent += self.ttft_weight * self._prev_queued * period

        reason = self._decide(parsed)
        self.hold = reason in ("wait", "idle")
        if reason == "wait":
            if self._hold_start is None:
                self._hold_start = self._clock
            self._prev_queued = sum(r.queued for r in parsed)
        else:
            self._hold_start = None
            self._rent = 0.0
            self._prev_queued = 0.0
        if self.log_stats:
            self._log(reason)
        return reason

    def _learn(self, rows: list[_RankRow], period: float) -> None:
        if self._skip_sample:
            self._skip_sample = False
            return
        if period <= 0.0 or not any(r.done for r in rows):
            return
        max_tok = max(r.done_tokens for r in rows)
        if max_tok > 0.0:
            if self._t_dec is None:
                return
            m_ktok = max_tok / 1e3
            s_ktok = sum(r.done_tokens for r in rows) / 1e3
            extra_s = self._cost.predict_ms(m_ktok, s_ktok) / 1e3
            if self._cost.w < 1.0 or period < _STALL_FACTOR * (self._t_dec + extra_s):
                self._mix_rejects = 0
                self._cost.update(m_ktok, s_ktok, (period - self._t_dec) * 1e3)
            else:
                self._mix_rejects += 1
                if self._mix_rejects >= _MAX_STALL_REJECTS:
                    self._mix_rejects = 0
                    self._cost.update(m_ktok, s_ktok, (period - self._t_dec) * 1e3)
            return

        if not self._t_dec_warm:
            self._t_dec_init.append(period)
            ordered = sorted(self._t_dec_init)
            # Index (n - 1) // 2 takes min on sample 2 (drops compile spike)
            # and median on sample 3.
            self._t_dec = ordered[(len(ordered) - 1) // 2]
            self._t_dec_at = self._clock
            if len(self._t_dec_init) >= _DEC_INIT_SAMPLES:
                self._t_dec_warm = True
                self._t_dec_init.clear()
        elif self._t_dec is not None and period < _STALL_FACTOR * self._t_dec:
            self._t_dec_rejects = 0
            w = math.exp(-(self._clock - self._t_dec_at) / _DEC_TAU_S)
            self._t_dec = w * self._t_dec + (1.0 - w) * period
            self._t_dec_at = self._clock
        else:
            self._t_dec_rejects += 1
            if self._t_dec_rejects >= _MAX_STALL_REJECTS:
                self._t_dec = period
                self._t_dec_at = self._clock
                self._t_dec_rejects = 0

    def _decide(self, rows: list[_RankRow]) -> str:
        decodes = sum(r.decodes for r in rows)
        if decodes == 0.0:
            return "no_decode"
        if self._t_dec is None or self._cost.w < 1.0:
            return "cold"
        # Mirror scheduler.py's per-rank no-decode exemption: if any rank will
        # run a prefill anyway, release all ranks to piggyback on the step.
        if any(r.decodes == 0.0 and r.x > 0.0 for r in rows):
            return "no_local_decode"

        avail = [r for r in rows if not (r.work > 0.0 and r.x == 0.0)]
        n_avail = len(avail)
        if n_avail == 0:
            return "blocked"

        t_dec = self._t_dec
        c = self._cost
        if not any(r.work for r in rows):
            m_typ = c.a_mat[0][1] / c.w if c.w > 0.0 else 0.0
            amort_ms = (
                (c.f + c.a * m_typ) * (1.0 - 1.0 / n_avail) if n_avail > 1 else c.f
            )
            if 0.5 * decodes * (amort_ms / 1e3) > self.ttft_weight * t_dec:
                return "idle"
            return "nothing"

        if sum(r.x for r in rows) == 0.0:
            return "blocked"

        held = 0.0 if self._hold_start is None else self._clock - self._hold_start
        slack = min(r.slack for r in rows)
        m_ktok = max(r.x for r in rows) / 1e3
        s_ktok = sum(r.x for r in rows) / 1e3
        t_mix = t_dec + c.predict_ms(m_ktok, s_ktok) / 1e3
        lag = self.lag_steps * t_dec
        h = min(self.ttft_target - held, slack - t_mix) - lag
        if h <= 0.0:
            return "deadline"

        avail_budget_ktok = sum(self._prefill_budget(r.decodes) for r in avail) / 1e3
        f_waste = (
            c.f * max(0.0, 1.0 - s_ktok / avail_budget_ktok)
            if avail_budget_ktok > 0.0
            else 0.0
        )
        a_waste = c.a * max(0.0, m_ktok - s_ktok / n_avail)
        # Factor 0.5 aligns default ttft_weight=1.0 with the E2E optimum.
        buy_cost = 0.5 * decodes * (f_waste + a_waste) / 1e3
        wait = min(t_dec, h)
        queued = sum(r.queued for r in rows)
        if self._rent + self.ttft_weight * queued * wait < buy_cost:
            return "wait"
        return "cost"

    def _log(self, reason: str) -> None:
        self._reasons[reason] = self._reasons.get(reason, 0) + 1
        if self._clock - self._logged_at < _LOG_INTERVAL_S:
            return
        self._logged_at = self._clock
        c = self._cost
        logger.info(
            "Prefill hold: %s | t_dec=%.1fms F=%.1fms a=%.1fms/ktok "
            "b=%.1fms/ktok w=%.0f",
            " ".join(f"{k}={v}" for k, v in sorted(self._reasons.items())),
            (self._t_dec or 0.0) * 1e3,
            c.f,
            c.a,
            c.b,
            c.w,
        )
        self._reasons = {}
