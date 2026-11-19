# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost-based prefill hold across data parallel ranks.

With DP attention + EP, all ranks step in lockstep, so a prefill on one rank
makes the step mixed for every rank. Each step, every rank reports a row of
local state (`PrefillHold.local_row`). The rows are all-gathered by the DP
engine core's state sync, and every rank runs the same deterministic
`PrefillHold.update` on them, so all ranks hold or release together.

- Hold while waiting pays: the decode time saved by doing the queued prefill
  at the per-token cost predicted after one more decode step must exceed
  `alpha` times the TTFT the wait adds, even at the 97.5% lower confidence
  bound.
- Bound the hold per request: a request of age `a` with `k` releases still
  ahead of it may be held `(T - a - (k - 1) * e) / k - t_mix` more, where `e`
  is the time held so far, `t_mix` is the mixed step time and `T` is the TTFT
  target. A request older than `T` releases every step.
- Step costs come from an online least-squares fit of measured step times,
  with no latency model. Until the fit is pinned down, it releases.
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

# Layout of the row each rank reports per step.
ROW_SIZE = 14
(
    _DONE,  # a step with scheduled tokens finished since the previous sync
    _PERIOD,  # seconds since the previous sync
    _DONE_TOKENS,  # prefill tokens of that step
    _DONE_ATTN,  # attention work of that step, sum(n * (ctx + n / 2))
    _WORK,  # has prefill work
    _BLOCKED,  # has prefill work but no free slot to start it
    _X,  # prefill tokens it would run if released now, at most one budget
    _X_ATTN,  # attention work of those
    _DECODES,  # running decode requests
    _QUEUED,  # requests with prefill work, 0 if blocked
    _SLACK,  # min over them of (T - age - (k - 1) * e) / k
    _ARRIVED,  # requests that arrived since the previous sync
    _ARRIVED_UNCACHED,  # their prompt tokens not in the prefix cache
    _ARRIVED_CACHED,  # their prompt tokens in the prefix cache
) = range(ROW_SIZE)

# Time constant of the arrival rate and request size estimates.
_RATE_TAU_S = 30.0
# Time constant of the decode-only step time estimate.
_DEC_TAU_S = 1.0
# The cost fit forgets over about this many mixed steps.
_FIT_WINDOW = 200.0
_RIDGE = 1e-6
_LOG_INTERVAL_S = 60.0

# Two-sided 95% (one-sided 97.5%) Student t quantiles by degrees of freedom.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
         30: 2.042}  # fmt: skip


def _t975(dof: float) -> float:
    if dof > 30:
        return 1.96
    return _T975[max(k for k in _T975 if k <= dof)]


def _inverse(m: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse of a small nonsingular matrix."""
    n = len(m)
    a = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(m)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(a[i][k]))
        a[k], a[p] = a[p], a[k]
        pivot = a[k][k]
        a[k] = [v / pivot for v in a[k]]
        for i in range(n):
            f = a[i][k]
            if i != k and f != 0.0:
                a[i] = [v - f * u for v, u in zip(a[i], a[k])]
    return [row[n:] for row in a]


class _StepCostFit:
    """Online fit of a mixed step's extra time over a decode-only step.

    The model is `F + a * M + b * S + c * C` ms, where `M` and `S` are the
    max and the sum over ranks of prefill Ktok, and `C` is the max over ranks
    of prefill attention work in Gtok^2. Weighted least squares, each step's
    weight decaying by `1 - 1 / _FIT_WINDOW` per later mixed step. There is no
    prior: a tiny ridge only keeps the solve finite along combinations the
    steps never varied, and `sd` reports how poorly a prediction is pinned
    there. A residual beyond 3 times the running mean absolute residual is
    down-weighted to that bound.
    """

    def __init__(self) -> None:
        self.a = [[0.0] * 4 for _ in range(4)]
        self.b = [0.0] * 4
        self.yy = 0.0
        self.w = 0.0
        self.theta = [0.0] * 4
        self.p: list[list[float]] | None = None
        self.var: float | None = None
        self.sig: float | None = None

    @staticmethod
    def features(tokens: list[float], attn: list[float]) -> list[float]:
        return [1.0, max(tokens) / 1e3, sum(tokens) / 1e3, max(attn) / 1e9]

    @property
    def dof(self) -> float:
        return self.w - 4

    def predict_ms(self, phi: list[float]) -> float:
        return sum(t * f for t, f in zip(self.theta, phi))

    def sd_ms(self, d: list[float]) -> float:
        """Standard error of `theta . d`; inf until there is a residual dof."""
        if self.var is None or self.p is None:
            return math.inf
        q = sum(
            di * sum(pij * dj for pij, dj in zip(pi, d)) for di, pi in zip(d, self.p)
        )
        return math.sqrt(self.var * max(q, 0.0))

    def update(self, phi: list[float], y_ms: float) -> None:
        err = abs(y_ms - self.predict_ms(phi))
        wt = 3 * self.sig / err if self.sig is not None and err > 3 * self.sig else 1.0
        self.sig = err if self.sig is None else 0.95 * self.sig + 0.05 * err
        keep = 1.0 - 1.0 / _FIT_WINDOW
        self.w = keep * self.w + wt
        self.yy = keep * self.yy + wt * y_ms * y_ms
        for i in range(4):
            self.b[i] = keep * self.b[i] + wt * phi[i] * y_ms
            for j in range(4):
                self.a[i][j] = keep * self.a[i][j] + wt * phi[i] * phi[j]
        self.p = _inverse(
            [
                [
                    aij * (1.0 + _RIDGE / self.w) if i == j else aij
                    for j, aij in enumerate(row)
                ]
                for i, row in enumerate(self.a)
            ]
        )
        self.theta = [sum(pij * bj for pij, bj in zip(pi, self.b)) for pi in self.p]
        if self.dof >= 1:
            a_theta = [
                sum(aij * tj for aij, tj in zip(ai, self.theta)) for ai in self.a
            ]
            rss = self.yy + sum(
                t * (at - 2 * bi) for t, at, bi in zip(self.theta, a_theta, self.b)
            )
            self.var = max(rss, 0.0) / self.dof


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
    # Async KV loads finished but not yet admitted: one finishing lets a
    # waiting request start without changing `num_waiting`.
    num_loaded: int
    max_active_reqs: int
    num_cached_tokens: Callable[["Request"], int]

    @property
    def scan_key(self) -> tuple[int, int, int]:
        """Changes when the queue may scan differently, other than by time."""
        return len(self.prefill_chunks), self.num_waiting, self.num_loaded


class _QueueScan(NamedTuple):
    """`local_row`'s fields read off the prefill queue at monotonic time `at`,
    when its `PrefillQueueState.scan_key` was `key`."""

    at: float
    key: tuple[int, int, int]
    x: int
    x_attn: float
    queued: int
    slack: float


class PrefillHold:
    """Decides, once per step and identically on every DP rank, whether all
    ranks hold their prefills on the next step. The DP engine core feeds every
    rank's `local_row` to `update` each step, through the scheduler; without
    that, it never holds.

    Args:
        queue_state: Returns a read-only view of the scheduler's queues.
        ttft_target_s: TTFT target `T` in seconds.
        alpha: Weight of one second of TTFT per waiting request against one
            second of decode stall per running request.
        token_budget: Max tokens per step per rank.
        lag_steps: Steps between a decision and the step it takes effect on
            the GPU beyond the next one (`max_concurrent_batches - 1`, so 1
            with async scheduling and no pipeline parallelism).
        log_stats: Whether this rank logs a periodic summary; DP rank 0 does.

    """

    def __init__(
        self,
        queue_state: Callable[[], PrefillQueueState],
        ttft_target_s: float,
        alpha: float,
        token_budget: int,
        lag_steps: int = 0,
        log_stats: bool = False,
    ) -> None:
        self._queue_state = queue_state
        self.ttft_target = ttft_target_s
        self.alpha = alpha
        self.budget = token_budget
        self.lag_steps = lag_steps
        self.log_stats = log_stats
        # Whether to hold prefills on the next step.
        self.hold = False

        # Global state: identical on every rank, updated only from the rows.
        self._fit = _StepCostFit()
        self._t_dec: float | None = None
        self._t_dec_at = 0.0
        self._clock = 0.0
        self._rate = 0.0
        self._w_reqs = 0.0
        self._w_uncached = 0.0
        self._w_cached = 0.0
        self._hold_start: float | None = None
        self._skip_sample = True
        self._reasons: dict[str, int] = {}
        self._logged_at = 0.0

        # Local state.
        self._last_sync: float | None = None
        self._arrived = [0, 0, 0]
        self._cached_tokens: dict[str, int] = {}
        # (output, prefill tokens, attention work) of scheduled steps awaiting
        # their output, and those of the last finished step not yet reported.
        self._pending: deque[tuple[SchedulerOutput, int, float]] = deque()
        self._done: tuple[int, float] | None = None
        self._scan: _QueueScan | None = None

    def reset(self) -> None:
        """Called when all ranks go idle; the rate estimates decay over it."""
        self.hold = False
        self._hold_start = None
        self._skip_sample = True
        self._pending.clear()
        self._done = None

    def on_arrival(
        self, request: "Request", num_cached_tokens: Callable[["Request"], int]
    ) -> None:
        cached_tokens = num_cached_tokens(request)
        self._cached_tokens[request.request_id] = cached_tokens
        self._arrived[0] += 1
        self._arrived[1] += request.num_tokens - cached_tokens
        self._arrived[2] += cached_tokens

    def on_scheduled(
        self, output: "SchedulerOutput", requests: Mapping[str, "Request"]
    ) -> None:
        """Records a scheduled step's prefill tokens and attention work, the
        sum over prefill chunks of `n * (ctx + n / 2)` for `n` new tokens after
        `ctx` computed ones. Must run before the scheduler advances
        `num_computed_tokens`, so that `is_prefill_chunk` is pre-step."""
        if output.total_num_scheduled_tokens == 0:
            return
        num_scheduled = output.num_scheduled_tokens
        tokens, attn = 0, 0.0
        # Every request admitted from the waiting queue starts a prefill.
        chunks = [(r.req_id, r.num_computed_tokens) for r in output.scheduled_new_reqs]
        cached = output.scheduled_cached_reqs
        chunks += [
            (req_id, ctx)
            for req_id, ctx in zip(cached.req_ids, cached.num_computed_tokens)
            if requests[req_id].is_prefill_chunk
        ]
        for req_id, ctx in chunks:
            n = num_scheduled[req_id]
            tokens += n
            attn += n * (ctx + n / 2)
        self._pending.append((output, tokens, attn))

    def on_step_finished(self, output: "SchedulerOutput") -> None:
        """Marks a step recorded by `on_scheduled` as finished, to time it."""
        for i, (pending, tokens, attn) in enumerate(self._pending):
            if pending is output:
                for _ in range(i + 1):
                    self._pending.popleft()
                self._done = (tokens, attn)
                return

    def prepare_row(self) -> None:
        """Scans the prefill queue ahead of the next `local_row`. The engine
        core calls this while the GPU runs a step, off the critical path."""
        state = self._queue_state()
        at = time.monotonic()
        self._scan = (
            self._scan_queue(state, at, self._held(at))
            if state.prefill_chunks or state.num_waiting
            else None
        )

    def local_row(self) -> list[float]:
        """This rank's row for the next all-gather."""
        state = self._queue_state()
        now = time.monotonic()
        held = self._held(now)
        period = 0.0 if self._last_sync is None else now - self._last_sync
        self._last_sync = now
        scan, self._scan = self._scan, None
        row = [0.0] * ROW_SIZE
        row[_PERIOD] = period
        row[_SLACK] = self.ttft_target
        done = self._done
        if done is not None:
            self._done = None
            row[_DONE] = 1.0
            row[_DONE_TOKENS], row[_DONE_ATTN] = done
        row[_ARRIVED], row[_ARRIVED_UNCACHED], row[_ARRIVED_CACHED] = self._arrived
        self._arrived = [0, 0, 0]

        chunks = state.prefill_chunks
        if len(self._cached_tokens) > 2 * state.num_waiting + 256:
            self._cached_tokens = {
                r.request_id: self._cached_tokens[r.request_id]
                for r in state.waiting()
                if r.request_id in self._cached_tokens
            }
        row[_DECODES] = max(0, len(state.running) - len(chunks))
        if not chunks and next(state.waiting(), None) is None:
            return row
        row[_WORK] = 1.0
        if not chunks and len(state.running) >= state.max_active_reqs:
            row[_BLOCKED] = 1.0
            return row

        # Aborts, the finished step and finished KV loads may have changed the
        # queue since the prepared scan.
        if scan is None or scan.key != state.scan_key:
            scan = self._scan_queue(state, now, held)
        row[_X] = scan.x
        row[_X_ATTN] = scan.x_attn
        row[_QUEUED] = scan.queued
        # Every allowance shrinks at rate 1 while holding and at most 1 if not,
        # so this is exact while holding and a lower bound otherwise.
        row[_SLACK] = scan.slack - (now - scan.at)
        return row

    def _held(self, at: float) -> float:
        """How long the current hold has lasted at monotonic time `at`."""
        if self._hold_start is None:
            return 0.0
        since_sync = 0.0 if self._last_sync is None else at - self._last_sync
        return self._clock - self._hold_start + since_sync

    def _scan_queue(
        self, state: PrefillQueueState, at: float, held: float
    ) -> _QueueScan:
        wall = time.time()
        budget = self.budget
        target = self.ttft_target
        x = queued = queue_tokens = 0
        x_attn = 0.0
        slack = target
        for arrival, remaining, ctx in self._prefill_queue(state):
            if x < budget:
                n = min(remaining, budget - x)
                x += n
                x_attn += n * (ctx + n / 2)
            queued += 1
            queue_tokens += remaining
            k = -(-queue_tokens // budget)
            age = wall - arrival
            slack = min(slack, (target - age - (k - 1) * min(held, age)) / k)
            if slack <= 0:
                # Releases whatever the rest of the queue is.
                break
        return _QueueScan(at, state.scan_key, x, x_attn, queued, slack)

    def _prefill_queue(
        self, state: PrefillQueueState
    ) -> Iterator[tuple[float, int, int]]:
        """(arrival time, uncached tokens left, cached or computed tokens) of
        each request with prefill work, in-progress prefills first."""
        for req in sorted(state.prefill_chunks, key=lambda r: r.arrival_time):
            remaining = req.num_tokens - req.num_computed_tokens
            if remaining > 0:
                yield req.arrival_time, remaining, req.num_computed_tokens
        cached_tokens = self._cached_tokens
        probe = state.num_cached_tokens
        for req in state.waiting():
            # A request holding KV blocks, such as a finished async KV load,
            # resumes from them.
            cached = req.num_computed_tokens or cached_tokens.get(req.request_id)
            if cached is None:
                cached = cached_tokens[req.request_id] = probe(req)
            if req.num_tokens > cached:
                yield req.arrival_time, req.num_tokens - cached, cached

    def update(self, rows: list[list[float]]) -> str:
        """Advances the global state with all ranks' rows and sets `hold`.

        Returns:
            Why it holds (`wait`, or `idle` with no prefill work yet) or
            releases.

        """
        num_ranks = len(rows)
        period = sum(r[_PERIOD] for r in rows) / num_ranks
        self._clock += period
        self._learn(rows, period)

        decay = math.exp(-period / _RATE_TAU_S)
        arrived = sum(r[_ARRIVED] for r in rows)
        self._rate = self._rate * decay + arrived / _RATE_TAU_S
        self._w_reqs = self._w_reqs * decay + arrived
        self._w_uncached = self._w_uncached * decay + sum(
            r[_ARRIVED_UNCACHED] for r in rows
        )
        self._w_cached = self._w_cached * decay + sum(r[_ARRIVED_CACHED] for r in rows)

        reason = self._decide(rows)
        self.hold = reason in ("wait", "idle")
        if reason == "wait":
            if self._hold_start is None:
                self._hold_start = self._clock
        else:
            self._hold_start = None
        if self.log_stats:
            self._log(reason)
        return reason

    def _learn(self, rows: list[list[float]], period: float) -> None:
        if self._skip_sample:
            self._skip_sample = False
            return
        if not any(r[_DONE] for r in rows):
            return
        tokens = [r[_DONE_TOKENS] for r in rows]
        if any(tokens):
            if self._t_dec is not None:
                phi = self._fit.features(tokens, [r[_DONE_ATTN] for r in rows])
                self._fit.update(phi, (period - self._t_dec) * 1e3)
        elif self._t_dec is None:
            self._t_dec = period
            self._t_dec_at = self._clock
        else:
            w = math.exp(-(self._clock - self._t_dec_at) / _DEC_TAU_S)
            self._t_dec = w * self._t_dec + (1.0 - w) * period
            self._t_dec_at = self._clock

    def _decide(self, rows: list[list[float]]) -> str:
        """`wait` or `idle` to hold, otherwise why to release."""
        decodes = sum(r[_DECODES] for r in rows)
        cold = self._t_dec is None or self._fit.dof < 1
        if decodes == 0:
            return "no_decode"
        if cold:
            return "cold"
        blocked = [bool(r[_BLOCKED]) for r in rows]
        if not any(r[_WORK] for r in rows):
            # The next step may admit what arrives before it: hold if a typical
            # request alone on one rank would be held.
            if self._w_reqs <= 0:
                return "nothing"
            size = min(self._w_uncached / self._w_reqs, self.budget)
            ctx = self._w_cached / self._w_reqs
            x = [size] + [0.0] * (len(rows) - 1)
            attn = [size * (ctx + size / 2)] + [0.0] * (len(rows) - 1)
            reason = self._judge(x, attn, blocked, decodes, 1, 0.0, self.ttft_target)
            return "idle" if reason == "wait" else "nothing"
        x = [r[_X] for r in rows]
        if sum(x) == 0:
            return "blocked"
        held = 0.0 if self._hold_start is None else self._clock - self._hold_start
        return self._judge(
            x,
            [r[_X_ATTN] for r in rows],
            blocked,
            decodes,
            sum(r[_QUEUED] for r in rows),
            held,
            min(r[_SLACK] for r in rows),
        )

    def _judge(
        self,
        x: list[float],
        attn: list[float],
        blocked: list[bool],
        decodes: float,
        queued: float,
        held: float,
        slack: float,
    ) -> str:
        fit = self._fit
        t_dec = self._t_dec
        assert t_dec is not None
        z = _t975(fit.dof)
        phi = fit.features(x, attn)
        extra = max(0.0, fit.predict_ms(phi)) / 1e3
        t_mix = t_dec + extra
        lag = self.lag_steps * t_dec
        # How much longer this hold may last, at the point estimate (h0) and
        # budgeting the upper bound of t_mix (h).
        h0 = min(self.ttft_target - held, slack - t_mix) - lag
        h = min(h0, slack - t_mix - z * fit.sd_ms(phi) / 1e3 - lag)
        if h <= 0:
            return "deadline" if h0 <= 0 else "unsure"

        # Expected prefill after one more decode step, assuming arrivals spread
        # evenly over the ranks.
        wait = min(t_dec, h)
        rate = self._rate / len(x)
        size = self._w_uncached / self._w_reqs if self._w_reqs > 0 else 0.0
        ctx = self._w_cached / self._w_reqs if self._w_reqs > 0 else 0.0
        budget = self.budget
        p_new = 1.0 - math.exp(-rate * wait)
        add = [
            0.0
            if b
            else p_new * min(size, budget)
            if xr == 0
            else min(budget - xr, rate * wait * size)
            for xr, b in zip(x, blocked)
        ]
        sx, sa = sum(x), sum(add)
        if sa <= 0:
            return "cost"
        phi_hat = fit.features(
            [xr + a for xr, a in zip(x, add)],
            [i + a * (ctx + a / 2) for i, a in zip(attn, add)],
        )
        c_hat = max(0.0, fit.predict_ms(phi_hat)) / 1e3 / (sx + sa)
        gain = decodes * sx * (extra / sx - c_hat)
        cost = self.alpha * queued * wait
        if gain <= cost:
            return "cost"
        r = sx / (sx + sa)
        d = [p - r * q for p, q in zip(phi, phi_hat)]
        if gain - decodes * z * fit.sd_ms(d) / 1e3 > cost:
            return "wait"
        return "unsure"

    def _log(self, reason: str) -> None:
        self._reasons[reason] = self._reasons.get(reason, 0) + 1
        if self._clock - self._logged_at < _LOG_INTERVAL_S:
            return
        self._logged_at = self._clock
        total = sum(self._reasons.values())
        held = self._reasons.get("wait", 0) + self._reasons.get("idle", 0)
        f, a, b, c = self._fit.theta
        logger.info(
            "DP prefill hold: held %.1f%% of %d steps, decisions %s; "
            "t_dec %.1f ms, fit F=%.1f ms a=%.2f b=%.2f ms/Ktok c=%.2f ms/Gtok^2 "
            "(dof %.0f), rate %.2f req/s",
            100.0 * held / total,
            total,
            self._reasons,
            1e3 * (self._t_dec or 0.0),
            f,
            a,
            b,
            c,
            self._fit.dof,
            self._rate,
        )
        self._reasons = {}
