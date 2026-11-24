# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time-to-first-token forecast for a request that would arrive now.

The engine measures how long its own steps take and replays its scheduling
rule over the requests it already holds. The result is a curve: seconds to
first token as a function of the number of prompt tokens the new request needs
computed. A router evaluates the curve with its own knowledge of how much of a
prompt each engine has cached, so no per-request call to the engine is needed.
"""

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Any

import numpy as np

from vllm import envs

# Features of one step: (1, tokens, sum of context, sum of tokens * context).
Features = tuple[float, float, float, float]


def curve_value(segments: list[Any], new_tokens: int) -> float | None:
    """Seconds on a curve of `[u0, t0, u1, t1]` segments; None past its end."""
    for u0, t0, u1, t1 in segments:
        if new_tokens <= u1:
            return t0 + (t1 - t0) * max(new_tokens - u0, 0) / max(u1 - u0, 1)
    return None


class IngestModel:
    """Seconds between the server receiving a request and the scheduler's
    input thread having it, as a line over prompt tokens.

    Fitted over recent arrivals with exponentially fading weights.
    """

    def __init__(self, keep: float = 0.98) -> None:
        self.keep = keep
        self.n = self.sx = self.sy = self.sxx = self.sxy = 0.0
        self.base = 0.0
        self.per_token = 0.0

    def observe(self, prompt_tokens: int, seconds: float) -> None:
        k = self.keep
        self.n = self.n * k + 1.0
        self.sx = self.sx * k + prompt_tokens
        self.sy = self.sy * k + seconds
        self.sxx = self.sxx * k + prompt_tokens * prompt_tokens
        self.sxy = self.sxy * k + prompt_tokens * seconds
        mean_x, mean_y = self.sx / self.n, self.sy / self.n
        var = self.sxx / self.n - mean_x * mean_x
        cov = self.sxy / self.n - mean_x * mean_y
        # Without a spread of prompt sizes the mean is all there is.
        slope = cov / var if self.n >= 8 and var > 1e6 else 0.0
        self.per_token = max(slope, 0.0)
        self.base = max(mean_y - self.per_token * mean_x, 0.0)
        if self.per_token == 0.0:
            self.base = mean_y

    def predict(self, prompt_tokens: int) -> float:
        return self.base + self.per_token * prompt_tokens


class StepTimeModel:
    """Least-squares fit of step wall time over recent steps.

    Samples are kept per step size (log2 of scheduled tokens), so a long run
    of small steps does not erase what large steps cost. After the first
    `warmup` samples the fit is redone at most every `refit_seconds`, so an
    engine running short steps does not pay for it on each of them.
    """

    def __init__(
        self,
        per_bucket: int = 64,
        refit_every: int = 8,
        refit_seconds: float = 0.5,
        warmup: int = 256,
    ) -> None:
        self._buckets: dict[int, deque[tuple[Features, float]]] = {}
        self._per_bucket = per_bucket
        self._refit_every = refit_every
        self._refit_seconds = refit_seconds
        self._warmup = warmup
        self._fit_time = 0.0
        self._since_fit = 0
        self.num_samples = 0
        self.coef: tuple[float, float, float, float] | None = None
        self.floor = 0.0

    @staticmethod
    def features(tokens: int, ctx_sum: float, tok_ctx_sum: float) -> Features:
        return (1.0, float(tokens), float(ctx_sum), float(tok_ctx_sum))

    def observe(self, x: Features, seconds: float) -> None:
        bucket = int(x[1]).bit_length()
        samples = self._buckets.get(bucket)
        if samples is None:
            samples = self._buckets[bucket] = deque(maxlen=self._per_bucket)
        samples.append((x, seconds))
        self.num_samples += 1
        self._since_fit += 1
        if self.coef is None or (
            self._since_fit >= self._refit_every
            and (
                self.num_samples <= self._warmup
                or time.monotonic() - self._fit_time >= self._refit_seconds
            )
        ):
            self._fit()

    def _fit(self) -> None:
        self._since_fit = 0
        self._fit_time = time.monotonic()
        rows = [s for samples in self._buckets.values() for s in samples]
        if len(rows) < 4:
            return
        x = np.array([r[0] for r in rows], dtype=np.float64)
        y = np.array([r[1] for r in rows], dtype=np.float64)
        scale = np.maximum(np.abs(x).max(axis=0), 1e-9)
        xs = x / scale
        keep = np.ones(len(y), dtype=bool)
        coef = np.zeros(4)
        for _ in range(2):
            active = [0, 1, 2, 3]
            coef = np.zeros(4)
            # Drop a feature whose coefficient comes out negative and refit.
            while active:
                sol, *_ = np.linalg.lstsq(xs[keep][:, active], y[keep], rcond=None)
                if (sol >= 0).all():
                    coef[active] = sol
                    break
                active.pop(int(np.argmin(sol)))
            # One pass of outlier trimming (pauses, first-use compilation).
            resid = np.abs(xs @ coef - y)
            cut = 4.0 * np.median(resid) + 1e-4
            new_keep = resid <= cut
            if new_keep.all() or new_keep.sum() < 4:
                break
            keep = new_keep
        self.coef = tuple(float(c) for c in coef / scale)  # type: ignore[assignment]
        self.floor = 0.5 * float(y[keep].min())

    def predict(self, x: Features) -> float:
        c = self.coef
        if c is None:
            return 0.0
        return max(c[0] + c[1] * x[1] + c[2] * x[2] + c[3] * x[3], self.floor)


class PaceModel:
    """How much slower than the step-time fit the last few steps ran.

    Data-parallel ranks step together, so a rank's steps slow down with its
    peers' load, which its own step sizes do not show. The fit follows over a
    minute of steps, the pace within a few.
    """

    def __init__(self, window: float = 10.0) -> None:
        self.window = window
        self._recent: deque[tuple[float, float]] = deque(maxlen=5)
        self._value = (1.0, 0.0)

    def observe(self, ratio: float, now: float) -> None:
        """Record one step's measured over fitted seconds."""
        self._recent.append((now, min(max(ratio, 0.25), 8.0)))
        recent = sorted(r for t, r in self._recent if now - t <= self.window)
        # One slow step, such as a pause, does not move the forecast.
        self._value = (recent[len(recent) // 2] if len(recent) >= 3 else 1.0, now)

    def value(self, now: float) -> float:
        """The median ratio; 1 without steps in the last `window` seconds."""
        pace, at = self._value
        return pace if now - at <= self.window else 1.0


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Scheduler state the forecast replays from. Immutable; swapped whole."""

    time: float
    # When the oldest unfinished step started executing.
    exec_started: float
    # Unfinished steps, oldest first: (features, seats freed when it returns).
    inflight: tuple[tuple[Features, int], ...] = ()
    # Requests still computing their prompt, in scheduling order:
    # (remaining tokens, context tokens, holds a seat, decode steps after).
    prefills: tuple[tuple[int, int, bool, int], ...] = ()
    # Running requests past their prompt: (context tokens, steps left).
    decodes: tuple[tuple[int, int], ...] = ()
    # Sequence number of the last arrival the scheduler has taken in.
    seen_seq: int = 0
    # Expected seconds of the step without requests (data-parallel sync)
    # executing since `exec_started`; 0 when there is none.
    idle_step: float = 0.0


@dataclass(slots=True)
class _InFlight:
    features: Features
    scheduled_at: float
    req_ids: tuple[str, ...]


@dataclass(slots=True)
class _SimReq:
    remaining: int
    context: int
    seated: bool
    decode_after: int
    probe: bool = False


@dataclass(slots=True)
class _SimBatch:
    end: float = 0.0
    start: float = 0.0
    frees: int = 0
    done: list[_SimReq] = field(default_factory=list)
    # Probe bookkeeping: tokens of the probe in this step and its totals after.
    probe_tokens: int = 0
    probe_total: int = 0
    probe_context: int = 0
    tokens: int = 0
    ctx_sum: float = 0.0
    tok_ctx_sum: float = 0.0
    step: int = 0


class Forecaster:
    """Tracks step timing and scheduler state, and replays them forward."""

    def __init__(
        self,
        token_budget: int,
        max_seats: int,
        long_prefill_threshold: int,
        adaptive_threshold: bool,
        max_model_len: int,
        queue_size: int = 1,
    ) -> None:
        self.token_budget = token_budget
        self.max_seats = max_seats
        self.long_prefill_threshold = long_prefill_threshold
        self.adaptive_threshold = adaptive_threshold
        self.max_model_len = max_model_len
        self.queue_size = queue_size
        # A probe not served within this many steps has no forecast.
        self.max_replay_steps = 4096
        self.model = StepTimeModel()
        self._inflight: dict[int, _InFlight] = {}
        self._last_output = 0.0
        # The state every forecast is replayed from. The scheduler hooks swap
        # it whole; readers on other threads take one reference.
        self.snapshot = Snapshot(time=time.monotonic(), exec_started=0.0)
        # Steps the engine runs without requests to stay in step with its
        # data-parallel peers: smoothed duration, start of the current one.
        self.idle_step_seconds = 0.0
        self._idle_started = 0.0
        self._idle_steps = 0
        self.pace_model = PaceModel()
        # Time a request spends waiting for a KV connector to load its prefix
        # (a restore from another tier) before it can be scheduled, smoothed.
        self.restore_seconds = 0.0
        self._restores = 0
        # Time a request spends in the server before the scheduler's input
        # thread has it. The forecast counts from the server's arrival stamp.
        self.ingest = IngestModel()
        # Per-arrival records for diagnosis, off unless asked for.
        self._detail: deque[dict[str, float]] | None = (
            deque(maxlen=256) if envs.VLLM_TTFT_FORECAST_DEBUG else None
        )
        self._dbg: dict[str, tuple[float, dict[str, float]]] = {}
        self._seen_seq = 0
        # Arrivals the scheduler has not taken in yet:
        # (seq, remaining, context, decode steps after). Owned by the engine's
        # input thread.
        self._pending: deque[tuple[int, int, int, int]] = deque()
        self._next_seq = 0
        # Share of time the engine is busy and tokens per step, both smoothed
        # over about `load_window` seconds. They stand in for requests that
        # arrive after the forecast is made.
        self.load_window = 30.0
        self.busy_share = 0.0
        self.tokens_per_step = 0.0
        # Own forecasts awaiting their outcome, and recent (forecast,
        # expected-arrivals forecast, actual) results.
        self._awaiting: dict[str, tuple[float, float, float]] = {}
        self._scored: deque[tuple[float, float, float]] = deque(maxlen=512)
        # Idle-engine curves by cached-prefix length, refreshed with the fit.
        self._floors: dict[int, tuple[Any, list[tuple[int, float, int, float]]]] = {}
        # Waiting requests hold no blocks, so their cached prefix can be
        # evicted before they run. Each step re-reads it for as many of them
        # as fit in `refresh_budget` seconds, continuing where the last step
        # stopped.
        self.refresh_budget = 100e-6
        self._refresh_cursor = 0
        self._refresh_changes = 0
        # Cost of the forecast itself, reported by describe().
        self._hook_seconds = 0.0
        self._hook_calls = 0
        self._replay_seconds = 0.0
        self._replay_calls = 0

    # ---- scheduler side (engine loop thread) ----

    def note_added(self, request: Any) -> None:
        seq = getattr(request, "forecast_seq", 0)
        if seq > self._seen_seq:
            self._seen_seq = seq
        made = getattr(request, "forecast_made", None)
        if made is not None:
            self._awaiting[request.request_id] = made
            dbg = getattr(request, "forecast_dbg", None)
            if dbg is not None:
                dbg["added"] = time.monotonic() - made[0]
                self._dbg[request.request_id] = (made[0], dbg)

    def on_schedule(self, scheduler: Any, scheduler_output: Any) -> None:
        now = time.monotonic()
        tokens = 0
        ctx_sum = 0.0
        tok_ctx_sum = 0.0
        for req_id, n in scheduler_output.num_scheduled_tokens.items():
            request = scheduler.requests.get(req_id)
            if request is None or n <= 0:
                continue
            if (entry := self._dbg.get(req_id)) is not None:
                entry[1].setdefault("sched", now - entry[0])
                entry[1]["steps"] = entry[1].get("steps", 0) + 1
            # num_computed_tokens already includes this step's tokens.
            before = request.num_computed_tokens - n
            tokens += n
            ctx_sum += before
            tok_ctx_sum += n * (before + n / 2)
        self._inflight[id(scheduler_output)] = _InFlight(
            StepTimeModel.features(tokens, ctx_sum, tok_ctx_sum),
            now,
            tuple(scheduler_output.num_scheduled_tokens),
        )
        self._publish(scheduler, now, refresh=True)
        self._hook_seconds += time.monotonic() - now
        self._hook_calls += 1

    def on_output(self, scheduler: Any, scheduler_output: Any) -> None:
        now = time.monotonic()
        step = self._inflight.pop(id(scheduler_output), None)
        if step is not None and step.features[1] > 0:
            started = max(step.scheduled_at, self._last_output)
            predicted = self.model.predict(step.features)
            if predicted > 0.0:
                self.pace_model.observe((now - started) / predicted, now)
            self.model.observe(step.features, now - started)
            gap = now - self._last_output if self._last_output else now - started
            keep = math.exp(-max(gap, 1e-6) / self.load_window)
            busy = min((now - started) / max(gap, 1e-6), 1.0)
            self.busy_share = self.busy_share * keep + busy * (1 - keep)
            self.tokens_per_step += 0.05 * (step.features[1] - self.tokens_per_step)
        if step is not None and self._awaiting:
            for req_id in step.req_ids:
                made = self._awaiting.get(req_id)
                if made is None:
                    continue
                request = scheduler.requests.get(req_id)
                if request is None or request.num_output_tokens > 0:
                    del self._awaiting[req_id]
                    self._scored.append((made[1], made[2], now - made[0]))
                    entry = self._dbg.pop(req_id, None)
                    if entry is not None and self._detail is not None:
                        entry[1]["actual"] = now - made[0]
                        self._detail.append(entry[1])
        self._last_output = now
        self._publish(scheduler, now)
        self._hook_seconds += time.monotonic() - now
        self._hook_calls += 1

    def pace(self, now: float | None = None) -> float:
        return self.pace_model.value(time.monotonic() if now is None else now)

    def on_idle_step_start(self) -> None:
        """A step without requests starts. An arrival waits for it to end."""
        snap = self.snapshot
        if snap.inflight:
            return
        now = time.monotonic()
        self._idle_started = now
        self.snapshot = replace(
            snap,
            time=now,
            exec_started=now,
            idle_step=max(self.idle_step_seconds, 1e-6),
        )

    def on_idle_step_end(self) -> None:
        if not self._idle_started:
            return
        now = time.monotonic()
        seconds = now - self._idle_started
        self._idle_started = 0.0
        self._idle_steps += 1
        if self._idle_steps == 1:
            self.idle_step_seconds = seconds
        else:
            self.idle_step_seconds += 0.2 * (seconds - self.idle_step_seconds)
        self.snapshot = replace(self.snapshot, time=now, idle_step=0.0)

    def _note_restore(self, request: Any, now: float) -> None:
        """Time the request's wait for a KV connector load, start to end."""
        status = getattr(request, "status", None)
        restoring = getattr(status, "name", "") == "WAITING_FOR_REMOTE_KVS"
        started = getattr(request, "forecast_restore_at", None)
        if restoring:
            if started is None:
                request.forecast_restore_at = now
        elif started is not None:
            request.forecast_restore_at = None
            seconds = now - started
            self._restores += 1
            if self._restores == 1:
                self.restore_seconds = seconds
            else:
                self.restore_seconds += 0.2 * (seconds - self.restore_seconds)

    def _cached_now(self, scheduler: Any, request: Any, believed: int) -> int:
        """Cached prefix length of a waiting request, read without side effects.

        Blocks are freed tail first, so the last block of the prefix being
        present means the whole prefix is; only otherwise is the prefix walked.
        """
        manager = scheduler.kv_cache_manager
        block = scheduler.block_size
        if believed >= block:
            last = believed // block - 1
            hit = manager.coordinator.find_longest_cache_hit(
                request.block_hashes[last : last + 1], block
            )[1]
            if hit:
                return believed
        # The last token is always recomputed.
        return manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_prompt_tokens - 1
        )[1]

    def _publish(self, scheduler: Any, now: float, refresh: bool = False) -> None:
        inflight = list(self._inflight.values())
        last_step_of: dict[str, int] = {}
        for i, step in enumerate(inflight):
            for req_id in step.req_ids:
                last_step_of[req_id] = i
        frees = [0] * len(inflight)
        prefills: list[tuple[int, int, bool, int]] = []
        decodes: list[tuple[int, int]] = []
        for request in scheduler.running:
            self._note_restore(request, now)
            remaining = request.num_prompt_tokens - request.num_computed_tokens
            max_tokens = request.max_tokens or 1
            if remaining > 0:
                prefills.append(
                    (remaining, request.num_computed_tokens, True, max_tokens - 1)
                )
                continue
            left = (
                max_tokens - request.num_output_tokens - request.num_output_placeholders
            )
            if left > 0:
                decodes.append((request.num_computed_tokens, left))
            elif (last := last_step_of.get(request.request_id)) is not None:
                # Finishes when its last scheduled step returns.
                frees[last] += 1
        refresh = refresh and self.refresh_budget > 0
        cursor = self._refresh_cursor
        deadline = now + self.refresh_budget
        position = -1
        # Requests holding KV blocks, such as those a connector is loading a
        # prefix for, are scheduled before the rest of the queue.
        queued = chain(scheduler.kv_holding_waiting, scheduler.waiting)
        for position, request in enumerate(queued):
            self._note_restore(request, now)
            remaining = getattr(request, "forecast_remaining", None)
            if remaining is None or request.num_computed_tokens > 0:
                remaining = request.num_prompt_tokens - request.num_computed_tokens
            elif refresh and position >= cursor:
                try:
                    cached = self._cached_now(
                        scheduler, request, request.num_prompt_tokens - remaining
                    )
                except Exception:
                    self.refresh_budget = 0.0
                    refresh = False
                else:
                    fresh = max(request.num_prompt_tokens - cached, 1)
                    if fresh != remaining:
                        request.forecast_remaining = remaining = fresh
                        self._refresh_changes += 1
                    cursor = position + 1
                    refresh = time.monotonic() < deadline
            if remaining <= 0:
                continue
            prefills.append(
                (
                    remaining,
                    request.num_prompt_tokens - remaining,
                    False,
                    (request.max_tokens or 1) - 1,
                )
            )
        # Start over once the end of the queue has been reached.
        self._refresh_cursor = 0 if cursor > position else cursor
        exec_started = (
            max(self._last_output, inflight[0].scheduled_at) if inflight else now
        )
        self.snapshot = Snapshot(
            time=now,
            exec_started=exec_started,
            inflight=tuple((s.features, f) for s, f in zip(inflight, frees)),
            prefills=tuple(prefills),
            decodes=tuple(decodes),
            seen_seq=self._seen_seq,
        )

    # ---- arrival side (engine input thread) ----

    def note_arrival(self, request: Any, cached_tokens: int) -> None:
        self._next_seq += 1
        remaining = max(request.num_prompt_tokens - cached_tokens, 1)
        ingest = time.time() - request.arrival_time
        if not 0.0 <= ingest < 60.0:
            ingest = 0.0
        if self.ready:
            now = time.monotonic()
            plain = self._last(
                self._replay(self.snapshot, now, cached_tokens, remaining)
            )
            expected = self._last(
                self._replay(self.snapshot, now, cached_tokens, remaining, True)
            )
            if plain is not None and expected is not None:
                # Scored from the server's arrival stamp, with the ingest time
                # forecast before this request's own is counted.
                ahead = self.ingest.predict(request.num_prompt_tokens)
                plain += ahead
                expected += ahead
                request.forecast_made = (now - ingest, plain, expected)
                if self._detail is not None:
                    snap = self.snapshot
                    request.forecast_dbg = {
                        "forecast": plain,
                        "ingest": ingest,
                        "ingest_forecast": ahead,
                        "new": remaining,
                        "snap_age": now - snap.time,
                        "idle_step": snap.idle_step,
                        "exec_age": now - snap.exec_started,
                        "inflight": len(snap.inflight),
                        "prefills": len(snap.prefills),
                        "idle_curve": curve_value(self.floor(cached_tokens), remaining)
                        or 0.0,
                    }
        self.ingest.observe(request.num_prompt_tokens, ingest)
        request.forecast_seq = self._next_seq
        request.forecast_remaining = remaining
        self._pending.append(
            (self._next_seq, remaining, cached_tokens, (request.max_tokens or 1) - 1)
        )

    # ---- forecast (any thread; reads one snapshot) ----

    @property
    def ready(self) -> bool:
        return self.model.coef is not None and self.model.num_samples >= 8

    @staticmethod
    def _last(segments: list[tuple[int, float, int, float]]) -> float | None:
        return segments[-1][3] if segments else None

    def _replay(
        self,
        snap: Snapshot,
        now: float,
        probe_cached: int,
        probe_tokens: int,
        expect_arrivals: bool = False,
        pending: bool = True,
        pace: float | None = None,
    ) -> list[tuple[int, float, int, float]]:
        """Replay scheduling with a probe request appended at `now`.

        With `expect_arrivals`, steps the probe would have to itself are
        shared with other requests as often as the engine has recently been
        busy: the long-prefill cap applies for that share of the step and
        the step carries that share of a typical step's tokens.

        Returns one segment per step that computes probe tokens:
        (tokens before, seconds if the probe ended just after `tokens before`,
        tokens after, seconds if it ended at `tokens after`).
        """
        predict = self.model.predict
        if pace is None:
            pace = self.pace(now)
        if pace != 1.0:
            fitted = predict

            def predict(x: Features) -> float:
                return fitted(x) * pace

        budget_cfg = self.token_budget
        threshold = self.long_prefill_threshold
        share = self.busy_now() if expect_arrivals else 0.0
        shared_cap = threshold
        if shared_cap and self.adaptive_threshold:
            shared_cap = max(shared_cap, budget_cfg // 2)
        while self._pending and self._pending[0][0] <= snap.seen_seq:
            self._pending.popleft()
        reqs = [_SimReq(r, c, s, d) for r, c, s, d in snap.prefills]
        if pending:
            reqs += [_SimReq(r, c, False, d) for _, r, c, d in list(self._pending)]
        probe = _SimReq(probe_tokens, probe_cached, False, 0, probe=True)
        reqs.append(probe)

        # Decoding requests as one group: each takes one token per step. The
        # heap holds (last step, context - step at entry) to drop them on time.
        step = 0
        finish: list[tuple[int, int]] = [(left, c) for c, left in snap.decodes]
        heapq.heapify(finish)
        num_decodes = len(finish)
        decode_ctx = float(sum(c for c, _ in snap.decodes))
        seats = num_decodes + sum(1 for r in reqs if r.seated)
        seats += sum(f for _, f in snap.inflight)

        ends: deque[_SimBatch] = deque()
        t_end = max(now, snap.exec_started)
        if snap.idle_step:
            ends.append(_SimBatch(end=max(snap.exec_started + snap.idle_step, now)))
        for i, (x, f) in enumerate(snap.inflight):
            t_end = (snap.exec_started if i == 0 else t_end) + predict(x)
            t_end = max(t_end, now)
            ends.append(_SimBatch(end=t_end, frees=f))

        segments: list[tuple[int, float, int, float]] = []
        t = now
        probe_total = 0
        for _ in range(self.max_replay_steps):
            while len(ends) < self.queue_size:
                batch = _SimBatch()
                budget = budget_cfg
                eligible = num_decodes + len(reqs)
                cap = threshold if threshold > 0 and eligible > 1 else 0
                if cap and self.adaptive_threshold:
                    cap = max(cap, budget_cfg // eligible)
                step += 1
                if num_decodes:
                    n = min(num_decodes, budget)
                    budget -= n
                    batch.tokens += n
                    batch.ctx_sum += decode_ctx
                    batch.tok_ctx_sum += decode_ctx
                    decode_ctx += n
                    while finish and finish[0][0] <= step:
                        _, base = heapq.heappop(finish)
                        num_decodes -= 1
                        decode_ctx -= base + step
                        batch.frees += 1
                for r in reqs:
                    if budget <= 0:
                        break
                    if r.remaining <= 0:
                        continue
                    if not r.seated:
                        if seats >= self.max_seats:
                            break
                        r.seated = True
                        seats += 1
                    n = min(r.remaining, budget)
                    if cap and n > cap:
                        n = cap
                    elif share and eligible == 1 and r.probe and probe_total:
                        # A later arrival can only join after the probe's
                        # first step.
                        if shared_cap and n > shared_cap:
                            n = max(int(n - share * (n - shared_cap)), 1)
                        others = share * min(self.tokens_per_step, budget - n)
                        batch.tokens += int(others)
                    budget -= n
                    batch.tokens += n
                    batch.ctx_sum += r.context
                    batch.tok_ctx_sum += n * (r.context + n / 2)
                    if r.probe:
                        batch.probe_tokens = n
                        batch.probe_context = r.context
                        probe_total += n
                        batch.probe_total = probe_total
                    r.context += n
                    r.remaining -= n
                    if r.remaining == 0:
                        batch.done.append(r)
                if batch.tokens == 0:
                    step -= 1
                    break
                batch.step = step
                batch.start = max(t, ends[-1].end if ends else t)
                batch.end = batch.start + predict(
                    StepTimeModel.features(
                        batch.tokens, batch.ctx_sum, batch.tok_ctx_sum
                    )
                )
                ends.append(batch)
            if not ends:
                break
            batch = ends.popleft()
            t = batch.end
            seats -= batch.frees
            for r in batch.done:
                if r.decode_after > 0:
                    heapq.heappush(finish, (step + r.decode_after, r.context - step))
                    num_decodes += 1
                    decode_ctx += r.context
                else:
                    seats -= 1
            if batch.done:
                reqs = [r for r in reqs if r.remaining > 0]
            if batch.probe_tokens:
                # Same step with the probe taking a single token.
                n = batch.probe_tokens
                c = batch.probe_context
                short = StepTimeModel.features(
                    batch.tokens - n + 1,
                    batch.ctx_sum,
                    batch.tok_ctx_sum - n * (c + n / 2) + c,
                )
                segments.append(
                    (
                        batch.probe_total - n,
                        batch.start + predict(short) - now,
                        batch.probe_total,
                        batch.end - now,
                    )
                )
                if batch.probe_total >= probe_tokens:
                    break
        return segments

    def curve(
        self,
        cached_tokens: int = 0,
        expect_arrivals: bool = False,
        tolerance: float = 0.02,
    ) -> list[tuple[int, float, int, float]]:
        """Seconds to first token by new tokens, as line segments.

        A request with `u` new tokens, `u0 < u <= u1`, is forecast at
        `t0 + (t1 - t0) * (u - u0) / (u1 - u0)`.
        """
        if not self.ready:
            return []
        now = time.monotonic()
        probe_tokens = max(self.max_model_len - cached_tokens, 1)
        segments = self._replay(
            self.snapshot, now, cached_tokens, probe_tokens, expect_arrivals
        )
        self._replay_seconds += time.monotonic() - now
        self._replay_calls += 1
        return self._with_ingest(self._merge(segments, tolerance), cached_tokens)

    def _with_ingest(
        self, segments: list[tuple[int, float, int, float]], cached_tokens: int
    ) -> list[tuple[int, float, int, float]]:
        """Add the server's ingest time for a prompt of each size."""
        ingest = self.ingest.predict
        return [
            (u0, t0 + ingest(cached_tokens + u0), u1, t1 + ingest(cached_tokens + u1))
            for u0, t0, u1, t1 in segments
        ]

    @staticmethod
    def _merge(
        segments: list[tuple[int, float, int, float]], tolerance: float
    ) -> list[tuple[int, float, int, float]]:
        """Join consecutive segments that lie on one line within `tolerance`."""
        merged: list[tuple[int, float, int, float]] = []
        group: list[tuple[int, float, int, float]] = []

        def fits(seg: tuple[int, float, int, float]) -> bool:
            # Every join inside the group must stay on the line to seg's end.
            u0, t0 = group[0][0], group[0][1]
            slope = (seg[3] - t0) / (seg[2] - u0)
            for g, nxt in zip(group, group[1:] + [seg]):
                at = t0 + slope * (g[2] - u0)
                if abs(at - g[3]) > tolerance * g[3]:
                    return False
                if abs(at - nxt[1]) > tolerance * nxt[1]:
                    return False
            return True

        for seg in segments:
            if group and len(group) < 64 and fits(seg):
                group.append(seg)
                continue
            if group:
                merged.append((group[0][0], group[0][1], group[-1][2], group[-1][3]))
            group = [seg]
        if group:
            merged.append((group[0][0], group[0][1], group[-1][2], group[-1][3]))
        return merged

    def floor(
        self, cached_tokens: int = 0, tolerance: float = 0.02
    ) -> list[tuple[int, float, int, float]]:
        """The curve of this engine with nothing else to do.

        A consumer holding a curve that is `age` seconds old can use
        `max(floor, curve - age)`: queued work drains as time passes, the
        request's own compute does not.
        """
        if not self.ready:
            return []
        key = (cached_tokens, self.model.coef)
        now = time.monotonic()
        cached = self._floors.get(cached_tokens)
        if cached is not None and cached[0] == key:
            merged = cached[1]
        else:
            idle = Snapshot(time=now, exec_started=now)
            probe_tokens = max(self.max_model_len - cached_tokens, 1)
            merged = self._merge(
                self._replay(
                    idle, now, cached_tokens, probe_tokens, pending=False, pace=1.0
                ),
                tolerance,
            )
            self._floors[cached_tokens] = (key, merged)
        # Every step of an idle engine's curve stretches with the pace.
        pace = self.pace(now)
        if pace != 1.0:
            merged = [(u0, t0 * pace, u1, t1 * pace) for u0, t0, u1, t1 in merged]
        return self._with_ingest(merged, cached_tokens)

    def ttft(
        self, new_tokens: int, cached_tokens: int = 0, expect_arrivals: bool = False
    ) -> float | None:
        if not self.ready:
            return None
        now = time.monotonic()
        new_tokens = max(new_tokens, 1)
        segments = self._replay(
            self.snapshot, now, cached_tokens, new_tokens, expect_arrivals
        )
        if not segments:
            return None
        u0, t0, u1, t1 = segments[-1]
        ingest = self.ingest.predict(cached_tokens + new_tokens)
        if new_tokens >= u1:
            return t1 + ingest
        return t0 + (t1 - t0) * (new_tokens - u0) / max(u1 - u0, 1) + ingest

    def decode_step(self, context_tokens: int = 0) -> float | None:
        """Seconds per output token if one more request decoded now."""
        if not self.ready:
            return None
        snap = self.snapshot
        ctx = float(sum(c for c, _ in snap.decodes) + context_tokens)
        return self.pace() * self.model.predict(
            StepTimeModel.features(len(snap.decodes) + 1, ctx, ctx)
        )

    def self_check(self) -> dict[str, Any]:
        """How the engine's own recent forecasts compared with what happened."""
        scored = list(self._scored)
        if len(scored) < 5:
            return {"n": len(scored)}
        a = np.array(scored)
        out: dict[str, Any] = {"n": len(scored)}
        for name, col in (("plain", 0), ("expected", 1)):
            err = np.abs(a[:, col] - a[:, 2]) / np.maximum(a[:, 2], 1e-6)
            out[name] = {
                "median_pct": float(np.median(err) * 100),
                "p90_pct": float(np.percentile(err, 90) * 100),
                "bias_s": float(np.mean(a[:, col] - a[:, 2])),
            }
        return out

    def busy_now(self) -> float:
        """Share of recent time the engine was busy, as of now.

        `busy_share` moves only when a step returns, so an engine that went
        idle would keep its last value; the time since its last step counts
        as idle here.
        """
        if self._inflight or not self._last_output:
            return self.busy_share
        idle = max(time.monotonic() - self._last_output, 0.0)
        return self.busy_share * math.exp(-idle / self.load_window)

    def describe(self) -> dict[str, Any]:
        snap = self.snapshot
        return {
            "ready": self.ready,
            "age": time.monotonic() - snap.time,
            "step_samples": self.model.num_samples,
            "step_coef": self.model.coef,
            "inflight_steps": len(snap.inflight),
            "prefills": len(snap.prefills),
            "decodes": len(snap.decodes),
            "pending": sum(1 for p in list(self._pending) if p[0] > snap.seen_seq),
            "queue_size": self.queue_size,
            "token_budget": self.token_budget,
            "idle_steps": self._idle_steps,
            "idle_step_seconds": self.idle_step_seconds,
            "pace": self.pace(),
            "restores": self._restores,
            "restore_seconds": self.restore_seconds,
            "ingest_seconds": self.ingest.base,
            "ingest_seconds_per_token": self.ingest.per_token,
            **(
                {"detail": list(self._detail)[-16:]} if self._detail is not None else {}
            ),
            "busy_share": self.busy_now(),
            "tokens_per_step": self.tokens_per_step,
            "self_check": self.self_check(),
            "refresh_changes": self._refresh_changes,
            "hook_us": 1e6 * self._hook_seconds / max(self._hook_calls, 1),
            "hook_calls": self._hook_calls,
            "curve_us": 1e6 * self._replay_seconds / max(self._replay_calls, 1),
            "curve_calls": self._replay_calls,
        }
