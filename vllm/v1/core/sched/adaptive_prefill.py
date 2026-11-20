# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, soft-SLO prefill admission with delayed device timing feedback.

No CUDA calls or model dependencies belong here. A budget is provisional until
the scheduler commits its actual allocation result; failed allocations cost no
service credit. Memory feasibility remains the KV manager's responsibility.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median


@dataclass
class PrefillBudget:
    limit_ms: float
    token_ms: float
    max_tokens: int
    request_limit_ms: float | None = None
    force_progress: bool = False
    costs: dict[str, tuple[float, int]] = field(default_factory=dict)

    @property
    def used_ms(self) -> float:
        return sum(cost for cost, _ in self.costs.values())

    def limit(self, request_id: str, tokens: int, context: int) -> int:
        # A conservative context-weighted bootstrap, refined by device feedback.
        unit_ms = self.token_ms * (1.0 + context / 8192)
        available = max(0.0, self.limit_ms - self.used_ms)
        if self.force_progress and not self.costs and available < unit_ms:
            # One aged token may exceed the soft time target; the controller
            # bounds the number of outstanding steps that can use this escape.
            return min(tokens, self.max_tokens, 1)
        if self.request_limit_ms is not None:
            # A long running prompt must leave admission room for other prompts.
            # Permit one token when a fair share is smaller than a token's cost.
            available = min(available, max(unit_ms, self.request_limit_ms))
        return min(tokens, self.max_tokens, int(available / unit_ms))

    def admit(self, request_id: str, tokens: int, context: int) -> None:
        self.costs[request_id] = (
            tokens * self.token_ms * (1.0 + context / 8192),
            tokens,
        )

    def refund(self, request_id: str) -> None:
        self.costs.pop(request_id, None)


@dataclass(frozen=True)
class PendingStep:
    units: float
    decode_reqs: int
    reserved_ms: float


class AdaptivePrefillController:
    """Feedback controller, not a hard latency or overload guarantee.

    Service credits give prefill a bounded allowance even when decode consumes
    the estimated target. This deliberately trades a soft decode target against
    progress; it never overrides the engine's allocation/alignment rules.
    """

    def __init__(
        self,
        target_ms: float = 40,
        token_ms: float = 0.02,
        min_share: float = 0.1,
        max_wait_ms: float = 250,
    ):
        if not all(
            math.isfinite(x) and x > 0
            for x in (target_ms, token_ms, min_share, max_wait_ms)
        ):
            raise ValueError("Adaptive prefill parameters must be finite and positive")
        if min_share >= 1:
            raise ValueError("min_share must be less than one")
        self.target_ms = target_ms
        self.token_ms = token_ms
        self.min_share = min_share
        self.max_wait_ms = max_wait_ms
        self.decode_ms = 1.0
        self.step_ms = 0.0
        self._has_step_baseline = False
        self._decode_timings: deque[tuple[int, float]] = deque(maxlen=32)
        self._prefill_costs: deque[float] = deque(maxlen=5)
        self._prefill_sample: tuple[float, float] | None = None
        self.scale = 1.0
        self.credit_ms = 0.0
        self.last_progress: dict[str, float] = {}
        self.pending: dict[int, PendingStep] = {}

    def begin(
        self, now: float, decode_reqs: int, max_tokens: int, prefill_ids: set[str]
    ) -> PrefillBudget:
        self.last_progress = {
            rid: self.last_progress.get(rid, now) for rid in prefill_ids
        }
        if not prefill_ids:
            self.credit_ms = 0
        else:
            self.credit_ms = min(
                self.target_ms, self.credit_ms + self.target_ms * self.min_share
            )
        inflight = sum(step.reserved_ms for step in self.pending.values())
        step_allowance = max(
            0.0,
            self.target_ms * self.scale - self.step_ms - self.decode_ms * decode_reqs,
        )
        allowance = max(0.0, step_allowance - inflight)
        aged = any(
            (now - t) * 1000 >= self.max_wait_ms for t in self.last_progress.values()
        )
        if aged or decode_reqs == 0:
            # Already-issued work must finish before spending the same reserve.
            allowance = max(allowance, max(0, self.credit_ms - inflight))
        if (
            decode_reqs == 0
            and not any(step.decode_reqs for step in self.pending.values())
            and len(self.pending) < 2
        ):
            # With no decode in the pipeline, allow the next prefill step its
            # own time budget instead of repeatedly splitting one queued budget.
            allowance = max(allowance, step_allowance)
        if len(self.pending) >= 2:
            allowance = 0.0
        return PrefillBudget(
            allowance,
            self.token_ms,
            max_tokens,
            allowance / max(1, len(prefill_ids)),
            force_progress=aged and len(self.pending) < 2,
        )

    def commit(
        self,
        step_id: int,
        budget: PrefillBudget,
        actual: dict[str, int],
        contexts: dict[str, int],
        decode_reqs: int,
        now: float,
    ) -> None:
        units = 0.0
        for rid, count in actual.items():
            if rid in contexts:
                units += count * (1.0 + contexts[rid] / 8192)
                self.last_progress[rid] = now
        cost = units * budget.token_ms
        self.credit_ms = max(0.0, self.credit_ms - cost)
        self.pending[step_id] = PendingStep(units, decode_reqs, cost)

    def complete(self, step_id: int, device_ms: float | None) -> None:
        step = self.pending.pop(step_id, None)
        if step is None or device_ms is None:
            return
        if not math.isfinite(device_ms) or device_ms <= 0:
            return
        if device_ms > self.target_ms:
            self.scale = max(0.1, self.scale * 0.7)
        elif device_ms < self.target_ms * 0.8:
            self.scale = min(1.0, self.scale + 0.02)
        if step.units:
            if not step.decode_reqs:
                if self._has_step_baseline:
                    self.step_ms = min(self.step_ms, device_ms)
                elif self._prefill_sample is not None:
                    units, elapsed = self._prefill_sample
                    delta = step.units - units
                    if abs(delta) >= 1:
                        slope = (device_ms - elapsed) / delta
                        overhead = device_ms - slope * step.units
                        if slope > 0 and overhead >= 0:
                            self.step_ms = overhead
                            self._has_step_baseline = True
                self._prefill_sample = (step.units, device_ms)
            # One timing cannot distinguish fixed overhead from per-token cost.
            # Keep the bootstrap until a decode baseline or varying prefill sizes
            # identify overhead, and use a short median to reject isolated spikes.
            if self._has_step_baseline:
                residual = device_ms - self.step_ms - self.decode_ms * step.decode_reqs
                if residual > 0:
                    self._prefill_costs.append(residual / step.units)
                if len(self._prefill_costs) >= 3:
                    measured = median(self._prefill_costs)
                    weight = 0.5 if measured > self.token_ms else 0.05
                    self.token_ms = max(
                        1e-6, (1 - weight) * self.token_ms + weight * measured
                    )
        elif step.decode_reqs:
            self._decode_timings.append((step.decode_reqs, device_ms))
            timings: dict[int, float] = {}
            for count, elapsed in self._decode_timings:
                timings[count] = min(timings.get(count, elapsed), elapsed)
            mean_count = sum(timings) / len(timings)
            mean_ms = sum(timings.values()) / len(timings)
            variance = sum((count - mean_count) ** 2 for count in timings)
            self.decode_ms = (
                max(
                    0.0,
                    sum(
                        (count - mean_count) * (elapsed - mean_ms)
                        for count, elapsed in timings.items()
                    )
                    / variance,
                )
                if variance
                else 0.0
            )
            # A single decode population identifies only the whole batch cost.
            self.step_ms = max(0.0, mean_ms - self.decode_ms * mean_count)
            self._has_step_baseline = True
