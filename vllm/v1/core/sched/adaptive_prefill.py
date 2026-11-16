# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, soft-SLO prefill admission with delayed device timing feedback.

No CUDA calls or model dependencies belong here. A budget is provisional until
the scheduler commits its actual allocation result; failed allocations cost no
service credit. Memory feasibility remains the KV manager's responsibility.
"""

import math
from dataclasses import dataclass, field


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
            # One aged token may exceed the soft time target. Never issue this
            # escape quantum while an earlier batch is still outstanding.
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
        allowance = max(
            0.0, self.target_ms * self.scale - self.decode_ms * decode_reqs - inflight
        )
        aged = any(
            (now - t) * 1000 >= self.max_wait_ms for t in self.last_progress.values()
        )
        if aged or decode_reqs == 0:
            # Already-issued work must finish before spending the same reserve.
            allowance = max(allowance, max(0, self.credit_ms - inflight))
        if decode_reqs == 0 and not self.pending:
            allowance = max(allowance, self.target_ms)
        return PrefillBudget(
            allowance,
            self.token_ms,
            max_tokens,
            allowance / max(1, len(prefill_ids)),
            force_progress=aged and not self.pending,
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
            residual = max(0.001, device_ms - self.decode_ms * step.decode_reqs)
            measured = residual / step.units
            # React quickly to underestimation; decay estimates slowly.
            weight = 0.5 if measured > self.token_ms else 0.05
            self.token_ms = max(1e-6, (1 - weight) * self.token_ms + weight * measured)
        elif step.decode_reqs:
            self.decode_ms = 0.9 * self.decode_ms + 0.1 * device_ms / step.decode_reqs
