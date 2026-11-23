# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-clock prefill cadence (TokenSpeed `buildFusedPlan` semantics).

A step that carries prefill owes the running decodes `interval` (N) decode
rounds and, separately, `min_interval` (M) decode rounds. While a round is
owed, prefill is held as long as there is a decode to run instead; a step with
nothing to decode runs prefill as usual, so the engine never idles to honor
the cadence. Only rounds that actually ran a decode pay the debt down.

The N clock is suspended ("lifted") once any pending prompt has been held N
rounds in total, across its chunks and other prompts' intervals, which bounds
how long a prompt can wait. Nothing lifts the M clock: an overdue prompt's
chunks alternate with M decode rounds instead of running back to back.

This module is pure bookkeeping (no scheduler or torch imports); the scheduler
decides what a held step looks like.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class _Deferrable(Protocol):
    prefill_deferred_rounds: int


@dataclass(frozen=True)
class CadenceDecision:
    owes: bool
    """Decode rounds are owed: hold prefill unless there is nothing to decode."""
    n_lifted: bool
    """The N clock had rounds left but was suspended by aging or a full
    budget of pending prefill (diagnostics only)."""
    pending: bool = True
    """A prompt was pending, so a held step counts as held (diagnostics
    only)."""


class PrefillCadence:
    def __init__(
        self,
        interval: int,
        min_interval: int,
        release_on_full_budget: bool = False,
    ) -> None:
        self.interval = interval
        self.min_interval = min_interval
        self.release_on_full_budget = release_on_full_budget
        # Rounds still owed on each clock. Both start at 0, so the first step
        # is never gated.
        self.interval_left = 0
        self.min_interval_left = 0
        self._last_decision: CadenceDecision | None = None
        # Counters for logging.
        self.num_prefill_steps = 0
        self.num_held_steps = 0
        self.num_aging_lifts = 0

    @property
    def enabled(self) -> bool:
        return self.interval > 0 or self.min_interval > 0

    def decide(
        self,
        max_deferred_rounds: int,
        pending_prefill_tokens: int = 0,
        token_budget: int = 0,
        pending: bool = True,
    ) -> CadenceDecision:
        """Decide whether this step owes decode rounds.

        Args:
            max_deferred_rounds: the largest `prefill_deferred_rounds` among
                the prompts that could prefill now.
            pending_prefill_tokens: prefill tokens those prompts still need;
                only used with `release_on_full_budget`.
            token_budget: the step's token budget.
            pending: whether any prompt is pending; with none, a held step is
                not counted as held.

        """
        lift_interval = max_deferred_rounds >= self.interval or (
            self.release_on_full_budget
            and token_budget > 0
            and pending_prefill_tokens >= token_budget
        )
        owes = self.min_interval_left > 0 or (
            self.interval_left > 0 and not lift_interval
        )
        decision = CadenceDecision(
            owes=owes,
            n_lifted=lift_interval and self.interval_left > 0,
            pending=pending,
        )
        self._last_decision = decision
        return decision

    def commit(self, pushed_prefill: bool, pushed_decode: bool) -> None:
        """Account for the step just scheduled (forward.cpp buildFusedPlan)."""
        decision = self._last_decision
        self._last_decision = None
        if pushed_prefill:
            self.interval_left = self.interval
            self.min_interval_left = self.min_interval
            self.num_prefill_steps += 1
            if decision is not None and decision.n_lifted:
                self.num_aging_lifts += 1
        elif pushed_decode:
            self.interval_left = max(self.interval_left - 1, 0)
            self.min_interval_left = max(self.min_interval_left - 1, 0)
            if decision is not None and decision.owes and decision.pending:
                self.num_held_steps += 1


def note_deferred(requests: Iterable[_Deferrable]) -> None:
    """Count one held round on each prompt (TS `NotePrefillDeferred`)."""
    for request in requests:
        request.prefill_deferred_rounds += 1
