# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-DP prefill delayer (`prefill_schedule_policy="delayer"`).

The DP ranks of a MoE model step in lockstep, so a step in which any rank
prefills is a long step for all of them. Once per step every rank reports its
prefill state, the reports are summed across the DP group, and every rank
takes the same decision from the sums: FIRE lets the ranks prefill in the next
step, HOLD defers prefill on the ranks that have decodes to run.

A HOLD waits, within bounds, for the ranks to align (each rank has prefill to
run, or waiting work that its peers' waiting cannot make admittable) and for
the pending prompt tokens to fill most of the prefilling ranks' token budgets.
All bounds count steps, which advance alike on every rank; the only wall-clock
bound compares each rank's own requests and is summed as a flag.
"""

from collections import Counter
from typing import NamedTuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# FIRE once the pending prompt tokens fill this fraction of the prefillable
# ranks' token budgets.
FILL_TARGET = 0.9
# Upper bounds on a HOLD, in steps; a prompt chunk in progress holds KV blocks.
MAX_HOLD_STEPS = 200
PARTIAL_MAX_HOLD_STEPS = 100
# FIRE once the group's work has not changed for this many steps.
STALL_STEPS = 10
# A rank at this KV cache usage cannot grow its prefill batch by waiting.
KV_HIGH_USAGE = 0.9
_LOG_EVERY = 1000


class PrefillDelayerState(NamedTuple):
    """One rank's prefill state, or its sum over the DP group."""

    # Has prefill it can run: a prompt chunk in progress or an admittable
    # waiting request.
    prefillable: int
    # Has waiting work it cannot admit now (no free request slot, KV nearly
    # full, or the requests are not ready); waiting on peers does not help it.
    blocked: int
    # Has neither, and none of its decodes can finish (freeing a slot for a new
    # request) within the remaining hold budget.
    unreachable: int
    # Prompt tokens left to prefill, capped at the token budget.
    pending_tokens: int
    decodes: int
    # Prefillable with KV cache usage at or above KV_HIGH_USAGE.
    kv_high: int
    # Has a prompt chunk in progress.
    partial: int
    # The oldest admittable waiting request has queued past
    # `prefill_max_wait_ms`.
    queue_hot: int
    # Scheduled prefill since the previous report.
    prefill_ran: int
    # Requests added since the previous report.
    arrived: int


class PrefillDelayer:
    def __init__(self, dp_size: int, token_budget: int, prefill_interval: int):
        self.dp_size = dp_size
        self.token_budget = token_budget
        # Decode-only steps after a prefill-carrying step.
        self.protected_steps = prefill_interval - 1
        # Summed across the DP group in place by the engine core.
        self.buffer = torch.zeros(len(PrefillDelayerState._fields), dtype=torch.int32)
        self._protected_left = 0
        self._decisions: Counter[str] = Counter()
        self.reset()

    def reset(self) -> None:
        """Starts a new hold episode (also at each DP wave boundary)."""
        self._hold_steps = 0
        self._stall_steps = 0
        self._prev_pending = -1
        self._prev_prefillable = -1

    @property
    def hold_budget(self) -> int:
        """Steps left before the current hold is released regardless."""
        return MAX_HOLD_STEPS - self._hold_steps

    def load(self, state: PrefillDelayerState) -> torch.Tensor:
        """Fills and returns the buffer to sum across the DP group."""
        self.buffer.copy_(torch.tensor(state, dtype=torch.int32))
        return self.buffer

    def step(self) -> bool:
        """Whether the next step may prefill, from the summed buffer."""
        fire, reason = self.decide(PrefillDelayerState(*self.buffer.tolist()))
        self._decisions[reason] += 1
        if self._decisions.total() % _LOG_EVERY == 0:
            logger.debug("Prefill delayer decisions: %s", dict(self._decisions))
        return fire

    def decide(self, state: PrefillDelayerState) -> tuple[bool, str]:
        """FIRE (True) or HOLD from the DP-summed state, with the rule that
        decided it."""
        if state.prefill_ran:
            self._protected_left = self.protected_steps
        if self._protected_left > 0:
            self._protected_left -= 1
            if state.prefillable and state.decodes:
                # Not a hold step, but work arriving meanwhile still counts.
                if self._work_changed(state):
                    self._stall_steps = 0
                return False, "protected"

        if not state.prefillable:
            return self._fire("no_prefill")
        # Bounds that release even an unaligned or underfilled prefill.
        if not state.decodes:
            return self._fire("no_decode")
        if state.kv_high:
            return self._fire("kv_high")
        if state.queue_hot:
            return self._fire("max_wait")
        if self._hold_steps >= MAX_HOLD_STEPS:
            return self._fire("max_hold")
        if state.partial and self._hold_steps >= PARTIAL_MAX_HOLD_STEPS:
            return self._fire("partial")

        # Pending tokens are capped per rank, so a flat sum alone does not mean
        # no new work: ranks turning (non-)prefillable and arrivals count too.
        if self._work_changed(state):
            self._stall_steps = 0
        else:
            self._stall_steps += 1
        stalled = self._stall_steps >= STALL_STEPS

        if state.prefillable + state.blocked < self.dp_size:
            # A rank without prefill gets work when one of its decodes
            # finishes; wait for it unless that cannot happen within the bound.
            if state.unreachable and stalled:
                return self._fire("stall")
            return self._hold("align")
        fill_target = FILL_TARGET * state.prefillable * self.token_budget
        if state.pending_tokens >= fill_target:
            return self._fire("fill")
        if stalled:
            return self._fire("stall")
        return self._hold("fill")

    def _work_changed(self, state: PrefillDelayerState) -> bool:
        changed = (
            state.pending_tokens > self._prev_pending
            or state.prefillable != self._prev_prefillable
            or state.arrived > 0
        )
        self._prev_pending = state.pending_tokens
        self._prev_prefillable = state.prefillable
        return changed

    def _fire(self, reason: str) -> tuple[bool, str]:
        self.reset()
        return True, reason

    def _hold(self, reason: str) -> tuple[bool, str]:
        self._hold_steps += 1
        return False, reason
