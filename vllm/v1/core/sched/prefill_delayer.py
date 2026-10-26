# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-DP prefill delayer.

The DP ranks of a MoE model step in lockstep, so a step in which any rank runs
prefill is a long step for every rank. Once per engine step all ranks
all-reduce their prefill state and take the same decision: FIRE lets every rank
with prefill work run it in the next step, HOLD defers all prefill (new requests
and in-progress chunks) on every rank. The rules follow ATOM's PrefillDelayer.
"""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler

# FIRE once the pending prefill tokens fill this fraction of the prefillable
# ranks' token budgets.
FILL_TARGET = 0.9
# Upper bounds on HOLD, in steps.
TTFT_MAX_HOLD_STEPS = 200
PARTIAL_MAX_HOLD_STEPS = 100
STALL_STEPS = 10
# FIRE when a prefillable rank's KV cache usage reaches this fraction.
KV_HIGH_USAGE = 0.9


def local_state(scheduler: "Scheduler", token_budget: int) -> list[int]:
    """[prefillable, pending prefill tokens (capped), decodes, KV high,
    chunk in progress, prefill ran since the last call] of one rank."""
    pending = decodes = partial = waiting = 0
    for request in scheduler.running:
        if request.is_prefill_chunk:
            partial = 1
            pending += max(request.num_tokens - request.num_computed_tokens, 0)
        else:
            decodes += 1
    for request in scheduler.waiting:
        if request in scheduler.deferred_waiting:
            continue
        waiting = 1
        if pending >= token_budget:
            break
        pending += max(request.num_tokens - request.num_computed_tokens, 0)
    prefillable = int(bool(partial or waiting))
    kv_high = int(
        bool(prefillable) and scheduler.kv_cache_manager.usage >= KV_HIGH_USAGE
    )
    ran = int(scheduler.ran_prefill)
    scheduler.ran_prefill = False
    return [prefillable, min(pending, token_budget), decodes, kv_high, partial, ran]


class PrefillDelayer:
    def __init__(self, dp_size: int, token_budget: int, protected_steps: int):
        self.dp_size = dp_size
        self.token_budget = token_budget
        self.protected_steps = protected_steps
        self._buf = torch.zeros(6, dtype=torch.int64)
        self._first = True
        self.reset()

    def reset(self) -> None:
        self._protected_left = 0
        self._hold_steps = 0
        self._stall_steps = 0
        self._prev_pending = -1

    def step(self, scheduler: "Scheduler", dp_group: dist.ProcessGroup) -> bool:
        """Returns whether the next step may run prefill. Must run on every DP
        rank in every engine step (a CPU all-reduce on the DP group)."""
        self._buf.copy_(
            torch.tensor(local_state(scheduler, self.token_budget), dtype=torch.int64)
        )
        dist.all_reduce(self._buf, op=dist.ReduceOp.SUM, group=dp_group)
        fire, _ = self.decide(*self._buf.tolist())
        return fire

    def decide(
        self,
        num_prefillable: int,
        pending: int,
        decodes: int,
        kv_high: int,
        partial: int,
        ran: int,
    ) -> tuple[bool, str]:
        """FIRE (True) or HOLD (False) from the DP-summed state, with the rule
        that decided it."""
        if ran:
            self._protected_left = self.protected_steps
        if self._first:
            self._first = False
            return self._fire("first")
        if self._protected_left > 0:
            self._protected_left -= 1
            if num_prefillable and decodes:
                return False, "protected"
        if num_prefillable == 0:
            return self._fire("no_prefill")
        if decodes == 0:
            return self._fire("no_decode")
        if kv_high:
            return self._fire("kv_high")
        if self._hold_steps >= TTFT_MAX_HOLD_STEPS:
            return self._fire("ttft")
        if partial and self._hold_steps >= PARTIAL_MAX_HOLD_STEPS:
            return self._fire("partial")
        if num_prefillable < self.dp_size:
            return self._hold("align")
        if pending >= FILL_TARGET * num_prefillable * self.token_budget:
            return self._fire("fill")
        if pending <= self._prev_pending:
            self._stall_steps += 1
        else:
            self._stall_steps = 0
        self._prev_pending = pending
        if self._stall_steps >= STALL_STEPS:
            return self._fire("stall")
        return self._hold("fill")

    def _fire(self, reason: str) -> tuple[bool, str]:
        self._hold_steps = 0
        self._stall_steps = 0
        self._prev_pending = -1
        return True, reason

    def _hold(self, reason: str) -> tuple[bool, str]:
        self._hold_steps += 1
        return False, reason
