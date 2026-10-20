# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Transport-independent workload snapshots for completed engine iterations."""

from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING

from vllm.utils.math_utils import WelfordAccumulator

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import SchedulerInterface


@dataclass(frozen=True)
class ScheduledIterationWorkload:
    """Logical context lengths at the start of the scheduled model execution."""

    num_prefill_requests: int = 0
    num_prefill_tokens: int = 0
    num_decode_requests: int = 0
    num_decode_tokens: int = 0
    sum_prefill_kv_tokens: int = 0
    # Population variance of kv_read + scheduled_tokens / 2, in tokens squared.
    var_prefill_length: float = 0.0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


@dataclass(frozen=True)
class QueuedIterationWorkload:
    """Waiting work after output processing, or at dummy/idle submission.

    Preempted requests and requests waiting for remote KV are grouped as
    queued decode work. This queue category is distinct from the scheduler's
    execution phase: a resumed request may need prefill recomputation.
    """

    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


@dataclass(slots=True)
class IterationWorkloadAccumulator:
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_kv_tokens: int = 0
    prefill_lengths: WelfordAccumulator = field(default_factory=WelfordAccumulator)
    decode_lengths: WelfordAccumulator = field(default_factory=WelfordAccumulator)

    def add_prefill(self, kv_tokens: int, scheduled_tokens: int) -> None:
        self.prefill_tokens += scheduled_tokens
        self.prefill_kv_tokens += kv_tokens
        # Keep half-token attention lengths exact during integer accumulation.
        self.prefill_lengths.add(2 * kv_tokens + scheduled_tokens)

    def add_decode(self, kv_tokens: int, scheduled_tokens: int) -> None:
        self.decode_tokens += scheduled_tokens
        self.decode_lengths.add(kv_tokens)

    def snapshot(self) -> ScheduledIterationWorkload:
        return ScheduledIterationWorkload(
            num_prefill_requests=self.prefill_lengths.count,
            num_prefill_tokens=self.prefill_tokens,
            num_decode_requests=self.decode_lengths.count,
            num_decode_tokens=self.decode_tokens,
            sum_prefill_kv_tokens=self.prefill_kv_tokens,
            var_prefill_length=self.prefill_lengths.variance / 4,
            sum_decode_kv_tokens=self.decode_lengths.total,
            var_decode_kv_tokens=self.decode_lengths.variance,
        )


def extract_queued_workload(scheduler: "SchedulerInterface") -> QueuedIterationWorkload:
    from vllm.v1.request import RequestStatus

    waiting = scheduler.waiting
    kv_holding_waiting = scheduler.kv_holding_waiting
    assert waiting is not None
    assert kv_holding_waiting is not None
    prefill = WelfordAccumulator()
    decode = WelfordAccumulator()
    # Deferred requests are already in these disjoint queues.
    for request in chain(waiting, kv_holding_waiting):
        if request.status in (
            RequestStatus.PREEMPTED,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
        ):
            decode.add(request.num_computed_tokens)
        else:
            prefill.add(request.num_tokens)
    return QueuedIterationWorkload(
        num_prefill_requests=prefill.count,
        sum_prefill_tokens=prefill.total,
        var_prefill_length=prefill.variance,
        num_decode_requests=decode.count,
        sum_decode_kv_tokens=decode.total,
        var_decode_kv_tokens=decode.variance,
    )
