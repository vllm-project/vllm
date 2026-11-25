# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workload snapshots that accompany iteration records.

All lengths are logical token counts, not physical KV cache occupancy.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass(frozen=True, slots=True)
class ScheduledWorkload:
    """KV context lengths of the requests scheduled in one iteration.

    A request's context is what it has computed before the iteration. Its
    prefill length is that context plus half of the tokens scheduled for it,
    about the mean context of those tokens. The variances are population
    variances. The request and token counts are in the iteration's record,
    where prefill is `num_ctx_*` and decode is `num_generation_*`. With async
    scheduling and speculative decoding, a decode context can exceed the true
    one by the draft tokens that the previous, still running iteration rejects.
    """

    sum_prefill_kv_tokens: int = 0
    var_prefill_length: float = 0.0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


@dataclass(frozen=True, slots=True)
class QueuedWorkload:
    """Requests in the waiting queues when the scheduler stats were made,
    deferred ones included. Stats are made on engine steps, so an idle engine
    does not update it.

    Preempted requests and requests waiting for remote KV count as queued
    decode work with their computed tokens, which a preemption resets to zero
    and an admitted remote load sets before the KV arrives. The other requests,
    streaming sessions that wait for more input included, count as queued
    prefill work with all their tokens. This is a queue category, not an
    execution phase: a resumed request may have to recompute its prefill.
    """

    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


def _variance(count: int, total: int, total_squares: int) -> float:
    """Population variance, exact up to the final division."""
    if not count:
        return 0.0
    return (count * total_squares - total * total) / (count * count)


def _variance_of(lengths: list[int]) -> float:
    return _variance(len(lengths), sum(lengths), sum(x * x for x in lengths))


def make_scheduled_workload(
    new_requests: Iterable["Request"],
    running_requests: Iterable["Request"],
    num_scheduled_tokens: Mapping[str, int],
) -> ScheduledWorkload:
    """Summarize the scheduled requests before their computed tokens advance.

    Prefill and decode requests are told apart as in `IterationDetails`.
    """
    prefill_kv_tokens = 0
    # Twice the prefill length, to keep half tokens exact.
    prefill_lengths: list[int] = []
    decode_lengths: list[int] = []
    for requests, is_new in ((new_requests, True), (running_requests, False)):
        for request in requests:
            kv_tokens = request.num_computed_tokens
            if is_new or request.is_prefill_chunk:
                prefill_kv_tokens += kv_tokens
                prefill_lengths.append(
                    2 * kv_tokens + num_scheduled_tokens[request.request_id]
                )
            else:
                decode_lengths.append(kv_tokens)
    return ScheduledWorkload(
        sum_prefill_kv_tokens=prefill_kv_tokens,
        var_prefill_length=_variance_of(prefill_lengths) / 4,
        sum_decode_kv_tokens=sum(decode_lengths),
        var_decode_kv_tokens=_variance_of(decode_lengths),
    )


class QueuedWorkloadTally:
    """The `QueuedWorkload` of the waiting queues, kept as requests enter and
    leave them.

    A request is remembered as it was counted, so that leaving subtracts
    exactly that. One whose status or token counts change while it waits must
    be passed to `refresh`.
    """

    def __init__(self) -> None:
        # vllm.v1.request imports this module through vllm.v1.metrics.stats.
        from vllm.v1.request import RequestStatus

        self._decode_statuses = (
            RequestStatus.PREEMPTED,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
        )
        # request -> (its queue, whether it is decode work, its length)
        self._counted: dict[Request, tuple[object, bool, int]] = {}
        # [count, sum, sum of squares] of the prefill and of the decode lengths.
        self._totals: tuple[list[int], list[int]] = ([0, 0, 0], [0, 0, 0])

    def add(self, queue: object, request: "Request") -> None:
        """Count a request that entered `queue`, as it is now."""
        counted = self._counted.get(request)
        if counted is not None:
            self.discard(counted[0], request)
        is_decode = request.status in self._decode_statuses
        length = request.num_computed_tokens if is_decode else request.num_tokens
        self._counted[request] = (queue, is_decode, length)
        totals = self._totals[is_decode]
        totals[0] += 1
        totals[1] += length
        totals[2] += length * length

    def discard(self, queue: object, request: "Request") -> None:
        """Stop counting a request that left `queue`, if it was in it."""
        counted = self._counted.get(request)
        if counted is not None and counted[0] is queue:
            del self._counted[request]
            _, is_decode, length = counted
            totals = self._totals[is_decode]
            totals[0] -= 1
            totals[1] -= length
            totals[2] -= length * length

    def refresh(self, request: "Request") -> None:
        """Recount a waiting request that changed in place."""
        counted = self._counted.get(request)
        if counted is not None:
            self.add(counted[0], request)

    def summary(self) -> QueuedWorkload:
        prefill, decode = self._totals
        return QueuedWorkload(
            num_prefill_requests=prefill[0],
            sum_prefill_tokens=prefill[1],
            var_prefill_length=_variance(*prefill),
            num_decode_requests=decode[0],
            sum_decode_kv_tokens=decode[1],
            var_decode_kv_tokens=_variance(*decode),
        )
