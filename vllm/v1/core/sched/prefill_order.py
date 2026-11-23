# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short-uncached-prefill-first ordering with aging (TokenSpeed
`enable_short_prefill_first`: `popShortPrefill` / `recordPrefillBypass`).

Only new prompts, i.e. requests that were never scheduled, are reordered: the
one with the fewest tokens left to compute after its prefix cache hit goes
first, ties keep submission order (`submission_key`: arrival time, then
request id, so it does not depend on how an earlier step reordered the queue).
Every other waiting request (preempted, resumed, KV-loaded or blocked) keeps its
FCFS position ahead of them.

Aging: each admission of a new prompt ahead of the oldest waiting new prompt
bumps the oldest's `prefill_bypassed`. Once it reaches `max_bypass` the oldest
goes first. Only the oldest ages, so short work keeps its priority after the
oldest is through, and a step that admits nothing ages nobody.

The ranking replays the aging that the admissions in its own order would cause,
which matches TokenSpeed re-evaluating the age after every admission within a
batch. The counter itself is only bumped by `note_admitted`, with the requests
the scheduler actually admitted.

This module is pure bookkeeping (no torch imports beyond the request status).
Ranking peeks the prefix cache of every new prompt on each call, at a cost that
grows with the cached hit length in blocks (about 0.9 ms for a 140K-token hit
with 64-token blocks on CPU). Results are not memoised, since a hit changes
whenever blocks are cached or evicted; reorder only on steps that may admit a
prefill (with the prefill cadence, the release steps).
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, TypeVar

from vllm.v1.request import RequestStatus


class _Orderable(Protocol):
    request_id: str
    status: RequestStatus
    arrival_time: float
    num_computed_tokens: int
    num_preemptions: int
    prefill_bypassed: int


class _PrefixCachePeek(Protocol):
    def peek_cached_prefix_len(self, request) -> int: ...


R = TypeVar("R", bound=_Orderable)


def is_new_prompt(request: _Orderable) -> bool:
    """Whether the request is a never-scheduled prompt (TokenSpeed `Submitted`).

    Must be evaluated before the request is admitted: admission sets
    `num_computed_tokens` and the status.
    """
    return (
        request.status == RequestStatus.WAITING
        and request.num_computed_tokens == 0
        and request.num_preemptions == 0
    )


def submission_key(request: _Orderable) -> tuple[float, str]:
    """TokenSpeed submission order: arrival time, then
    request id. Independent of the queue order, which earlier steps change."""
    return request.arrival_time, request.request_id


def oldest_new_prompt(requests: Iterable[R]) -> R | None:
    """The new prompt that ages: the first in `submission_key` order. None if
    there is no new prompt. The result does not depend on the order of
    `requests`, so it may be taken before or after `order_new_prompts`."""
    return min(filter(is_new_prompt, requests), key=submission_key, default=None)


def order_new_prompts(
    reqs: Iterable[R],
    uncached_fn: Callable[[R], int],
    max_bypass: int,
) -> list[R]:
    """Order waiting requests for admission.

    Args:
        reqs: The waiting requests, in queue order.
        uncached_fn: Tokens left to compute for a new prompt after its prefix
            cache hit. Called exactly once per new prompt.
        max_bypass: Admissions ahead of the oldest new prompt after which it
            goes first (TokenSpeed `kMaxPrefillBypasses` = 8).

    Returns:
        The requests that are not new prompts in their queue order, followed by
        the new prompts: shortest uncached work first, ties in submission
        order, except that an aged prompt goes first.

    """
    started: list[R] = []
    new: list[R] = []
    for request in reqs:
        (new if is_new_prompt(request) else started).append(request)
    if len(new) <= 1:
        return started + new

    new.sort(key=submission_key)
    num_new = len(new)
    work = [uncached_fn(r) for r in new]
    by_work = sorted(range(num_new), key=lambda i: (work[i], i))
    # Requests already aged by earlier steps, in submission order. Normally only
    # the oldest one can be aged; kept general for TokenSpeed parity.
    aged = [i for i in range(num_new) if new[i].prefill_bypassed >= max_bypass]

    taken = [False] * num_new
    ordered = started
    oldest = work_pos = aged_pos = 0
    # Bypasses of the current oldest, including the ones this order causes.
    oldest_bypassed = new[0].prefill_bypassed
    for _ in range(num_new):
        if taken[oldest]:
            while taken[oldest]:
                oldest += 1
            oldest_bypassed = new[oldest].prefill_bypassed
        while aged_pos < len(aged) and taken[aged[aged_pos]]:
            aged_pos += 1
        if oldest_bypassed >= max_bypass:
            pick = oldest
        elif aged_pos < len(aged):
            pick = aged[aged_pos]
        else:
            while taken[by_work[work_pos]]:
                work_pos += 1
            pick = by_work[work_pos]
        taken[pick] = True
        ordered.append(new[pick])
        if pick != oldest:
            oldest_bypassed += 1
    return ordered


def note_admitted(
    admitted: Sequence[_Orderable],
    oldest: _Orderable | None,
    new_prompts: Iterable[_Orderable] | None = None,
) -> None:
    """Age the oldest waiting new prompt by the admissions that bypass it.

    Args:
        admitted: The new prompts admitted this step, in admission order (only
            requests for which `is_new_prompt` held before admission).
        oldest: `oldest_new_prompt` of the waiting queue before admission.
            Ignored when `new_prompts` is given.
        new_prompts: The new prompts waiting before admission (a superset of
            `admitted`). When given, each admission is charged to the oldest
            new prompt still waiting at that point, as TokenSpeed
            `recordPrefillBypass` does, so one call per step reproduces it
            exactly. Otherwise only the admissions ahead of `oldest` are
            counted and the ones after it are charged to nobody.

    """
    if new_prompts is not None:
        candidates = sorted(new_prompts, key=submission_key)
        taken: set[int] = set()
        pos = 0
        for request in admitted:
            while pos < len(candidates) and id(candidates[pos]) in taken:
                pos += 1
            if pos < len(candidates) and candidates[pos] is not request:
                candidates[pos].prefill_bypassed += 1
            taken.add(id(request))
        return
    if oldest is None:
        return
    num_ahead = 0
    for request in admitted:
        if request is oldest:
            break
        num_ahead += 1
    oldest.prefill_bypassed += num_ahead


def make_uncached_fn(
    kv_cache_manager: _PrefixCachePeek,
    external_hit_fn: Callable[[_Orderable, int], int] | None = None,
) -> Callable[[_Orderable], int]:
    """`uncached_fn` for `order_new_prompts` from the local prefix cache.

    Args:
        kv_cache_manager: Provides the side-effect free
            `peek_cached_prefix_len`.
        external_hit_fn: Optional side-effect free estimate of the tokens a KV
            connector (e.g. a CPU tier) would add past the local hit, given the
            request and its local hit. TokenSpeed ranks by the longer of its
            device and host hits.

    The peek is repeated on every call (see the module docstring for its
    cost).

    """

    def uncached_fn(request) -> int:
        num_cached = kv_cache_manager.peek_cached_prefix_len(request)
        if external_hit_fn is not None:
            num_cached += external_hit_fn(request, num_cached)
        return max(request.num_tokens - num_cached, 0)

    return uncached_fn
