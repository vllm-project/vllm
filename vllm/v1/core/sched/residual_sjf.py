# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from collections.abc import Callable, Iterable

from vllm.v1.request import Request, RequestStatus


def residual_sjf_key(
    request: Request,
    cost_fn: Callable[[Request], int],
    now: float,
    max_wait_seconds: float,
) -> tuple[int, int, float, str]:
    if request.status != RequestStatus.RUNNING and (
        request.num_preemptions > 0 or request.num_computed_tokens > 0
    ):
        return 0, 0, request.arrival_time, request.request_id
    if now - request.arrival_time >= max_wait_seconds:
        return 1, 0, request.arrival_time, request.request_id
    return 2, cost_fn(request), request.arrival_time, request.request_id


def get_prefill_token_limits(
    running: list[Request],
    waiting: Iterable[Request],
    cost_fn: Callable[[Request], int],
    block_size: int,
    max_wait_ms: int,
    token_budget: int,
    input_budget: int,
    chunk_limit: int,
    free_slots: int,
    draft_slots: int,
) -> tuple[dict[str, int], bool]:
    """Share prefill budget at chunk boundaries without releasing resident KV.

    Waiting demand is an estimate; the scheduler still admits requests normally.
    The boolean reports a waiting reservation, so failed admissions can trigger
    a step without reservations to guarantee progress under KV pressure.
    """
    prefills = [r for r in running if r.is_prefill_chunk]
    if not prefills:
        return {}, False

    now = time.time()
    key = lambda r: residual_sjf_key(r, cost_fn, now, max_wait_ms / 1000)
    prefills.sort(key=key)
    running_prefills = deque(prefills)
    # Preserve admission phases: KV holders precede ordinary waiting requests.
    # Compare only the head of that stream with the next resident prefill.
    pending = deque(waiting if free_slots > 0 else ())
    limits = {}
    reserved_waiting = False
    while running_prefills or pending:
        if pending and (
            not running_prefills or key(pending[0]) < key(running_prefills[0])
        ):
            request = pending.popleft()
        else:
            request = running_prefills.popleft()
        is_running = request.status == RequestStatus.RUNNING
        if not is_running and free_slots <= 0:
            continue
        demand = (
            request.num_tokens - request.num_computed_tokens
            if is_running
            else max(1, cost_fn(request) * block_size)
        )
        if chunk_limit > 0:
            demand = min(demand, chunk_limit)
        allotted = max(0, min(demand, token_budget, input_budget - draft_slots))
        if is_running:
            limits[request.request_id] = allotted
        elif allotted:
            free_slots -= 1
            reserved_waiting = True
        token_budget -= allotted
        input_budget -= allotted + (draft_slots if allotted else 0)
    return limits, reserved_waiting
