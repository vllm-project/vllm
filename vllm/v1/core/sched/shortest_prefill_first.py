# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request ordering for the "shortest_prefill_first" scheduling policy.

Before each step, requests whose remaining prefill length is known are sorted
by it, shortest first, within each queue:

- running: prefill chunks, by prompt tokens not yet computed.
- kv_holding_waiting: by tokens not yet computed (their external hit is final).
- waiting / async_kv_waiting: requests whose external (KV connector) lookup has
  finished, by prompt tokens not in that hit. A hit of 0 is known.

A request whose remaining length is unknown (no or unfinished external lookup,
external reads skipped, or a decode) keeps its exact position: known requests
are only permuted among the positions they already hold, and ties keep their
current order. Only the order changes; budgets, chunking, lookups, loads,
allocation and preemption follow the scheduler unchanged, so the preemption
victim (the last running request) is now the one with the most prefill left.
"""

from collections.abc import Callable, MutableSequence
from itertools import islice
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1

logger = init_logger(__name__)

# Waiting-queue entries considered per step.
WINDOW = 64


def _sorted_slots(
    requests: list[Request], key: Callable[[Request], int | None]
) -> list[Request] | None:
    """Return `requests` with the known-key ones sorted (stably, ascending)
    among their own positions, or None if the order does not change."""
    slots = []
    keys = []
    for i, request in enumerate(requests):
        k = key(request)
        if k is not None:
            slots.append(i)
            keys.append(k)
    if len(slots) < 2:
        return None
    order = sorted(range(len(slots)), key=keys.__getitem__)
    if order == list(range(len(slots))):
        return None
    reordered = list(requests)
    for slot, src in zip(slots, order):
        reordered[slot] = requests[slots[src]]
    return reordered


class ShortestPrefillFirstOrder:
    """Sorts the scheduler's queues by known remaining prefill tokens."""

    def __init__(self, connector: "KVConnectorBase_V1 | None" = None) -> None:
        self.connector = connector
        logger.info("Shortest-prefill-first ordering ACTIVE (window=%d)", WINDOW)

    @staticmethod
    def _running_key(request: Request) -> int | None:
        remaining = request.num_prompt_tokens - request.num_computed_tokens
        return remaining if remaining > 0 else None

    @staticmethod
    def _holding_key(request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens

    def _waiting_key(self, request: Request) -> int | None:
        if self.connector is None or request.skip_reading_external_kv:
            return None
        if request.num_computed_tokens:
            return request.num_tokens - request.num_computed_tokens
        hit = self.connector.get_known_num_matched_tokens(request)
        if hit is None:
            return None
        return request.num_tokens - min(hit, request.num_tokens - 1)

    @staticmethod
    def _sort_list(requests: MutableSequence[Request], key) -> None:
        reordered = _sorted_slots(list(requests), key)
        if reordered is not None:
            requests[:] = reordered

    @staticmethod
    def _sort_queue(queue: FCFSRequestQueue, key, window: int | None) -> None:
        head = list(queue if window is None else islice(queue, window))
        reordered = _sorted_slots(head, key)
        if reordered is None:
            return
        # Drop the head in one pass and re-insert it in the new order.
        queue.remove_requests(head)
        queue.prepend_requests(reversed(reordered))

    def reorder(
        self,
        running: list[Request],
        kv_holding_waiting: FCFSRequestQueue,
        waiting_queues: tuple[FCFSRequestQueue, ...],
    ) -> None:
        self._sort_list(running, self._running_key)
        if len(kv_holding_waiting) > 1:
            self._sort_queue(kv_holding_waiting, self._holding_key, None)
        for queue in waiting_queues:
            if len(queue) > 1:
                self._sort_queue(queue, self._waiting_key, WINDOW)
