# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vllm-torchtpu's PrefixAwareScheduler
# (src/vllm_torchtpu/core/prefix_aware_scheduler.py), Copyright 2026 Google LLC,
# licensed under the Apache License, Version 2.0.
"""Shortest-prefill-first admission for the "shortest_prefill_first" policy.

FCFS admission is oblivious to whether a request's prompt prefix is already
cached. In multi-turn agentic workloads, admitting a long uncached prompt at
the head of the queue blocks ready follow-up requests whose prefixes are
already cached, and under high KV cache utilization can evict those cached
prefixes before they are reused.

Before each scheduling step, this inspects the ready queue within a lookahead
window and moves the request with the fewest uncached prompt tokens to the
front. Preempted requests rank ahead of all others, and ties go to the earliest
arrival. It modifies only the admission ordering; memory management, eviction
and preemption are unchanged.
"""

from itertools import islice
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.kv_cache_manager import KVCacheManager

logger = init_logger(__name__)

_LOG_EVERY = 1000

# Priority tiers: lower values admit earlier.
_TIER_PREEMPTED = 1
_TIER_NORMAL = 2


class ShortestPrefillFirstAdmission:
    """Admit ready requests prioritizing the fewest uncached prompt tokens."""

    #: Lookahead window of ready requests to evaluate for cached prefix length.
    WINDOW = 64

    def __init__(
        self,
        kv_cache_manager: "KVCacheManager",
        connector: "KVConnectorBase_V1 | None" = None,
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.connector = connector
        self._n = 0
        self._cold_tokens = 0
        self._prompt_tokens = 0
        self._scoring_broken = False

        logger.info("Shortest-prefill-first admission ACTIVE (window=%d)", self.WINDOW)

    def _prefix_warmth(self, request: Request) -> int:
        """Prompt tokens of `request` in the local KV cache or known to be in
        the external one."""
        kvm = self.kv_cache_manager
        if not kvm.enable_caching or request.get_prefix_cache_hit_limit() == 0:
            return 0
        if request.num_computed_tokens > 0:
            return request.num_computed_tokens
        max_hit = max(request.num_tokens - 1, 0)
        try:
            local = int(
                kvm.coordinator.find_longest_cache_hit(request.block_hashes, max_hit)[1]
            )
        except Exception:
            if not self._scoring_broken:
                self._scoring_broken = True
                logger.exception(
                    "Shortest-prefill-first: warmth scoring failed; "
                    "falling back to FCFS."
                )
            return 0
        external = None
        if self.connector is not None and not request.skip_reading_external_kv:
            external = self.connector.get_known_num_matched_tokens(request)
        return max(local, min(external or 0, max_hit))

    def _rank_and_warmth(
        self, request: Request
    ) -> tuple[tuple[int, float, float], int]:
        """Return (sort_key, warmth). Lower sort_key is admitted first."""
        warmth = self._prefix_warmth(request)
        cold_tokens = float(request.num_tokens - warmth)
        tier = _TIER_PREEMPTED if request.num_preemptions > 0 else _TIER_NORMAL
        return (tier, cold_tokens, request.arrival_time), warmth

    def promote(self, queue: FCFSRequestQueue) -> None:
        """Promote the request with the fewest uncached tokens within WINDOW."""
        if len(queue) < 2:
            return

        window = list(islice(queue, self.WINDOW))
        head = window[0]
        scored = [(self._rank_and_warmth(r), r) for r in window]
        (best_rank, best_warmth), best = min(scored, key=lambda t: t[0][0])

        if best is not head and best_rank < scored[0][0][0]:
            queue.remove(best)
            queue.prepend_request(best)

        self._n += 1
        self._cold_tokens += max(best.num_tokens - best_warmth, 0)
        self._prompt_tokens += best.num_tokens

        if self._n % _LOG_EVERY == 0:
            logger.info(
                "shortest-prefill-first: n=%d mean_cold_tokens=%.0f "
                "cold_frac=%.3f queue_depth=%d",
                self._n,
                self._cold_tokens / self._n,
                self._cold_tokens / max(self._prompt_tokens, 1),
                len(queue),
            )
