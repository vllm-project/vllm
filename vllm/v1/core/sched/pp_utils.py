# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import weakref
from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler


class PPMicrobatchCap:
    """Caps each PP step's token budget at the uncached prefill backlog /
    pp_size (at least `min_tokens`) plus running decode tokens, so a light
    prefill load is split into chunks that fill every pipeline stage."""

    def __init__(self, scheduler: "Scheduler", pp_size: int, min_tokens: int):
        self.scheduler = scheduler
        self.pp_size = pp_size
        self.min_tokens = min_tokens
        # (num_tokens, prefix-cache hit, probed cached prefix) of waiting requests.
        self._cached_tokens = weakref.WeakKeyDictionary[Request, tuple[int, int, int]]()

    @classmethod
    def create(cls, scheduler: "Scheduler", pp_size: int) -> "PPMicrobatchCap | None":
        """None without PP, when disabled, or when prefills can't be freely
        chunked."""
        config = scheduler.scheduler_config
        if (
            pp_size == 1
            or not config.pp_microbatch_min_tokens
            or not config.enable_chunked_prefill
            or config.disable_chunked_mm_input
            or scheduler.need_mamba_block_aligned_split
        ):
            return None
        return cls(scheduler, pp_size, config.pp_microbatch_min_tokens)

    def apply(self, budget: int) -> int:
        scheduler = self.scheduler
        waiting_queues = (
            scheduler.kv_holding_waiting,
            scheduler.waiting,
            scheduler.async_kv_waiting,
        )
        # Parked async KV loads are counted with the waiting requests.
        prefill = sum(
            max(0, request.num_tokens - request.num_computed_tokens)
            for request in scheduler._inflight_prefills
            if request.status == RequestStatus.RUNNING
        )
        if not prefill and not any(waiting_queues):
            return budget  # No prefill to split.
        decode = sum(
            1 + len(request.spec_token_ids)
            for request in scheduler.running
            if request.num_computed_tokens >= request.num_prompt_tokens
            and not scheduler._needs_no_more_steps(request)
        )
        for request in itertools.chain(*waiting_queues):
            if prefill >= budget * self.pp_size:
                break  # cap already at the full budget
            hit = self._num_cached_tokens(request)
            prefill += max(0, request.num_tokens - hit)
        # num_prefill_lookahead: the MTP lookahead reservation must leave a token.
        cap = max(
            self.min_tokens,
            scheduler.num_prefill_lookahead,
            cdiv(prefill, self.pp_size),
        )
        return min(budget, cap + decode)

    def _num_cached_tokens(self, request: Request) -> int:
        """`get_num_cached_tokens`, redone only when num_tokens changes
        (preemption) or more of the request's prefix has been cached since."""
        kv_cache_manager = self.scheduler.kv_cache_manager
        cached = self._cached_tokens.get(request)
        if cached is not None and cached[0] == request.num_tokens:
            _, num_cached, prefix = cached
            if kv_cache_manager.extend_cached_prefix(request, prefix) == prefix:
                return num_cached
        num_cached = kv_cache_manager.get_num_cached_tokens(request)
        prefix = kv_cache_manager.extend_cached_prefix(request, num_cached)
        self._cached_tokens[request] = (request.num_tokens, num_cached, prefix)
        return num_cached
