# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Collection
from copy import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def native_decode_config(config: "VllmConfig") -> "VllmConfig":
    """Create target-only metadata without publishing unpopulated draft KV."""
    parallel = config.parallel_config
    cache = config.cache_config
    if config.compilation_config.cudagraph_mode.has_piecewise_cudagraphs():
        raise ValueError(
            "Native draft suspension requires eager execution or "
            "full decode graphs; piecewise graph caches are not isolated."
        )
    if (
        config.scheduler_config.async_scheduling
        or any(
            size != 1
            for size in (
                parallel.pipeline_parallel_size,
                parallel.data_parallel_size,
                parallel.prefill_context_parallel_size,
                parallel.decode_context_parallel_size,
            )
        )
        or parallel.enable_dbo
    ):
        raise ValueError(
            "Native draft suspension requires synchronous scheduling, "
            "PP=DP=PCP=DCP=1 and DBO disabled."
        )
    if cache.enable_prefix_caching or config.kv_transfer_config is not None:
        raise ValueError(
            "Native draft suspension requires prefix caching and KV transfer "
            "disabled: suspended draft blocks are not populated."
        )
    if (
        cache.mamba_cache_mode != "none"
        or cache.use_replayssm
        or cache.use_kda_recoverssm
    ):
        raise ValueError(
            "Native draft suspension requires mamba_cache_mode=none, "
            "ReplaySSM and RecoverSSM disabled."
        )
    native = copy(config)
    native.speculative_config = None
    return native


class NativeDecodeState:
    """Track irreversible draft suspension by request ID, not batch position.

    ``suspend(num_reqs, req_ids)`` (the speculator's ``suspend_drafting``)
    decides which batches start a suspension. A batch containing a suspended
    request suspends all its requests. A drafter cannot catch up missing hidden
    states after an OFF step. Finish, not filtering or preemption, ends
    suspension. Pending verification is drained, followed by one target-only
    step to consume its accepted recurrent-state offset before selecting
    native metadata/graphs. No draft work runs during either drain.
    """

    def __init__(self, suspend: Callable[[int, Collection[str]], bool]) -> None:
        self.suspend = suspend
        self.disabled: set[str] = set()
        self.pending_state: set[str] = set()

    def select(
        self,
        req_ids: Collection[str],
        finished_req_ids: Collection[str],
        drafted_req_ids: Collection[str],
    ) -> tuple[bool, bool]:
        """Return (skip drafting, use native target) for this scheduled batch."""
        self.disabled.difference_update(finished_req_ids)
        self.pending_state.difference_update(finished_req_ids)
        disable = bool(req_ids) and (
            self.suspend(len(req_ids), req_ids) or not self.disabled.isdisjoint(req_ids)
        )
        native = (
            disable and not drafted_req_ids and self.pending_state.isdisjoint(req_ids)
        )
        if disable:
            self.disabled.update(req_ids)
        self.pending_state.difference_update(req_ids)
        self.pending_state.update(drafted_req_ids)
        return disable, native
