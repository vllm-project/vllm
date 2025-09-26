# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)

DRAFT_TOKEN_PLACEHOLDER = -1

class AsyncScheduler(Scheduler):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(vllm_config=vllm_config,
                         kv_cache_config=kv_cache_config,
                         structured_output_manager=structured_output_manager,
                         mm_registry=mm_registry,
                         include_finished_set=include_finished_set,
                         log_stats = log_stats)
        self.speculative_config = vllm_config.speculative_config

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        super()._update_after_schedule(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            num_scheduled_propose_token = \
                len(scheduler_output.scheduled_spec_decode_tokens[req_id])
            if (request.num_computed_tokens == request.num_tokens +
                    num_scheduled_propose_token + 
                    request.num_output_placeholders):
                if num_scheduled_propose_token > 0:
                    request.spec_token_ids = \
                        [DRAFT_TOKEN_PLACEHOLDER] * \
                            self.speculative_config.num_speculative_tokens
                request.num_output_placeholders += \
                    1 + num_scheduled_propose_token

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
        proposed_token_ids: list[int] = None,
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids)

        # Update the number of output placeholders.
        if not self.speculative_config:
            request.num_output_placeholders -= len(new_token_ids)
        else:
            assert proposed_token_ids is not None

            spec_tokens = len(proposed_token_ids)
            assert 1 <= len(new_token_ids) and len(new_token_ids) <= 1 + \
                    spec_tokens
            request.num_output_placeholders -= 1 + spec_tokens
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders)
        return new_token_ids, stopped
