# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        super()._update_after_schedule(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if (
                request.num_computed_tokens
                == request.num_tokens + request.num_output_placeholders
            ):
                # The request will generate a new token in this scheduling step.
                # TODO(woosuk): Support speculative decoding.
                request.num_output_placeholders += 1

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        cached_request_data = super()._make_cached_request_data(
            running_reqs,
            resumed_reqs,
            num_scheduled_tokens,
            spec_decode_tokens,
            req_to_new_blocks,
        )
        if resumed_reqs:
            # Include all output token ids for any resumed requests
            # since these aren't updated in the model runner in the
            # async scheduling case.
            cached_request_data.new_token_ids = [None] * len(running_reqs)
            cached_request_data.new_token_ids.extend(
                list(req.output_token_ids) for req in resumed_reqs
            )
        return cached_request_data
