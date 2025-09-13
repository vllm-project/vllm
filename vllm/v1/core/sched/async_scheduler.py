# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, []))
            if (
                request.num_computed_tokens
                == request.num_tokens
                + request.num_output_placeholders
                + cur_num_spec_tokens
            ):
                # The request will generate a new token plus num_spec_tokens
                # in this scheduling step.
                request.num_output_placeholders += 1 + cur_num_spec_tokens
                # Add a placeholder for the new token in spec_token_ids.
                # because the actual token id is not known yet. so just use -1
                # as a placeholder and the length of spec_token_ids is set to
                # self.num_spec_tokens. we will update the actual spec token id
                # in worker process.
                request.spec_token_ids = [-1] * self.num_spec_tokens

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )
        # num_output_placeholders = 0 happend when a request is preempted.
        # a preempted request will be added to waiting queue again and
        # num_output_placeholders is reset to 0,
        # so don't need to revert num_output_placeholders for this situation.
        if request.num_output_placeholders > 0:
            # Update the number of output placeholders.
            request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped

    def _update_computed_tokens_after_speculation(
        self,
        request: Request,
        num_rejected: int,
    ):
        """Update the computed tokens for each request, which is necessary
        for spec decoding. In sync scheduler, we need to revert
        num_computed_tokens by num_rejected tokens,
        but in async scheduler, we also need to revert num_output_placeholders
        by num_rejected tokens for spec decoding.
        """
        # num_computed_tokens = 0 happend when a request is preempted.
        # a preempted request will be added to waiting queue again and
        # num_computed_tokens is reset to 0,
        # so don't need to revert num_computed_tokens for this situation.
        if request.num_computed_tokens > 0:
            # when spec decoding is enabled, num_output_placeholders
            # is increased by num_spec_tokens in _update_after_schedule.
            # update num_output_placeholders here to reflect the actual number
            # of accepted output tokens.
            request.num_output_placeholders -= num_rejected
        super()._update_computed_tokens_after_speculation(
            request,
            num_rejected,
        )
