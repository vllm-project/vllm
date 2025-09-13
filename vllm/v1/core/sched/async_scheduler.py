# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

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
            spec_tokens = len(spec_decode_tokens.get(req_id, []))
            if (request.num_computed_tokens == request.num_tokens +
                    request.num_output_placeholders + spec_tokens):
                # The request will generate a new token plus num spec_tokens
                # in this scheduling step.
                request.num_output_placeholders += (1 + spec_tokens)
                # Add a placeholder for the new token in spec_token_ids.
                # because the actual token id is not known yet. so just use -1
                # as a placeholder and the length of spec_token_ids is set to
                # self.num_spec_tokens. we will update the actual spec token id
                # in worker process.
                if self.num_spec_tokens > 0:
                    request.spec_token_ids = [-1] * self.num_spec_tokens

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids)

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders)
        return new_token_ids, stopped

    def _update_computed_tokens(
        self,
        request: Request,
        scheduled_spec_token_ids: list[int],
        generated_token_ids: list[int],
        spec_decoding_status: Optional[SpecDecodingStats],
    ):
        num_draft_tokens = len(scheduled_spec_token_ids)
        num_accepted = len(generated_token_ids) - 1
        num_rejected = num_draft_tokens - num_accepted
        # when enable spec decoding, the number of num_output_placeholders
        # is incrased by num_spec_tokens in _update_after_schedule.
        # update num_output_placeholders here to reflect the actual number
        # of accepted output tokens.
        request.num_output_placeholders -= num_rejected
        # num_computed_tokens represents the number of tokens
        # processed in the current step, considering scheduled
        # tokens and rejections. If some tokens are rejected,
        # num_computed_tokens is decreased by the number of rejected
        # tokens.
        request.num_computed_tokens -= num_rejected
        spec_decoding_stats = self.make_spec_decoding_stats(
            spec_decoding_status,
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted,
        )
        return spec_decoding_stats
