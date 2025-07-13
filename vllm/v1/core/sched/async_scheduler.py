# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import check_stop
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
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            if (request.num_computed_tokens == request.num_tokens +
                    request.num_output_placeholders):
                # The request will generate a new token in this scheduling step.
                # TODO(woosuk): Support speculative decoding.
                request.num_output_placeholders += 1

    def _update_request(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        assert not request.is_finished()
        if not new_token_ids:
            return new_token_ids, False

        status_before_update = request.status
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            # Decrease the number of output placeholders and append the token.
            request.num_output_placeholders -= 1
            assert request.num_output_placeholders >= 0
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                break
        if stopped:
            new_token_ids = new_token_ids[:num_new]

        # Now that the request has actual output tokens, we can cache the
        # blocks. NOTE(woosuk): Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders)

        return new_token_ids, stopped
