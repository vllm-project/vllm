# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_async = True

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        super()._update_after_schedule(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            if request.num_computed_tokens == request.num_tokens:
                # Pre-allocate the slot for output token ids.
                request.num_output_placeholder += 1

    def _update_request(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        if not new_token_ids:
            return new_token_ids, False

        # Replace the pre-allocated placeholder token with the actual token.
        request.num_output_placeholder -= len(new_token_ids)
        assert request.num_output_placeholder >= 0
        request.append_output_token_ids(new_token_ids)

        # Now that the request has actual output tokens, we can cache the
        # blocks. As an optimization, we only cache the blocks if the number
        # of computed tokens is a multiple of the block size.
        if request.num_computed_tokens % self.block_size == 0:
            self.kv_cache_manager.cache_blocks(request,
                                               request.num_computed_tokens)

        # NOTE: In async scheduling, the placeholder token should be ignored
        # when checking the stop condition.
        n = request.num_output_placeholder
        request.num_output_placeholder = 0
        stopped = check_stop(request, self.max_model_len)
        request.num_output_placeholder = n
        return new_token_ids, stopped
