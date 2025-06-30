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

    def update_before_output(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        super().update_before_output(scheduler_output)

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id in num_scheduled_tokens:
            request = self.requests[req_id]
            self._free_encoder_inputs(request)

            if request.num_computed_tokens == request.num_tokens:
                # Pre-allocate.
                request.append_output_token_ids(-1)

    def update_request(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        if new_token_ids:
            assert len(new_token_ids) == 1
            new_token_id = new_token_ids[0]
            request._output_token_ids[-1] = new_token_id
            request._all_token_ids[-1] = new_token_id

            # Now that the request has actual output tokens, we can cache the
            # blocks. As an optimization, we only cache the blocks if the
            # number of computed tokens is a multiple of the block size.
            if request.num_computed_tokens % self.block_size == 0:
                self.kv_cache_manager.cache_blocks(request,
                                                   request.num_computed_tokens)

        # NOTE(woosuk): Even if new_token_ids is empty, we still need to check
        # for stopping, because the request may be stopped by the length limit.
        stopped = check_stop(request, self.max_model_len)
        return new_token_ids, stopped
