# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        # Dynamic SD: effective K for the current step, updated from
        # ModelRunnerOutput after each model execution.
        self._dynamic_num_spec_tokens: int | None = None

        # Dynamic SD async scheduling state:
        # optimal K learned from the latest model output, to be applied
        # at the beginning of the next schedule() call.
        self._pending_optimal_k: int | None = None
        # req_id -> committed spec token count from the most recent
        # _update_after_schedule. Used to correct accounting when
        # the optimal K changes between scheduling and output processing.
        self._in_flight_decode_req_k: dict[str, int] = {}

    def _update_placeholders_from_dynamic_sd(self, optimal_k: int | None) -> None:
        """Update placeholder count based on Dynamic SD decision."""
        if optimal_k is None:
            self._dynamic_num_spec_tokens = None
            self._spec_token_placeholders = [-1] * self.num_spec_tokens
        elif optimal_k != self._dynamic_num_spec_tokens:
            self._dynamic_num_spec_tokens = optimal_k
            self._spec_token_placeholders = [-1] * optimal_k

    def _apply_pending_dynamic_sd_update(self) -> None:
        """Apply a deferred dynamic SD K change at the start of schedule().

        When update_from_output() learns a new optimal K, it cannot
        immediately correct the in-flight step's accounting because we
        need to target the right set of requests (those committed in the
        most recent _update_after_schedule, tracked in
        _in_flight_decode_req_k). This method is called at the beginning
        of schedule() so that:
          1. The in-flight step's over/under-committed accounting is fixed.
          2. request.spec_token_ids is updated before the scheduling loop
             reads it.
          3. _spec_token_placeholders reflects the new K for the current
             scheduling step.
        """
        optimal_k = self._pending_optimal_k
        if optimal_k is None:
            return

        self._pending_optimal_k = None
        self._update_placeholders_from_dynamic_sd(optimal_k)

        for req_id, committed_k in self._in_flight_decode_req_k.items():
            diff = committed_k - optimal_k
            if diff <= 0:
                # K stayed the same or increased; the in-flight step
                # under-allocated (if anything) but we cannot retroactively
                # add more spec tokens to a step already on the GPU.
                # Just update spec_token_ids for the next scheduling step.
                request = self.requests.get(req_id)
                if request is not None and not request.is_finished():
                    request.spec_token_ids = self._spec_token_placeholders
                continue

            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue

            request.num_output_placeholders -= diff
            request.num_computed_tokens -= diff
            request.spec_token_ids = self._spec_token_placeholders

        self._in_flight_decode_req_k = {}

    def schedule(self) -> SchedulerOutput:
        self._apply_pending_dynamic_sd_update()
        return super().schedule()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        optimal_k = model_runner_output.optimal_num_speculative_tokens
        if optimal_k is not None:
            self._pending_optimal_k = optimal_k
        return super().update_from_output(scheduler_output, model_runner_output)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        self._in_flight_decode_req_k = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            if cur_num_spec_tokens > 0:
                self._in_flight_decode_req_k[req_id] = cur_num_spec_tokens

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

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
