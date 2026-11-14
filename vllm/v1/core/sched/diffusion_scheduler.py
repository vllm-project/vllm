# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canvas-width handling and async read deferral for diffusion requests."""

from collections.abc import Iterable

import torch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import DraftTokenIds, LogprobsTensors, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


def diffusion_canvas_width(request: Request, canvas_length: int) -> int:
    """The canvas width a diffusion request asked for, else the served one."""
    params = request.sampling_params
    extra = params.extra_args if params is not None else None
    width = extra.get("diffusion_canvas_length") if extra else None
    return int(width) if width else canvas_length


def _read_in_flight(request: Request, width: int) -> bool:
    """True when a read-only request has all its denoise steps in flight.

    The request emits its canvas on the last step and ends, so a further
    step is discarded.
    """
    params = request.sampling_params
    extra = params.extra_args if params is not None else None
    if not extra or not extra.get("diffusion_read_only"):
        return False
    steps = extra.get("diffusion_max_steps")
    if not steps:
        return False
    # Each in-flight denoise step holds one canvas of placeholders.
    return request.num_output_placeholders >= steps * width


class DiffusionScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pending_prompt_logprobs: dict[str, LogprobsTensors] = {}
        self._pending_prompt_token_id_logprobs: dict[str, torch.Tensor] = {}

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        if not (
            self._pending_prompt_logprobs
            or self._pending_prompt_token_id_logprobs
            or model_runner_output.prompt_logprobs_dict
            or model_runner_output.prompt_token_id_logprobs_dict
        ):
            return super().update_from_output(scheduler_output, model_runner_output)

        # Diffusion prefills score the prompt without emitting tokens.
        for req_id, scores in model_runner_output.prompt_logprobs_dict.items():
            request = self.requests.get(req_id)
            if (
                scores is not None
                and request is not None
                and not request.is_finished()
                and not (
                    request.drop_stale_output and request.num_stale_output_tokens > 0
                )
            ):
                self._pending_prompt_logprobs[req_id] = scores
        for req_id, scores in model_runner_output.prompt_token_id_logprobs_dict.items():
            request = self.requests.get(req_id)
            if (
                request is not None
                and not request.is_finished()
                and not (
                    request.drop_stale_output and request.num_stale_output_tokens > 0
                )
            ):
                self._pending_prompt_token_id_logprobs[req_id] = scores

        model_runner_output.prompt_logprobs_dict = {}
        model_runner_output.prompt_token_id_logprobs_dict = {}
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for client_outputs in outputs.values():
            for output in client_outputs.outputs:
                output.new_prompt_logprobs_tensors = self._pending_prompt_logprobs.pop(
                    output.request_id, None
                )
                output.prompt_token_id_logprobs = (
                    self._pending_prompt_token_id_logprobs.pop(output.request_id, None)
                )
        return outputs

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[Request]:
        finished = super().finish_requests(request_ids, finished_status)
        for request in finished:
            self._pending_prompt_logprobs.pop(request.request_id, None)
            self._pending_prompt_token_id_logprobs.pop(request.request_id, None)
        return finished

    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        self._pending_prompt_logprobs.pop(request.request_id, None)
        self._pending_prompt_token_id_logprobs.pop(request.request_id, None)
        super()._preempt_request(request, timestamp, drop_stale_output)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for i, (req_id, token_ids) in enumerate(
            zip(draft_token_ids.req_ids, draft_token_ids.draft_token_ids)
        ):
            request = self.requests.get(req_id)
            if request is None:
                continue
            width = diffusion_canvas_width(request, self.num_spec_tokens)
            # Remove padded positions before the base scheduler validates grammar.
            if len(token_ids) > width:
                draft_token_ids.draft_token_ids[i] = token_ids[:width]
        super().update_draft_token_ids(draft_token_ids)


class DiffusionAsyncScheduler(DiffusionScheduler, AsyncScheduler):
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        for request in self.running:
            width = diffusion_canvas_width(request, self.num_spec_tokens)
            # The placeholders are as wide as the served canvas.
            if len(request.spec_token_ids) > width:
                request.spec_token_ids = request.spec_token_ids[:width]
            if _read_in_flight(request, width):
                # Scheduler.schedule()'s max_tokens guard cannot be reached
                # from a subclass. schedule() advances current_step before it
                # checks decode eligibility, so current_step + 2 skips this
                # call alone. max keeps a longer pipeline-parallel wait.
                request.next_decode_eligible_step = max(
                    request.next_decode_eligible_step, self.current_step + 2
                )
        return super().schedule(throttle_prefills)
