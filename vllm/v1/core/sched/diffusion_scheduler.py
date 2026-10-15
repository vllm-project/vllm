# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canvas-width handling and async read deferral for diffusion requests."""

from vllm.utils.diffusion import is_one_step_read
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.request import Request


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
        diffusion_config = self.vllm_config.diffusion_config
        self.single_pass_reads = bool(
            diffusion_config is not None and diffusion_config.single_pass_reads
        )

    def _final_prefill_spec_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        token_budget: int,
    ) -> list[int]:
        if (
            not self.single_pass_reads
            or not is_one_step_read(request.sampling_params)
            or num_computed_tokens >= request.num_prompt_tokens
            or num_computed_tokens + num_new_tokens != request.num_tokens
        ):
            return []
        width = diffusion_canvas_width(request, self.num_spec_tokens)
        if (
            num_new_tokens + width > token_budget
            or request.num_tokens + width >= self.max_model_len
        ):
            return []
        return [-1] * width

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
