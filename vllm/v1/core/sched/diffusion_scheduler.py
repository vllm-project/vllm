# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async scheduling for diffusion requests.

These rules live here so Scheduler.schedule() and AsyncScheduler stay
unchanged. VllmConfig selects this class for a diffusion model under async
scheduling. A sync scheduler creates no output placeholders, which both rules
read.
"""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
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


class DiffusionAsyncScheduler(AsyncScheduler):
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
