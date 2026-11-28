# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canvas-width handling and async read deferral for diffusion requests."""

from dataclasses import replace

import torch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import DraftTokenIds, LogprobsTensors, ModelRunnerOutput
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


def _hold_prompt_scores(scores: dict, held: dict, emitting: set[str]) -> dict:
    scores = dict(scores)
    for req_id in scores.keys() - emitting:
        held[req_id] = scores.pop(req_id)
    for req_id in held.keys() & emitting:
        # A recomputed prefill after preemption supplies newer scores.
        scores.setdefault(req_id, held[req_id])
    return scores


class DiffusionScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Prompt scores from the token-free prefill, held until the first canvas.
        self._held_logprobs: dict[str, LogprobsTensors | None] = {}
        self._held_token_id_logprobs: dict[str, torch.Tensor] = {}

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        out = model_runner_output
        held_lp, held_tid = self._held_logprobs, self._held_token_id_logprobs
        if (
            held_lp
            or held_tid
            or out.prompt_logprobs_dict
            or out.prompt_token_id_logprobs_dict
        ):
            sampled = out.sampled_token_ids or ()
            emitting = {r for r, t in zip(out.req_ids, sampled) if t}
            out = replace(
                out,
                prompt_logprobs_dict=_hold_prompt_scores(
                    out.prompt_logprobs_dict, held_lp, emitting
                ),
                prompt_token_id_logprobs_dict=_hold_prompt_scores(
                    out.prompt_token_id_logprobs_dict, held_tid, emitting
                ),
            )
        engine_outputs = super().update_from_output(scheduler_output, out)
        for held in (held_lp, held_tid):
            for req_id in list(held):
                # Keep scores whose output the base scheduler skipped.
                if (r := self.requests.get(req_id)) is None or r.num_output_tokens:
                    del held[req_id]
        return engine_outputs

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
