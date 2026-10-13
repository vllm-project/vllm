# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the simulated-execution unit tests."""

from vllm.v1.worker.simulated.timing import (
    SimRequestStep,
    StepDescriptor,
)


def sim_req(req_id: str, scheduled: int, computed: int, prompt: int) -> SimRequestStep:
    return SimRequestStep(
        req_id=req_id,
        num_scheduled_tokens=scheduled,
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_cached_tokens=0,
        num_spec_tokens=0,
        is_prefill=computed < prompt,
    )


def sim_step(index: int, *reqs: SimRequestStep) -> StepDescriptor:
    prefill = [r for r in reqs if r.is_prefill]
    decode = [r for r in reqs if not r.is_prefill]
    return StepDescriptor(
        step_index=index,
        requests=tuple(reqs),
        num_prefill_tokens=sum(r.num_scheduled_tokens for r in prefill),
        num_decode_tokens=sum(r.num_scheduled_tokens for r in decode),
        num_prefill_reqs=len(prefill),
        num_decode_reqs=len(decode),
        num_common_prefix_blocks=0,
        num_kv_tokens=sum(r.num_computed_tokens + r.num_scheduled_tokens for r in reqs),
    )


class StubConfig:
    """Providers only receive the config object; built-ins ignore it."""
