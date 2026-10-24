# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A scheduler that leaves a running request out of exactly one step.

This is what happens when the token budget is used up by other requests: the
request stays RUNNING (it is not preempted), the V1 model runner drops it from
the persistent batch for that step and re-adds it on the next one.

A request opts in with ``SamplingParams(extra_args={"skip_step_at": n})``: it
is skipped once, on the first step after it has at least ``n`` output tokens
and its previous step accepted more than one token (spec decode), or right at
``n`` output tokens without spec decode. Needs
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so tests can read ``skipped``.
"""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

skipped: list[tuple[str, int]] = []


class _SkipStepOnceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_num_sampled: dict[str, int] = {}
        self._done: set[str] = set()

    @staticmethod
    def _skip_at(request) -> int | None:
        extra = request.sampling_params.extra_args if request.sampling_params else None
        return (extra or {}).get("skip_step_at")

    def _should_skip(self, request) -> bool:
        skip_at = self._skip_at(request)
        if skip_at is None or request.request_id in self._done:
            return False
        if request.num_output_tokens < skip_at:
            return False
        if self.vllm_config.speculative_config is None:
            return True
        return self._last_num_sampled.get(request.request_id, 0) > 1

    def update_from_output(self, scheduler_output, model_runner_output):
        sampled = model_runner_output.sampled_token_ids or []
        for i, req_id in enumerate(model_runner_output.req_ids or []):
            if i < len(sampled):
                self._last_num_sampled[req_id] = len(sampled[i])
        return super().update_from_output(scheduler_output, model_runner_output)

    def schedule(self, *args, **kwargs):
        held = [r for r in self.running if self._should_skip(r)]
        for r in held:
            self.running.remove(r)
            self._done.add(r.request_id)
            skipped.append((r.request_id, r.num_output_tokens))
        try:
            return super().schedule(*args, **kwargs)
        finally:
            for r in reversed(held):
                self.running.insert(0, r)


class SkipStepOnceScheduler(_SkipStepOnceMixin, Scheduler):
    pass


class SkipStepOnceAsyncScheduler(_SkipStepOnceMixin, AsyncScheduler):
    pass
