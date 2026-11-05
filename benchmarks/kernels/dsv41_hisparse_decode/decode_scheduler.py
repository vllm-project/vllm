# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only prefill barrier. Never use this scheduler for online serving."""

import json
import time

from vllm.v1.core.sched.scheduler import Scheduler


class DecodeBarrierScheduler(Scheduler):
    """Hold after one decode until every member has released prefill workspace.

    The ordinary scheduler has no prefill/decode phase flag. Temporarily making
    completed prompts appear caught up gives them zero scheduled tokens while
    other prompts advance. Their real computed-token counts are restored before
    model execution; KV allocations and request state remain native.
    """

    def schedule(self, *args, **kwargs):
        requests = list(self.requests.values())
        sizes = {
            (r.sampling_params.extra_args or {}).get("benchmark_decode_cohort", 1)
            for r in requests
            if r.sampling_params is not None
        }
        assert len(sizes) <= 1, sizes
        count = next(iter(sizes), 1)
        ready = [r for r in self.running if r.num_computed_tokens > r.num_prompt_tokens]
        held = ready if len(ready) < count else []
        saved = [(r, r.num_computed_tokens) for r in held]
        if saved and getattr(self, "_barrier_started", None) is None:
            self._barrier_started = time.monotonic()
        if not saved:
            self._barrier_started = None
        elif time.monotonic() - self._barrier_started > 120:
            raise RuntimeError("Decode cohort cannot reach the prefill barrier")
        for request, _ in saved:
            request.num_computed_tokens = (
                request.num_tokens_with_spec + request.num_output_placeholders
            )
        try:
            output = super().schedule(*args, **kwargs)
            if output.total_num_scheduled_tokens:
                self._barrier_started = time.monotonic() if saved else None
            assert not set(output.num_scheduled_tokens).intersection(
                r.request_id for r, _ in saved
            )
            assert not output.preempted_req_ids, (
                "Cohort exceeded physical KV capacity",
                output.preempted_req_ids,
            )
        finally:
            for request, computed in saved:
                request.num_computed_tokens = computed
        if (
            count > 1
            and len(ready) == count
            and not getattr(self, "_released_ids", set()).issuperset(
                r.request_id for r in ready
            )
        ):
            self._released_ids = {r.request_id for r in ready}
            print(
                json.dumps(
                    dict(
                        decode_barrier_released=count,
                        computed_tokens=[r.num_computed_tokens for r in ready],
                    )
                ),
                flush=True,
            )
        return output
