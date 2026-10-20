# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData, GrammarOutput, SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


@dataclass
class _PrefillBatch:
    requests: list[Request]
    first: SchedulerOutput
    continuation: SchedulerOutput
    group: int = 0


class LayeredPrefillScheduler(Scheduler):
    """Pipeline prefill batches through groups inside each PP rank.

    KV allocation and admission use the normal scheduler. While a batch is
    traversing the pipeline, its requests cannot be preempted or rescheduled.
    Eligible decode requests execute a full forward alongside prefill groups.
    In-flight decodes wait for their sampling results. Admit the next batch when
    the first rank is free, while earlier batches run on downstream ranks.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._num_groups = self.scheduler_config.num_layer_groups
        self._pending: deque[_PrefillBatch] = deque()
        self._stages: list[_PrefillBatch | None] = [
            None
        ] * self.parallel_config.pipeline_parallel_size
        self._prefill_req_ids: set[str] = set()

    @staticmethod
    def _is_prefilling(request: Request) -> bool:
        return (
            request.num_computed_tokens < request.num_prompt_tokens
            or request.num_computed_tokens < request.num_tokens - 1
        )

    def _update_after_schedule(self, output: SchedulerOutput) -> None:
        self._prefill_req_ids = {
            req_id
            for req_id in output.num_scheduled_tokens
            if self._is_prefilling(self.requests[req_id])
        }
        decode = self._select_requests(
            output, output.num_scheduled_tokens.keys() - self._prefill_req_ids
        )
        super()._update_after_schedule(decode)
        output.has_structured_output_requests = decode.has_structured_output_requests

    @staticmethod
    def _select_requests(output: SchedulerOutput, req_ids: set[str]) -> SchedulerOutput:
        cached = output.scheduled_cached_reqs
        indices = [i for i, req_id in enumerate(cached.req_ids) if req_id in req_ids]
        tokens = {
            req_id: count
            for req_id, count in output.num_scheduled_tokens.items()
            if req_id in req_ids
        }
        return replace(
            output,
            scheduled_new_reqs=[
                req for req in output.scheduled_new_reqs if req.req_id in req_ids
            ],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[cached.req_ids[i] for i in indices],
                resumed_req_ids=cached.resumed_req_ids & req_ids,
                new_token_ids=[cached.new_token_ids[i] for i in indices]
                if cached.new_token_ids
                else [],
                all_token_ids={
                    req_id: ids
                    for req_id, ids in cached.all_token_ids.items()
                    if req_id in req_ids
                },
                new_block_ids=[cached.new_block_ids[i] for i in indices],
                num_computed_tokens=[cached.num_computed_tokens[i] for i in indices],
                num_output_tokens=[cached.num_output_tokens[i] for i in indices],
            ),
            num_scheduled_tokens=tokens,
            total_num_scheduled_tokens=sum(tokens.values()),
        )

    def _enqueue_batch(self, output: SchedulerOutput) -> None:
        req_ids = list(output.num_scheduled_tokens)
        requests = [self.requests[req_id] for req_id in req_ids]
        continuation = CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in req_ids],
            all_token_ids={},
            new_block_ids=[None] * len(req_ids),
            num_computed_tokens=[req.num_computed_tokens for req in requests],
            num_output_tokens=[req.num_output_tokens for req in requests],
        )
        first = replace(
            output,
            finished_req_ids=set(),
            preempted_req_ids=set(),
            new_block_ids_to_zero=None,
            kv_cache_block_copies=None,
        )
        self._pending.append(
            _PrefillBatch(
                requests,
                first,
                replace(
                    first,
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=continuation,
                ),
            )
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        def keep_active(job: _PrefillBatch) -> bool:
            requests = [
                req
                for req in job.requests
                if self.requests.get(req.request_id) is req
                and req.status == RequestStatus.RUNNING
            ]
            if len(requests) != len(job.requests):
                job.requests = requests
                req_ids = {req.request_id for req in requests}
                job.first = self._select_requests(job.first, req_ids)
                job.continuation = self._select_requests(job.continuation, req_ids)
            return bool(requests)

        self._pending = deque(job for job in self._pending if keep_active(job))
        for rank, job in enumerate(self._stages):
            if job is not None and not keep_active(job):
                self._stages[rank] = None

        admit_prefills = not self._pending and self._stages[0] is None
        housekeeping = self._schedule_available(throttle_prefills, admit_prefills)
        if self._prefill_req_ids:
            self._enqueue_batch(
                self._select_requests(housekeeping, self._prefill_req_ids)
            )
        if not self._pending and not any(self._stages):
            return housekeeping
        decode = self._select_requests(
            housekeeping,
            housekeeping.num_scheduled_tokens.keys() - self._prefill_req_ids,
        )

        finished = housekeeping.finished_req_ids | self.finished_req_ids
        preempted = (
            housekeeping.preempted_req_ids or set()
        ) | self.reset_preempted_req_ids
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()
        # Every rank applies housekeeping before either forward, even if its
        # local prefill stage is idle. Do not zero/copy the same KV blocks twice.
        decode = replace(decode, new_block_ids_to_zero=None, kv_cache_block_copies=None)
        paused = self._pause_state == PauseState.PAUSED_ALL
        if not paused and self._stages[0] is None and self._pending:
            self._stages[0] = self._pending.popleft()

        stages = []
        for job in self._stages:
            output = SchedulerOutput.make_empty()
            if job is not None and not paused:
                output = replace(
                    job.first if job.group == 0 else job.continuation,
                    layer_group_idx=job.group,
                )
            stages.append(
                replace(
                    output,
                    finished_req_ids=finished,
                    preempted_req_ids=preempted,
                    new_block_ids_to_zero=housekeeping.new_block_ids_to_zero,
                    kv_cache_block_copies=housekeeping.kv_cache_block_copies,
                )
            )

        # Move only after constructing every rank's output: a transfer at the
        # end of this iteration becomes runnable in the next iteration.
        for rank in range(len(self._stages) - 1, -1, -1):
            job = self._stages[rank]
            if job is None or paused:
                continue
            job.group += 1
            if job.group == self._num_groups:
                self._stages[rank] = None
                job.group = 0
                if rank + 1 < len(self._stages):
                    assert self._stages[rank + 1] is None
                    self._stages[rank + 1] = job

        completed = stages[-1]
        if completed.layer_group_idx == self._num_groups - 1:
            super()._update_after_schedule(completed)
            completed = self._merge_completion(decode, completed)
        else:
            completed = decode
        scheduled = {
            req_id: num_tokens
            for stage in stages
            for req_id, num_tokens in stage.num_scheduled_tokens.items()
        }
        scheduled.update(decode.num_scheduled_tokens)
        return replace(
            housekeeping,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
            finished_req_ids=finished,
            preempted_req_ids=preempted,
            has_structured_output_requests=completed.has_structured_output_requests,
            layered_prefill_outputs=stages,
            layered_decode_output=decode,
            layered_completion_output=completed,
        )

    def _schedule_available(
        self, throttle_prefills: bool, admit_prefills: bool
    ) -> SchedulerOutput:
        """Allocate new work without rescheduling or preempting pinned batches."""
        running = self.running
        waiting, kv_waiting = self.waiting, self.kv_holding_waiting
        budget = self.max_num_scheduled_tokens
        running_limit = self.max_num_running_reqs
        active_limit = self.max_num_active_reqs
        jobs = [job for job in (*self._pending, *self._stages) if job is not None]
        pinned = {req.request_id for job in jobs for req in job.requests}
        self.running = [
            req
            for req in running
            if req.request_id not in pinned
            and (admit_prefills or not self._is_prefilling(req))
        ]
        self.running.sort(key=self._is_prefilling)
        hidden_running = len(running) - len(self.running)
        self.max_num_running_reqs = running_limit - hidden_running
        self.max_num_active_reqs = active_limit - hidden_running
        if admit_prefills:
            # Older batches can enter decode before this new batch drains.
            # Leave room for those tokens in every rank's retained budget.
            reserved_tokens = sum(
                req.num_computed_tokens + job.first.num_scheduled_tokens[req.request_id]
                >= req.num_tokens
                and req.num_output_tokens + 1 < req.max_tokens
                for job in jobs
                for req in job.requests
            )
            # Pending samples can make these decodes eligible before the new
            # prefill drains, even though they cannot run at admission time.
            reserved_tokens += sum(
                req.num_in_flight_tokens > 0
                and not self._is_prefilling(req)
                and req.num_output_tokens + 1 < req.max_tokens
                for req in self.running
            )
        else:
            self.waiting = create_request_queue(self.policy)
            self.kv_holding_waiting = create_request_queue(self.policy)
            # Ranks execute their own retained batch, not the union of all
            # pipeline batches. Decode must fit alongside each local batch.
            reserved_tokens = max(
                (job.first.total_num_scheduled_tokens for job in jobs), default=0
            )
        self.max_num_scheduled_tokens = max(0, budget - reserved_tokens)
        try:
            return super().schedule(throttle_prefills)
        finally:
            if not admit_prefills:
                waiting.prepend_requests(reversed(list(self.waiting)))
            self.waiting, self.kv_holding_waiting = waiting, kv_waiting
            admitted = self.running
            original_ids = {req.request_id for req in running}
            self.running = [
                req for req in running if req.status == RequestStatus.RUNNING
            ] + [req for req in admitted if req.request_id not in original_ids]
            self.max_num_scheduled_tokens = budget
            self.max_num_running_reqs = running_limit
            self.max_num_active_reqs = active_limit

    @staticmethod
    def _merge_completion(
        decode: SchedulerOutput, prefill: SchedulerOutput
    ) -> SchedulerOutput:
        cached_decode = decode.scheduled_cached_reqs
        cached_prefill = prefill.scheduled_cached_reqs
        cached = CachedRequestData(
            req_ids=cached_decode.req_ids + cached_prefill.req_ids,
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None]
            * (len(cached_decode.req_ids) + len(cached_prefill.req_ids)),
            num_computed_tokens=(
                cached_decode.num_computed_tokens + cached_prefill.num_computed_tokens
            ),
            num_output_tokens=(
                cached_decode.num_output_tokens + cached_prefill.num_output_tokens
            ),
        )
        return replace(
            decode,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=decode.num_scheduled_tokens
            | prefill.num_scheduled_tokens,
            total_num_scheduled_tokens=(
                decode.total_num_scheduled_tokens + prefill.total_num_scheduled_tokens
            ),
            has_structured_output_requests=(
                decode.has_structured_output_requests
                or prefill.has_structured_output_requests
            ),
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        if scheduler_output.layered_completion_output is not None:
            scheduler_output = scheduler_output.layered_completion_output
        return super().get_grammar_bitmask(scheduler_output)

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]:
        if scheduler_output.layered_completion_output is not None:
            scheduler_output = scheduler_output.layered_completion_output
        for req_id in scheduler_output.num_scheduled_tokens:
            if (request := self.requests.get(req_id)) is not None:
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
        return super().update_from_output(scheduler_output, model_runner_output)
