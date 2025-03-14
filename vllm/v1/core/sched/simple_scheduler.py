# SPDX-License-Identifier: Apache-2.0
import time
from collections import deque
from collections.abc import Iterable
from typing import Optional, Union

from vllm.config import (CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.common import CommonSchedulerStates
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.logging import (record_preempted, record_queued,
                                        record_scheduled)
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


class SimpleScheduler(SchedulerInterface):
    """A simple scheduler.

    Mixed prefill & decode: X
    Chunked prefills: X
    Padding-aware scheduling: X (TODO)
    Prefix caching: O (if `enable_prefix_caching`)
    Cascade attention: O (if `enable_prefix_caching`)
    Speculative decoding: X
    Structured outputs: X (TODO)
    LoRA: X (TODO)
    Vision language models: X (TODO)
    Pipeline parallelism: X
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig],
        log_stats: bool,
        structured_output_manager: StructuredOutputManager,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.speculative_config = speculative_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        assert self.lora_config is None
        assert self.speculative_config is None

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        if self.max_model_len > self.max_num_scheduled_tokens:
            raise ValueError(
                "SimpleScheduler requires `max_model_len` "
                f"({self.max_model_len}) to be <= "
                f"`max_num_batched_tokens` ({self.max_num_scheduled_tokens})")

        num_gpu_blocks = cache_config.num_gpu_blocks
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            max_model_len=self.max_model_len,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
            log_stats=self.log_stats)
        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        # The requests that have been scheduled and are being executed
        # by the executor.
        self.scheduled_req_ids: set[str] = set()

        # Misc states for the scheduler.
        self.states = CommonSchedulerStates()

    def schedule(self) -> SchedulerOutput:
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule prefill requests.
        while self.waiting and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs:
                break
            request = self.waiting[0]

            # Get already-cached tokens.
            computed_blocks, num_computed_tokens = \
                self.kv_cache_manager.get_computed_blocks(request)
            # Number of tokens to be scheduled.
            # We use `request.num_tokens` instead of
            # `request.num_prompt_tokens` to consider the resumed requests,
            # which have output tokens.
            num_new_tokens = request.num_tokens - num_computed_tokens
            if num_new_tokens == 0:
                # This happens when prompt length is divisible by the block
                # size and all blocks are cached. Now we force to recompute
                # the last block. Note that we have to re-compute an entire
                # block because allocate_slots() assumes num_computed_tokens
                # is always a multiple of the block size. This limitation
                # can potentially be removed in the future to slightly
                # improve the performance.
                num_computed_tokens -= self.block_size
                num_new_tokens = self.block_size
                computed_blocks.pop()
            if num_new_tokens > token_budget:
                # NOTE(woosuk): Since the scheduler doesn't do chunked prefills,
                # the request cannot be scheduled.
                break
            assert num_new_tokens > 0

            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, computed_blocks)
            if new_blocks is None:
                # The request cannot be scheduled.
                break

            # Schedule the request.
            self.waiting.popleft()
            assert not request.use_structured_output, \
                "Structured outputs are not supported by the simple scheduler."
            self.running.append(request)
            self.scheduled_req_ids.add(request.request_id)
            if self.log_stats:
                record_scheduled(request, scheduled_timestamp)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in computed_blocks + new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            # NOTE(woosuk): Check that the request is "fully" scheduled.
            assert num_computed_tokens + num_new_tokens == request.num_tokens

        # If no prefill was scheduled, schedule decode requests.
        num_prefill_reqs = (len(scheduled_new_reqs) +
                            len(scheduled_resumed_reqs))
        if num_prefill_reqs == 0:
            req_index = 0
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                num_new_tokens = (request.num_tokens -
                                  request.num_computed_tokens)
                # NOTE(woosuk): This is decode, so num_new_tokens should be 1.
                assert num_new_tokens == 1

                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request, num_new_tokens)
                    if new_blocks is None:
                        # The request cannot be scheduled.
                        # Preempt the lowest-priority request.
                        preempted_req = self.running.pop()
                        self.kv_cache_manager.free(preempted_req)
                        preempted_req.status = RequestStatus.PREEMPTED
                        preempted_req.num_computed_tokens = 0
                        if self.log_stats:
                            record_preempted(preempted_req,
                                             scheduled_timestamp)

                        self.waiting.appendleft(preempted_req)
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            # No more request to preempt.
                            can_schedule = False
                            break
                    else:
                        # The request can be scheduled.
                        can_schedule = True
                        break
                if not can_schedule:
                    break
                assert new_blocks is not None

                # Schedule the request.
                scheduled_running_reqs.append(request)
                self.scheduled_req_ids.add(request.request_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                req_index += 1

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # NOTE(woosuk): The scheduler should only schedule either prefill or
        # decode, not both.
        num_decode_reqs = len(scheduled_running_reqs)
        assert not (num_prefill_reqs > 0 and num_decode_reqs > 0)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self.states.make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                0,  # num_scheduled_spec_tokens
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self.states.make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                0,  # num_scheduled_spec_tokens
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.states.finished_req_ids,
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        self.states.finished_req_ids = set()
        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        new_running: list[Request] = []
        outputs: list[EngineCoreOutput] = []

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]
            request.num_computed_tokens += num_tokens_scheduled
            assert request.num_computed_tokens == request.num_tokens

            stopped = False
            new_logprobs = None
            new_token_ids: list[int] = []
            for output_token_id in generated_token_ids:
                request.append_output_token_ids(output_token_id)
                new_token_ids.append(output_token_id)

                # Check for stop and update request state.
                # This must be called before we make the EngineCoreOutput.
                stopped = check_stop(request, self.max_model_len)
                if stopped:
                    self._free_request(request)
                    break

            # Extract sample logprobs if needed.
            if request.sampling_params.logprobs is not None:
                assert logprobs is not None
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or prompt_logprobs_tensors is not None:
                # Add EngineCoreOutput for this Request.
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events()))

            self.scheduled_req_ids.remove(request.request_id)
            if not stopped:
                new_running.append(request)

        self.running = new_running
        return EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(),
        )

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            record_queued(request)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
                self.scheduled_req_ids.discard(request.request_id)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        self.states.free_request(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.states.finished_req_ids) > 0

    def get_num_unscheduled_requests(self) -> int:
        return self.get_num_unfinished_requests() - len(self.scheduled_req_ids)

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(self) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            gpu_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=self.kv_cache_manager.make_prefix_cache_stats(),
        )
