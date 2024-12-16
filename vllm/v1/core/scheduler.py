from collections import deque
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.base import PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.base import PlaceholderRange

logger = init_logger(__name__)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        # TODO: Support LoRA.
        assert lora_config is None, "V1 does not support LoRA yet."

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        num_gpu_blocks = cache_config.num_gpu_blocks
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            max_model_len=self.max_model_len,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)
        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: Dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: Set[str] = set()

        # OPTIMIZATION: Cache the RunningRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> RunningRequestData
        self.running_reqs_data: Dict[str, RunningRequestData] = {}

        # Encoder-related.
        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        # FIXME(woosuk): Below are placeholder values. We need to calculate the
        # actual values from the configurations.
        self.max_num_encoder_input_tokens = 16384
        # NOTE(woosuk): For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized and used, regardless of
        # the cache size. This is because the memory space for the encoder cache
        # is preallocated in the profiling run.
        self.encoder_cache_manager = EncoderCacheManager(cache_size=16384)

    def schedule(self) -> "SchedulerOutput":
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and num_tokens,
        # which is equal to len(prompt_token_ids) + len(output_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens. This is general enough to cover chunked prefills,
        # prefix caching, and the "jump decoding" optimization in the future.

        scheduled_new_reqs: List[Request] = []
        scheduled_resumed_reqs: List[Request] = []
        scheduled_running_reqs: List[Request] = []
        preempted_reqs: List[Request] = []

        req_to_new_block_ids: Dict[str, List[int]] = {}
        num_scheduled_tokens: Dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: Dict[str, List[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens

        # First, schedule the RUNNING requests.
        # NOTE(woosuk): At most 1 request in the RUNNING queue is allowed to be
        # in the "partial" state, where the request has some tokens computed
        # but not all. The constraint is due to the persistent batch in the
        # V1 model runner.
        # TODO(woosuk): Remove this constraint after refactoring model runner.
        has_partial_request = False
        req_index = 0
        while req_index < len(self.running):
            # Only the last request in the RUNNING queue can be "partial".
            assert not has_partial_request
            assert token_budget > 0
            request = self.running[req_index]
            num_new_tokens = request.num_tokens - request.num_computed_tokens
            num_new_tokens = min(num_new_tokens, token_budget)
            assert num_new_tokens > 0

            # Schedule encoder inputs.
            encoder_inputs_to_schedule, num_new_tokens, new_encoder_budget = (
                self._try_schedule_encoder_inputs(request,
                                                  request.num_computed_tokens,
                                                  num_new_tokens,
                                                  encoder_budget))
            assert num_new_tokens > 0

            while True:
                new_blocks = self.kv_cache_manager.append_slots(
                    request, num_new_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0

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
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1
            has_partial_request = (request.num_computed_tokens + num_new_tokens
                                   < request.num_tokens)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting:
                if has_partial_request:
                    break
                if len(self.running) == self.max_num_running_reqs:
                    break
                if token_budget == 0:
                    break

                request = self.waiting[0]
                # Get already-cached tokens.
                computed_blocks = self.kv_cache_manager.get_computed_blocks(
                    request)
                # NOTE(woosuk): Since incomplete blocks are not eligible for
                # sharing, `num_computed_tokens` is always a multiple of
                # `block_size`.
                num_computed_tokens = len(computed_blocks) * self.block_size
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if num_new_tokens == 0:
                    # The happens when prompt length is divisible by the block
                    # size and all blocks are cached. Now we force to recompute
                    # the last block. Note that we have to re-compute an entire
                    # block because allocate_slots() assumes num_computed_tokens
                    # is always a multiple of the block size. This limitation
                    # can potentially be removed in the future to slightly
                    # improve the performance.
                    num_computed_tokens -= self.block_size
                    num_new_tokens = self.block_size
                    computed_blocks.pop()
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, num_computed_tokens, num_new_tokens,
                     encoder_budget)
                if num_new_tokens == 0:
                    # The request cannot be scheduled.
                    break

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens, computed_blocks)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                self.waiting.popleft()
                self.running.append(request)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                has_partial_request = (num_computed_tokens + num_new_tokens <
                                       request.num_tokens)

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) == len(self.running))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id],
                                        req.num_computed_tokens)
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            ResumedRequestData.from_request(
                req, req_to_new_block_ids[req.request_id],
                req.num_computed_tokens) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_running_request_data(
                req, req_to_new_block_ids[req.request_id],
                req.num_computed_tokens) for req in scheduled_running_reqs
        ]
        preempted_req_ids = {req.request_id for req in preempted_reqs}
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_resumed_reqs=resumed_reqs_data,
            scheduled_running_reqs=running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            preempted_req_ids=preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
        )

        self.finished_req_ids = set()
        return scheduler_output

    def _make_running_request_data(
        self,
        request: Request,
        new_block_ids: List[int],
        num_computed_tokens: int,
    ) -> "RunningRequestData":
        # OPTIMIZATION: Cache the RunningRequestData objects to avoid creating
        # them at each scheduling step.
        if request.request_id in self.running_reqs_data:
            req_data = self.running_reqs_data[request.request_id]
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
        else:
            req_data = RunningRequestData.from_request(request, new_block_ids,
                                                       num_computed_tokens)
            self.running_reqs_data[request.request_id] = req_data
        return req_data

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> Tuple[List[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.
        """
        if not request.has_encoder_inputs():
            return [], num_new_tokens, encoder_budget

        encoder_inputs_to_schedule: List[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info["offset"]
            num_encoder_tokens = pos_info["length"]

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # The encoder input is already computed and cached.
                continue
            if not self.encoder_cache_manager.can_allocate(request, i):
                # The encoder cache is full. We can only schedule the decoder
                # tokens just before the encoder input.
                num_new_tokens = start_pos - num_computed_tokens
                break
            if num_encoder_tokens > encoder_budget:
                # The encoder budget is exhausted. We can only schedule the
                # decoder tokens up until the encoder input.
                # NOTE(woosuk): We assume that the encoder tokens should be
                # processed altogether, as the encoder usually uses
                # bidirectional attention.
                num_new_tokens = start_pos - num_computed_tokens
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> List[EngineCoreOutput]:
        # NOTE(woosuk): This method doesn't consider speculative decoding.
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        new_running: List[Request] = []
        engine_core_outputs: List[EngineCoreOutput] = []
        for request in self.running:
            req_id = request.request_id
            request.num_computed_tokens += num_scheduled_tokens[req_id]
            # When the request's num_computed_tokens catches up its num_tokens,
            # the request generates output tokens. Otherwise, we ignore the
            # sampler output for the request.
            assert request.num_computed_tokens <= request.num_tokens

            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            for input_id in list(cached_encoder_input_ids):
                start_pos = request.mm_positions[input_id]["offset"]
                num_tokens = request.mm_positions[input_id]["length"]
                if start_pos + num_tokens <= request.num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    self.encoder_cache_manager.free(request, input_id)

            if request.num_computed_tokens == request.num_tokens:
                req_index = model_runner_output.req_id_to_index[req_id]
                # NOTE(woosuk): Currently, we assume that each request
                # generates at most one token at each step.
                token_id = sampled_token_ids[req_index]
                request.append_output_token_ids(token_id)
                num_new_tokens = 1
                # TODO: Update the KV cache manager for prefix caching.

                # Check for stop and update request state.
                # This must be called before me make the EngineCoreOutput.
                stopped = self._check_stop(request)

                # Add EngineCoreOutput for this Request.
                output = EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=request.output_token_ids[-num_new_tokens:],
                    finished=request.is_finished(),
                    finish_reason=request.get_finished_reason(),
                    stop_reason=request.stop_reason)
                engine_core_outputs.append(output)

                # Breakout of the loop.
                if stopped:
                    continue

            new_running.append(request)
        self.running = new_running
        return engine_core_outputs

    def _check_stop(self, request: Request) -> bool:
        if (request.num_tokens >= self.max_model_len
                or request.num_output_tokens >= request.max_tokens):
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            self._free_request(request)
            return True

        sampling_params = request.sampling_params
        last_token_id = request.output_token_ids[-1]
        if (not sampling_params.ignore_eos
                and last_token_id == request.eos_token_id):
            request.status = RequestStatus.FINISHED_STOPPED
            self._free_request(request)
            return True

        if last_token_id in (sampling_params.stop_token_ids or ()):
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = last_token_id
            self._free_request(request)
            return True
        return False

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.requests[request.request_id] = request

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.running_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0


@dataclass
class NewRequestData:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List["MultiModalKwargs"]
    mm_positions: List["PlaceholderRange"]
    sampling_params: SamplingParams
    block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: List[int],
        num_computed_tokens: int,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt=request.prompt,
            mm_inputs=request.mm_inputs,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            block_ids=block_ids,
            num_computed_tokens=num_computed_tokens,
        )


@dataclass
class ResumedRequestData:

    req_id: str
    block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: List[int],
        num_computed_tokens: int,
    ) -> "ResumedRequestData":
        return cls(
            req_id=request.request_id,
            block_ids=block_ids,
            num_computed_tokens=num_computed_tokens,
        )


@dataclass
class RunningRequestData:

    req_id: str
    new_block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        new_block_ids: List[int],
        num_computed_tokens: int,
    ) -> "RunningRequestData":
        return cls(
            req_id=request.request_id,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )


@dataclass
class SchedulerOutput:

    scheduled_new_reqs: List[NewRequestData]
    scheduled_resumed_reqs: List[ResumedRequestData]
    scheduled_running_reqs: List[RunningRequestData]

    num_scheduled_tokens: Dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_encoder_inputs: Dict[str, List[int]]

    preempted_req_ids: Set[str]
    finished_req_ids: Set[str]
    free_encoder_input_ids: List[Tuple[str, int]]
