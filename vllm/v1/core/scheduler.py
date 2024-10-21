from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.multimodal import MultiModalDataDict
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

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

        num_gpu_blocks = cache_config.num_gpu_blocks
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        # Create the block space manager.
        self.kv_cache_manager = KVCacheManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=True)
        self.block_size = self.cache_config.block_size

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

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

    def schedule(self) -> "SchedulerOutput":
        scheduled_new_reqs: List[Request] = []
        scheduled_resumed_reqs: List[Request] = []
        scheduled_running_reqs: List[Request] = []
        preempted_reqs: List[Request] = []

        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and num_tokens,
        # which is equal to len(prompt_token_ids) + len(output_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens. This is general enough to cover chunked prefills,
        # prefix caching, and the "jump forward" optimization in the future.

        req_to_new_block_ids: Dict[str, List[int]] = {}
        num_scheduled_tokens: Dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running):
            if token_budget == 0:
                break

            request = self.running[req_index]
            num_new_tokens = request.num_tokens - request.num_computed_tokens
            num_new_tokens = min(num_new_tokens, token_budget)
            assert num_new_tokens > 0

            while True:
                new_block_ids = self.kv_cache_manager.append_slots(
                    request, num_new_tokens)
                if new_block_ids is None:
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
                        break
                else:
                    # The request can be scheduled.
                    scheduled_running_reqs.append(request)

                    req_to_new_block_ids[request.request_id] = new_block_ids
                    num_scheduled_tokens[request.request_id] = num_new_tokens
                    token_budget -= num_new_tokens
                    req_index += 1
                    break

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting:
                if len(self.running) == self.max_num_running_reqs:
                    break
                if token_budget == 0:
                    break

                request = self.waiting[0]
                # Get already-cached tokens.
                computed_block_ids = self.kv_cache_manager.get_computed_blocks(
                    request)
                # NOTE(woosuk): Since incomplete blocks are not eligible for
                # sharing, `num_computed_tokens` is always a multiple of
                # `block_size`.
                num_computed_tokens = len(computed_block_ids) * self.block_size
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0
                new_block_ids = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens, computed_block_ids)
                if new_block_ids is None:
                    # The request cannot be scheduled.
                    break
                request.num_computed_tokens = num_computed_tokens

                self.waiting.popleft()
                self.running.append(request)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                req_to_new_block_ids[request.request_id] = (
                    computed_block_ids + new_block_ids)
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING

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
            preempted_req_ids=preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
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

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> List[Tuple[Request, int]]:
        # NOTE(woosuk): This method doesn't consider speculative decoding.
        sampled_token_ids = model_runner_output.sampled_token_ids_cpu.tolist()
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        new_running: List[Request] = []
        # (request, num_sampled_tokens)
        sampled: List[Tuple[Request, int]] = []
        for request in self.running:
            req_id = request.request_id
            request.num_computed_tokens += num_scheduled_tokens[req_id]
            # When the request's num_computed_tokens catches up its num_tokens,
            # the request generates output tokens. Otherwise, we ignore the
            # sampler output for the request.
            assert request.num_computed_tokens <= request.num_tokens
            if request.num_computed_tokens == request.num_tokens:
                req_index = model_runner_output.req_id_to_index[req_id]
                # NOTE(woosuk): Currently, we assume that each request
                # generates at most one token at each step.
                token_id = sampled_token_ids[req_index]
                request.output_token_ids.append(token_id)
                sampled.append((request, 1))
                # TODO: Update the KV cache manager for prefix caching.

                # Check if the request is finished.
                stopped = self._check_stop(request)
                if stopped:
                    continue

            new_running.append(request)
        self.running = new_running
        return sampled

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
    multi_modal_data: Optional[MultiModalDataDict]
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
            prompt_token_ids=request.inputs["prompt_token_ids"],
            prompt=request.inputs.get("prompt"),
            multi_modal_data=request.inputs.get("multi_modal_data"),
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

    preempted_req_ids: Set[str]
    finished_req_ids: Set[str]
