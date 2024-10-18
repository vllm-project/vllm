from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.multimodal import MultiModalDataDict
from vllm.sampling_params import SamplingParams
from vllm_v1.core.kv_cache_manager import KVCacheManager
from vllm_v1.outputs import ModelRunnerOutput
from vllm_v1.request import Request, RequestStatus

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
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

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

        # Priority queues for requests.
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

        self.finished_req_ids: Set[str] = set()
        self.aborted_req_ids: Set[str] = set()

    def schedule(self) -> "SchedulerOutput":
        scheduled_new_reqs: List[Request] = []
        scheduled_resumed_reqs: List[Request] = []
        scheduled_running_reqs: List[Request] = []
        preempted_reqs: List[Request] = []

        req_to_new_block_ids: Dict[str, List[int]] = {}
        num_scheduled_tokens: Dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running):
            if token_budget == 0:
                break

            request = self.running[req_index]
            num_tokens = request.num_tokens - request.num_computed_tokens
            num_tokens = min(num_tokens, token_budget)
            assert num_tokens > 0

            while True:
                new_block_ids = self.kv_cache_manager.append_slots(
                    request, num_tokens)
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
                    num_scheduled_tokens[request.request_id] = num_tokens
                    token_budget -= num_tokens
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
                num_tokens = request.num_tokens - num_computed_tokens
                num_tokens = min(num_tokens, token_budget)
                assert num_tokens > 0
                new_block_ids = self.kv_cache_manager.allocate_slots(
                    request, num_tokens, computed_block_ids)
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
                num_scheduled_tokens[request.request_id] = num_tokens
                token_budget -= num_tokens
                request.status = RequestStatus.RUNNING

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) == len(self.running))

        # Construct the scheduler output.
        # NOTE(woosuk): When the batch size is large, creating these objects
        # can be expensive. We may need to optimize this.
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
            RunningRequestData.from_request(
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
            finished_req_ids=self.finished_req_ids,
            aborted_req_ids=self.aborted_req_ids,
        )

        self.finished_req_ids = set()
        self.aborted_req_ids = set()
        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> List[Request]:
        sampled_token_ids = model_runner_output.sampled_token_ids_cpu.tolist()
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        new_running: List[Request] = []
        finished_reqs: List[Request] = []
        for request in self.running:
            req_id = request.request_id
            # TODO: Consider speculative decoding.
            request.num_computed_tokens += num_scheduled_tokens[req_id]
            if request.num_computed_tokens >= request.num_prompt_tokens:
                req_index = model_runner_output.req_id_to_index[req_id]
                token_id = sampled_token_ids[req_index]
                request.output_token_ids.append(token_id)
                # TODO: Update the KV cache manager for prefix caching.

                if (request.num_tokens >= self.max_model_len
                        or request.num_output_tokens >= request.max_tokens):
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                    self.finished_req_ids.add(req_id)
                    finished_reqs.append(request)
                    self._free_request(request)
                    continue
            new_running.append(request)
        self.running = new_running
        return finished_reqs

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def abort_requests(self, request_ids: Union[str, Iterable[str]]) -> None:
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        request_ids = set(request_ids)

        # TODO: Optimize this.
        for queue in [self.waiting, self.running]:
            aborted_reqs: List[Request] = []
            for request in queue:
                if not request_ids:
                    break
                if request.request_id in request_ids:
                    request.status = RequestStatus.FINISHED_ABORTED
                    aborted_reqs.append(request)
                    request_ids.remove(request.request_id)

            for request in aborted_reqs:
                queue.remove(request)
                self.aborted_req_ids.add(request.request_id)
                self._free_request(request)

    def stop_requests(self, request_ids: Union[str, Iterable[str]]) -> None:
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        request_ids = set(request_ids)

        # TODO: Optimize this.
        for queue in [self.waiting, self.running]:
            stopped_reqs: List[Request] = []
            for request in queue:
                if not request_ids:
                    break
                if request.request_id in request_ids:
                    request.status = RequestStatus.FINISHED_STOPPED
                    stopped_reqs.append(request)
                    request_ids.remove(request.request_id)

            for request in stopped_reqs:
                queue.remove(request)
                self.finished_req_ids.add(request.request_id)
                self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.kv_cache_manager.free(request)

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
    aborted_req_ids: Set[str]
