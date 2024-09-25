import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (Callable, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.utils import Device

from vllm.request import Request, RequestStatus
from vllm.sampling_params import SamplingParams
from vllm.multimodal import MultiModalDataDict

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

        version = "v1"
        if self.scheduler_config.use_v2_block_manager:
            version = "v2"
        if self.scheduler_config.embedding_mode:
            version = "embedding"
        BlockSpaceManagerImpl = \
            BlockSpaceManager.get_block_space_manager_class(version)
        num_gpu_blocks = cache_config.num_gpu_blocks
        num_cpu_blocks = cache_config.num_cpu_blocks

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)
        self.block_size = self.cache_config.block_size

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        # Priority queues for requests.
        self.waiting: Deque[Request] = deque()
        self.running: Deque[Request] = deque()

        self.finished_req_ids: Set[str] = set()
        self.aborted_req_ids: Set[str] = set()

    def schedule(self) -> "SchedulerOutput":
        # Finish the requests that have reached the maximum length.
        self._check_stop_by_len()

        scheduled_new_reqs: List[Request] = []
        scheduled_resumed_reqs: List[Request] = []
        scheduled_running_reqs: List[Request] = []
        preempted_reqs: List[Request] = []

        req_to_new_block_ids: Dict[str, List[int]] = {}
        num_scheduled_tokens: Dict[str, int] = {}
        total_num_scheduled_tokens = 0
        num_remaining_tokens = self.max_num_scheduled_tokens

        # First, schedule the RUNNING requests.
        while self.running:
            if num_remaining_tokens == 0:
                break

            request = self.running[0]
            num_tokens = request.num_tokens - request.num_computed_tokens
            num_tokens = min(num_tokens, num_remaining_tokens)

            new_block_ids: List[int] = []
            while not self.block_manager.can_append_slots(request, num_tokens):
                new_block_ids = self.block_manager.append_slots(
                    request, num_tokens)
                if not new_block_ids:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    preempted_req = self.running.pop()
                    self.block_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0

                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)

                    if preempted_req == request:
                        break
            else:
                # The request can be scheduled.
                self.running.popleft()
                scheduled_running_reqs.append(request)

                req_to_new_block_ids[request.request_id] = new_block_ids
                num_scheduled_tokens[request.request_id] = num_tokens
                total_num_scheduled_tokens += num_tokens
                num_remaining_tokens -= num_tokens

                request.status = RequestStatus.RUNNING
                request.num_computed_tokens += num_tokens
                if request.num_tokens == request.num_computed_tokens:
                    # TODO(woosuk): Consider speculative decoding.
                    request.num_output_tokens += 1

        # Next, schedule the WAITING requests.
        while self.waiting:
            if preempted_reqs:
                break
            if len(self.running) == self.max_num_running_reqs:
                break
            if num_remaining_tokens == 0:
                break

            request = self.waiting[0]
            allocated = self.block_manager.allocate(request)
            if allocated is None:
                # The request cannot be scheduled.
                break

            # The request can be scheduled.
            computed_block_ids, new_block_ids = allocated

            # Get cached tokens.
            num_computed_blocks = len(computed_block_ids)
            num_computed_tokens = num_computed_blocks * self.block_size

            # Number of tokens to be scheduled.
            num_tokens = request.num_tokens - num_computed_tokens
            num_tokens = min(num_tokens, num_remaining_tokens)

            self.waiting.popleft()
            self.running.append(request)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                assert False, f"Invalid request status: {request.status}"

            req_to_new_block_ids[request.request_id] = (
                computed_block_ids + new_block_ids)
            num_scheduled_tokens[request.request_id] = num_tokens
            total_num_scheduled_tokens += num_tokens
            num_remaining_tokens -= num_tokens

            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens + num_tokens
            if request.num_tokens == request.num_computed_tokens:
                request.num_output_tokens += 1

        # Check if the scheduling constraints are satisfied.
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert num_remaining_tokens >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) == len(self.running))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            ResumedRequestData.from_request(
                req, req_to_new_block_ids[req.request_id])
            for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            RunningRequestData.from_request(
                req, req_to_new_block_ids[req.request_id])
            for req in scheduled_running_reqs
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

    def _check_stop_by_len(self) -> None:
        stopped_reqs: List[Request] = []
        # TODO: Optimize this.
        for request in self.running:
            if (request.num_tokens >= self.max_model_len 
                or request.num_output_tokens >= request.max_tokens):
                request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                stopped_reqs.append(request)
        for request in stopped_reqs:
            self.running.remove(request)
            self.finished_req_ids.add(request.request_id)
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.block_manager.free(request)                

    def has_unfinished_requests(self) -> bool:
        return self.waiting or self.running

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)


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
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.inputs["prompt_token_ids"],
            prompt=request.inputs.get("prompt"),
            multi_modal_data=request.inputs.get("multi_modal_data"),
            sampling_params=request.sampling_params,
            block_ids=block_ids,
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
    ) -> "ResumedRequestData":
        return cls(
            req_id=request.request_id,
            block_ids=block_ids,
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
    ) -> "RunningRequestData":
        return cls(
            req_id=request.request_id,
            new_block_ids=new_block_ids,
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
