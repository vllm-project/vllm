from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple,
                    Union)

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.base import PlaceholderRange
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus


class SchedulerInterface(ABC):

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> "EngineCoreOutputs":
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "Request") -> None:
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: "RequestStatus",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def has_unfinished_requests(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reset_prefix_cache(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> "SchedulerStats":
        raise NotImplementedError


@dataclass
class NewRequestData:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List["MultiModalKwargs"]
    mm_hashes: List[str]
    mm_positions: List["PlaceholderRange"]
    sampling_params: "SamplingParams"
    block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: "Request",
        block_ids: List[int],
        num_computed_tokens: int,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt=request.prompt,
            mm_inputs=request.mm_inputs,
            mm_hashes=request.mm_hashes,
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
        request: "Request",
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
        request: "Request",
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
    preempted_req_ids: Set[str]
    finished_req_ids: Set[str]

    num_scheduled_tokens: Dict[str, int]
    total_num_scheduled_tokens: int

    # Optional fields
    scheduled_encoder_inputs: Dict[str, List[int]] = \
        field(default_factory=dict)
    free_encoder_input_ids: List[Tuple[str, int]] = field(default_factory=list)
    num_common_prefix_blocks: int = 0
