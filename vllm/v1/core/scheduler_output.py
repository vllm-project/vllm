# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.base import PlaceholderRange
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request


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
    lora_request: Optional["LoRARequest"]

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
            lora_request=request.lora_request,
        )


@dataclass
class CachedRequestData:

    req_id: str
    # If resumed_from_preemption is False, new_block_ids will be appended to
    # the request's block IDs. If True, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_from_preemption: bool
    new_block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: "Request",
        resumed_from_preemption: bool,
        new_block_ids: List[int],
        num_computed_tokens: int,
    ) -> "CachedRequestData":
        return cls(
            req_id=request.request_id,
            resumed_from_preemption=resumed_from_preemption,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )


@dataclass
class SchedulerOutput:

    scheduled_new_reqs: List[NewRequestData]
    scheduled_cached_reqs: List[CachedRequestData]

    num_scheduled_tokens: Dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_encoder_inputs: Dict[str, List[int]]
    num_common_prefix_blocks: int

    finished_req_ids: Set[str]
    free_encoder_input_ids: List[Tuple[str, int]]
