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
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
        )


@dataclass
class CachedRequestData:

    req_id: str
    # If resumed_from_preemption is False, new_block_ids will be appended to
    # the request's block IDs. If True, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_from_preemption: bool
    new_token_ids: List[int]
    new_block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: "Request",
        resumed_from_preemption: bool,
        new_token_ids: List[int],
        new_block_ids: List[int],
    ) -> "CachedRequestData":
        return cls(
            req_id=request.request_id,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )


@dataclass
class SchedulerOutput:

    # List of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: List[NewRequestData]
    # List of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: List[CachedRequestData]

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: Dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: Dict[str, List[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: Dict[str, List[int]]
    # Number of common prefix blocks for all requests.
    # This can be used for cascade attention.
    num_common_prefix_blocks: int

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: Set[str]
    # List of (req_id, encoder_input_index) tuples.
    # Used to free the encoder cache.
    free_encoder_input_ids: List[Tuple[str, int]]
