# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from vllm._bc_linter import bc_linter_include

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata)
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request


@bc_linter_include
@dataclass
class NewRequestData:

    req_id: str
    prompt_token_ids: Optional[list[int]]
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: Optional[LoRARequest]
    prompt_embeds: Optional[torch.Tensor] = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> NewRequestData:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = (self.prompt_embeds.shape
                               if self.prompt_embeds else None)
        return (f"NewRequestData("
                f"req_id={self.req_id},"
                f"prompt_token_ids={self.prompt_token_ids},"
                f"mm_features={self.mm_features},"
                f"sampling_params={self.sampling_params},"
                f"block_ids={self.block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                f"lora_request={self.lora_request},"
                f"prompt_embeds_shape={prompt_embeds_shape}"
                ")")

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = len(
            self.prompt_token_ids
        ) if self.prompt_token_ids is not None else None
        prompt_embeds_shape = (self.prompt_embeds.shape
                               if self.prompt_embeds else None)
        return (f"NewRequestData("
                f"req_id={self.req_id},"
                f"prompt_token_ids_len={prompt_token_ids_len},"
                f"mm_features={self.mm_features},"
                f"sampling_params={self.sampling_params},"
                f"block_ids={self.block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                f"lora_request={self.lora_request},"
                f"prompt_embeds_shape={prompt_embeds_shape}"
                ")")


@bc_linter_include
@dataclass
class CachedRequestData:

    req_ids: list[str]
    # If resumed_from_preemption is False, new_block_ids will be appended to
    # the request's block IDs. If True, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_from_preemption: list[bool]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    new_block_ids: list[Optional[tuple[list[int], ...]]]
    num_computed_tokens: list[int]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        return cls(
            req_ids=[],
            resumed_from_preemption=[],
            new_token_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
        )


@bc_linter_include
@dataclass
class SchedulerOutput:

    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    # Dict of request ids to their index within the batch
    # for filling the next token bitmask
    structured_output_request_ids: dict[str, int]
    # the bitmask for the whole batch
    grammar_bitmask: Optional[npt.NDArray[np.int32]]

    # KV Cache Connector metadata.
    kv_connector_metadata: Optional[KVConnectorMetadata] = None
