from collections.abc import Callable, Mapping
from typing import Optional

import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request
from vllm.v1.streaming.engine import StreamingEngineCoreRequest


class StreamingRequest(Request):
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        streaming_sequence_id: int,
        close_session: bool,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: Optional["LoRARequest"] = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
    ) -> None:
        super().__init__(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            eos_token_id=eos_token_id,
            client_index=client_index,
            arrival_time=arrival_time,
            prompt_embeds=prompt_embeds,
            mm_features=mm_features,
            lora_request=lora_request,
            cache_salt=cache_salt,
            priority=priority,
            trace_headers=trace_headers,
            block_hasher=block_hasher,
        )
        self.streaming_sequence_id = streaming_sequence_id
        self.close_session = close_session

    @classmethod
    def from_engine_core_request(
        cls,
        request: StreamingEngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "StreamingRequest":
        return cls(
            request_id=request.request_id,
            streaming_sequence_id=request.streaming_sequence_id,
            close_session=request.close_session,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
        )
