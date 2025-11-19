import time
from collections.abc import Mapping
from typing import Any

from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.processor import Processor
from vllm.v1.streaming.engine import StreamingEngineCoreRequest


class StreamingProcessor(Processor):
    def _validate_streaming_sequence_id(self, streaming_sequence_id: int) -> None:
        if streaming_sequence_id < 0:
            raise ValueError(
                f"streaming_sequence_id must be >= 0, got {streaming_sequence_id}"
            )
    
    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: SamplingParams | PoolingParams,
        streaming_sequence_id: int,
        close_session: bool,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> StreamingEngineCoreRequest:
        self._validate_streaming_sequence_id(streaming_sequence_id)

        # For close_session requests, skip input preprocessing
        # and create a minimal request to signal session closure
        if close_session:
            return StreamingEngineCoreRequest(
                request_id=request_id,
                prompt_token_ids=[],
                mm_features=None,
                sampling_params=params if isinstance(params, SamplingParams) else None,
                pooling_params=params if isinstance(params, PoolingParams) else None,
                eos_token_id=None,
                arrival_time=arrival_time if arrival_time else time.time(),
                lora_request=lora_request,
                cache_salt=None,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                streaming_sequence_id=streaming_sequence_id,
                close_session=True,
            )

        req = super().process_inputs(
            request_id,
            prompt,
            params,
            arrival_time,
            lora_request,
            tokenization_kwargs,
            trace_headers,
            priority,
            data_parallel_rank,
        )
        streaming_req = StreamingEngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=req.prompt_token_ids,
            mm_features=req.mm_features,
            sampling_params=req.sampling_params,
            pooling_params=req.pooling_params,
            eos_token_id=req.eos_token_id,
            arrival_time=req.arrival_time,
            lora_request=req.lora_request,
            cache_salt=req.cache_salt,
            priority=req.priority,
            data_parallel_rank=req.data_parallel_rank,
            streaming_sequence_id=streaming_sequence_id,
            close_session=close_session,
        )
        del req
        return streaming_req
