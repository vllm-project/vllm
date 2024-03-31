from typing import Union


from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest)

from vllm.sampling_params import LogitsProcessor
from vllm.model_executor.guided_decoding.outlines_decoding import get_outlines_guided_decoding_logits_processor

async def get_guided_decoding_logits_processor(
        guided_decoding_backend: str,
        request: Union[CompletionRequest, ChatCompletionRequest],
        tokenizer) -> LogitsProcessor:
    if guided_decoding_backend == 'outlines':
        return await get_outlines_guided_decoding_logits_processor(request, tokenizer)
    raise ValueError(f"Unknown guided decoding backend '{guided_decoding_backend}'. Must be one of 'outlines'")