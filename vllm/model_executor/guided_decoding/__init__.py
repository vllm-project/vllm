from typing import Optional, Union

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest)
from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (
    get_lm_format_enforcer_guided_decoding_logits_processor)
from vllm.model_executor.guided_decoding.outlines_decoding import (
    get_outlines_guided_decoding_logits_processor)
from vllm.sampling_params import LogitsProcessor


async def get_guided_decoding_logits_processor(
        guided_decoding_backend: str, request: Union[CompletionRequest,
                                                     ChatCompletionRequest],
        tokenizer) -> Optional[LogitsProcessor]:
    if guided_decoding_backend == 'outlines':
        return await get_outlines_guided_decoding_logits_processor(
            request, tokenizer)
    if guided_decoding_backend == 'lm-format-enforcer':
        return await get_lm_format_enforcer_guided_decoding_logits_processor(
            request, tokenizer)

    raise ValueError(
        f"Unknown guided decoding backend '{guided_decoding_backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer'")
