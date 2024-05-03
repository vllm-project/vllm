import asyncio
import concurrent.futures
from typing import Optional

from vllm.model_executor.guided_decoding.fields import GuidedDecodingFields
from vllm.model_executor.guided_decoding.outlines_decoding import (
    get_outlines_guided_decoding_logits_processor)
from vllm.sampling_params import LogitsProcessor

global_thread_pool = None


async def get_guided_decoding_logits_processor_async(
        request: GuidedDecodingFields, tokenizer) -> Optional[LogitsProcessor]:
    global global_thread_pool
    if global_thread_pool is None:
        global_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4)
    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(
        global_thread_pool,
        get_guided_decoding_logits_processor,
        request,
        tokenizer,
    )


def get_guided_decoding_logits_processor(
        request: GuidedDecodingFields, tokenizer) -> Optional[LogitsProcessor]:
    if request.guided_decoding_backend == 'outlines':
        return get_outlines_guided_decoding_logits_processor(
            request, tokenizer)
    if request.guided_decoding_backend == 'lm-format-enforcer':
        ## Import moved inside function to avoide circular
        ## import with vllm.entrypoints.LLM.py
        from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (  # noqa
            get_lm_format_enforcer_guided_decoding_logits_processor)
        return get_lm_format_enforcer_guided_decoding_logits_processor(
            request, tokenizer)

    raise ValueError(
        f"Unknown guided decoding backend '{request.guided_decoding_backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer'")


__all__ = ['get_guided_decoding_logits_processor', 'GuidedDecodingFields']
