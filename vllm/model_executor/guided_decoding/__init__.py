import asyncio
import concurrent.futures
from typing import Optional, Union

from vllm.entrypoints.openai.protocol import (
    ChatCompletionNamedToolChoiceParam, ChatCompletionRequest,
    CompletionRequest)
from vllm.model_executor.guided_decoding.fields import GuidedDecodingFields
from vllm.model_executor.guided_decoding.outlines_decoding import (
    get_outlines_guided_decoding_logits_processor, get_outlines_guided_decoding_logits_processor_async)
from vllm.sampling_params import LogitsProcessor

global_thread_pool = None


async def get_guided_decoding_logits_processor_async(
        request: Union[CompletionRequest,
                   ChatCompletionRequest], tokenizer) -> Optional[LogitsProcessor]:
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
# async def get_guided_decoding_logits_processor_async(
#     guided_decoding_backend: str, request: Union[CompletionRequest,
#                                                      ChatCompletionRequest],
#         tokenizer) -> Optional[LogitsProcessor]:
#     request = _adapt_request_for_tool_use(request)

#     if guided_decoding_backend == 'outlines':
#         return await get_outlines_guided_decoding_logits_processor_async(
#             request, tokenizer)
#     if guided_decoding_backend == 'lm-format-enforcer':
#         from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (  # noqa
#             get_lm_format_enforcer_guided_decoding_logits_processor)
#         options = GuidedDecodingFields.from_openai_request(request)
#         return get_lm_format_enforcer_guided_decoding_logits_processor(
#             options, tokenizer)

    # raise ValueError(
    #     f"Unknown guided decoding backend '{guided_decoding_backend}'. "
    #     "Must be one of 'outlines, 'lm-format-enforcer'")


# async def get_guided_decoding_logits_processor(
#         guided_decoding_backend: str, request: Union[CompletionRequest,
#                                                      ChatCompletionRequest],
#         tokenizer) -> Optional[LogitsProcessor]:
#     request = _adapt_request_for_tool_use(request)

#     if guided_decoding_backend == 'outlines':
#         return await get_outlines_guided_decoding_logits_processor(
#             request, tokenizer)
#     if guided_decoding_backend == 'lm-format-enforcer':
#         return await get_lm_format_enforcer_guided_decoding_logits_processor(
#             request, tokenizer)

#     raise ValueError(
#         f"Unknown guided decoding backend '{guided_decoding_backend}'. "
#         "Must be one of 'outlines, 'lm-format-enforcer'")


def get_guided_decoding_logits_processor(
        request: Union[CompletionRequest,
                   ChatCompletionRequest, GuidedDecodingFields], tokenizer) -> Optional[LogitsProcessor]:
    # request = _adapt_request_for_tool_use(request)
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


def _adapt_request_for_tool_use(request: Union[CompletionRequest,
                                               ChatCompletionRequest]):
    # the legacy completion API does not support tool use
    if type(request) is CompletionRequest:
        return request

    # user has chosen to not use any tool
    if request.tool_choice == "none":
        return request

    # user has chosen to use a named tool
    if type(request.tool_choice) is ChatCompletionNamedToolChoiceParam:
        tool_name = request.tool_choice.function.name
        tools = {tool.function.name: tool.function for tool in request.tools}
        if tool_name not in tools:
            raise ValueError(
                f"Tool '{tool_name}' has not been passed in `tools`.")
        tool = tools[tool_name]
        request.guided_json = tool.parameters

    return request
