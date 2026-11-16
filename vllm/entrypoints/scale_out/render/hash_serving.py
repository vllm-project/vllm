# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.mm_features import (
    rebuild_mm_placeholders,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.inputs import EngineInput, mm_input, tokens_input
from vllm.multimodal.inputs import MultiModalKwargsItems

from .hash_protocol import HashResponse, HashTokensRequest
from .serving import ServingRender


async def hash_request(
    request: ChatCompletionRequest | CompletionRequest | HashTokensRequest,
    raw_request: Request,
    renderer: ServingRender,
    engine: EngineClient,
) -> HashResponse | ErrorResponse:
    """Render a request and hash its prompts using the running engine."""
    if isinstance(request, CompletionRequest) and request.prompt_embeds is not None:
        return renderer.create_error_response("The hash endpoint requires token IDs")
    if isinstance(request, ChatCompletionRequest):
        for message in request.messages:
            content = message.get("content")
            if isinstance(content, list) and any(
                part.get("type") == "prompt_embeds" for part in content
            ):
                return renderer.create_error_response(
                    "The hash endpoint requires token IDs"
                )
    result: GenerateRequest | list[GenerateRequest] | ErrorResponse
    if isinstance(request, ChatCompletionRequest):
        result = await renderer.render_chat_request(request, raw_request)
    elif isinstance(request, CompletionRequest):
        result = await renderer.render_completion_request(request)
    else:
        if error := await renderer._check_model(request):
            return error
        result = request
    if isinstance(result, ErrorResponse):
        return result
    if renderer.model_config.is_encoder_decoder:
        return renderer.create_error_response(
            "The hash endpoint requires a decoder-only model"
        )
    if not engine.vllm_config.cache_config.enable_prefix_caching:
        return renderer.create_error_response(
            "Enable prefix caching to use the hash endpoint"
        )

    lora_request = renderer._maybe_get_adapters(request, supports_default_mm_loras=True)
    rendered = result if isinstance(result, list) else [result]
    core_requests = []
    for prompt in rendered:
        engine_input: EngineInput
        if features := prompt.features:
            # Hashing only needs identifiers and positions, never tensor data.
            engine_input = mm_input(
                prompt_token_ids=prompt.token_ids,
                mm_kwargs=MultiModalKwargsItems(
                    {m: [None] * len(h) for m, h in features.mm_hashes.items()}
                ),
                mm_hashes=features.mm_hashes,
                mm_placeholders=rebuild_mm_placeholders(features.mm_placeholders),
                cache_salt=prompt.cache_salt,
            )
        else:
            engine_input = tokens_input(
                prompt_token_ids=prompt.token_ids, cache_salt=prompt.cache_salt
            )
        core_requests.append(
            engine.input_processor.process_inputs(
                prompt.request_id,
                engine_input,
                prompt.sampling_params,
                supported_tasks=("generate",),
                lora_request=lora_request,
            )
        )

    hashes = await engine.hash_requests(core_requests)
    return HashResponse(model=renderer.models.base_model_paths[0].name, **hashes)
