# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

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
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import BlockHash, get_request_block_hasher
from vllm.v1.request import Request as EngineRequest

from .hash_protocol import HashGroup, HashPrompt, HashResponse, HashTokensRequest
from .serving import ServingRender


async def hash_request(
    request: ChatCompletionRequest | CompletionRequest | HashTokensRequest,
    raw_request: Request,
    renderer: ServingRender,
    engine: EngineClient,
) -> HashResponse | ErrorResponse:
    """Render and hash prompts locally with the engine's initialized settings."""
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

    config = engine.get_prefix_hash_config()
    if config is None or config.kv_cache_groups is None:
        return renderer.create_error_response(
            "The engines must provide compatible prefix hash metadata"
        )
    block_hasher = get_request_block_hasher(
        config.hash_block_size,
        get_hash_fn_by_name(config.hash_algo),
        config.num_lookahead_tokens,
        none_hash=BlockHash(config.none_hash),
    )

    lora_request = renderer._maybe_get_adapters(request, supports_default_mm_loras=True)
    rendered = result if isinstance(result, list) else [result]
    prompts = []
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
        core_request = engine.input_processor.process_inputs(
            prompt.request_id,
            engine_input,
            prompt.sampling_params,
            supported_tasks=("generate",),
            lora_request=lora_request,
        )
        hashed = await asyncio.to_thread(
            EngineRequest.from_engine_core_request, core_request, block_hasher
        )
        prompts.append(
            HashPrompt(
                num_tokens=hashed.num_prompt_tokens,
                block_hashes=[h.hex() for h in hashed.block_hashes],
            )
        )

    return HashResponse(
        model=renderer.models.base_model_paths[0].name,
        hash_algorithm=config.hash_algo,
        hash_block_size=config.hash_block_size,
        num_hash_lookahead_tokens=config.num_lookahead_tokens,
        groups=[HashGroup.model_validate(group) for group in config.kv_cache_groups],
        prompts=prompts,
    )
