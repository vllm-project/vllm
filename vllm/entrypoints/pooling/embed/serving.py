# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Final, cast

import torch
from fastapi import Request
from typing_extensions import assert_never

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.openai.engine.serving import OpenAIServing, ServeContext
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingBytesResponse,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
)
from vllm.entrypoints.renderer import RenderConfig
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.pooling_params import PoolingParams
from vllm.utils.async_utils import merge_async_iterators
from vllm.utils.collection_utils import chunk_list
from vllm.utils.serial_utils import (
    encode_pooling_bytes,
    encode_pooling_output,
)

logger = init_logger(__name__)


EmbeddingServeContext = ServeContext[EmbeddingRequest]


class OpenAIServingEmbedding(OpenAIServing):
    request_id_prefix = "embd"

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template

        pooler_config = self.model_config.pooler_config

        # Avoid repeated attribute lookups
        self.supports_chunked_processing = bool(
            pooler_config and pooler_config.enable_chunked_processing
        )
        self.max_embed_len = (
            pooler_config.max_embed_len
            if pooler_config and pooler_config.max_embed_len
            else None
        )

    async def _preprocess(
        self,
        ctx: EmbeddingServeContext,
    ) -> ErrorResponse | None:
        try:
            ctx.lora_request = self._maybe_get_adapters(ctx.request)

            if isinstance(ctx.request, EmbeddingChatRequest):
                error_check_ret = self._validate_chat_template(
                    request_chat_template=ctx.request.chat_template,
                    chat_template_kwargs=ctx.request.chat_template_kwargs,
                    trust_request_chat_template=self.trust_request_chat_template,
                )
                if error_check_ret is not None:
                    return error_check_ret

                _, ctx.engine_prompts = await self._preprocess_chat(
                    ctx.request,
                    self.renderer,
                    ctx.request.messages,
                    chat_template=ctx.request.chat_template or self.chat_template,
                    chat_template_content_format=self.chat_template_content_format,
                    add_generation_prompt=ctx.request.add_generation_prompt,
                    continue_final_message=ctx.request.continue_final_message,
                    add_special_tokens=ctx.request.add_special_tokens,
                )
            elif isinstance(ctx.request, EmbeddingCompletionRequest):
                renderer = self._get_completion_renderer()
                ctx.engine_prompts = await renderer.render_prompt(
                    prompt_or_prompts=ctx.request.input,
                    config=self._build_render_config(ctx.request),
                )
            else:
                return self.create_error_response("Invalid classification request type")

            return None
        except (ValueError, TypeError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

    def _build_render_config(self, request: EmbeddingCompletionRequest) -> RenderConfig:
        # Set max_length based on chunked processing capability
        if self._should_use_chunked_processing(request):
            max_length = None
        else:
            max_length = self.max_embed_len or self.max_model_len

        return RenderConfig(
            max_length=max_length,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens,
        )

    def _build_response(
        self,
        ctx: EmbeddingServeContext,
    ) -> EmbeddingResponse | EmbeddingBytesResponse | ErrorResponse:
        final_res_batch_checked = ctx.final_res_batch

        encoding_format = ctx.request.encoding_format
        embed_dtype = ctx.request.embed_dtype
        endianness = ctx.request.endianness

        def encode_float_base64():
            items: list[EmbeddingResponseData] = []
            num_prompt_tokens = 0

            for idx, final_res in enumerate(final_res_batch_checked):
                item = EmbeddingResponseData(
                    index=idx,
                    embedding=encode_pooling_output(
                        final_res,
                        encoding_format=encoding_format,
                        embed_dtype=embed_dtype,
                        endianness=endianness,
                    ),
                )
                prompt_token_ids = final_res.prompt_token_ids

                items.append(item)
                num_prompt_tokens += len(prompt_token_ids)

            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                total_tokens=num_prompt_tokens,
            )

            return EmbeddingResponse(
                id=ctx.request_id,
                created=ctx.created_time,
                model=ctx.model_name,
                data=items,
                usage=usage,
            )

        def encode_bytes(bytes_only: bool) -> EmbeddingBytesResponse:
            content, items, usage = encode_pooling_bytes(
                pooling_outputs=final_res_batch_checked,
                embed_dtype=embed_dtype,
                endianness=endianness,
            )

            headers = (
                None
                if bytes_only
                else {
                    "metadata": json.dumps(
                        {
                            "id": ctx.request_id,
                            "created": ctx.created_time,
                            "model": ctx.model_name,
                            "data": items,
                            "usage": usage,
                        }
                    )
                }
            )

            return EmbeddingBytesResponse(content=content, headers=headers)

        if encoding_format == "float" or encoding_format == "base64":
            return encode_float_base64()
        elif encoding_format == "bytes" or encoding_format == "bytes_only":
            return encode_bytes(bytes_only=encoding_format == "bytes_only")
        else:
            assert_never(encoding_format)

    def _get_max_position_embeddings(self) -> int:
        """Get the model's effective maximum sequence length for chunking."""
        return self.model_config.max_model_len

    def _should_use_chunked_processing(self, request) -> bool:
        """Check if chunked processing should be used for this request."""
        return (
            isinstance(request, (EmbeddingCompletionRequest, EmbeddingChatRequest))
            and self.supports_chunked_processing
        )

    async def _process_chunked_request(
        self,
        ctx: EmbeddingServeContext,
        token_ids: list[int],
        pooling_params: PoolingParams,
        trace_headers: Mapping[str, str] | None,
        prompt_idx: int,
    ) -> list[AsyncGenerator[PoolingRequestOutput, None]]:
        """Process a single prompt using chunked processing."""
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        # Split into chunks using max_position_embeddings
        max_pos_embeddings = self._get_max_position_embeddings()
        # Process all chunks for MEAN aggregation
        for chunk_idx, chunk_tokens in enumerate(
            chunk_list(token_ids, max_pos_embeddings)
        ):
            # Create a request ID for this chunk
            chunk_request_id = f"{ctx.request_id}-prompt-{prompt_idx}-chunk-{chunk_idx}"

            # Create engine prompt for this chunk
            chunk_engine_prompt = TokensPrompt(prompt_token_ids=chunk_tokens)

            cache_hit_threshold = getattr(ctx.request, "cache_hit_threshold", None)
            # Log the chunk
            self._log_inputs(
                chunk_request_id,
                chunk_engine_prompt,
                params=pooling_params,
                lora_request=ctx.lora_request,
                cache_hit_threshold=cache_hit_threshold,
            )

            # Create generator for this chunk and wrap it to return indices
            original_generator = self.engine_client.encode(
                chunk_engine_prompt,
                pooling_params,
                chunk_request_id,
                lora_request=ctx.lora_request,
                trace_headers=trace_headers,
                priority=getattr(ctx.request, "priority", 0),
                cache_hit_threshold=cache_hit_threshold,
            )

            generators.append(original_generator)

        return generators

    def _validate_input(
        self,
        request: object,
        input_ids: list[int],
        input_text: str,
    ) -> TokensPrompt:
        """Override to support chunked processing for embedding requests."""
        token_num = len(input_ids)

        # Note: EmbeddingRequest doesn't have max_tokens
        if isinstance(request, (EmbeddingCompletionRequest, EmbeddingChatRequest)):
            # Check if chunked processing is enabled for pooling models
            enable_chunked = self._should_use_chunked_processing(request)

            # Use max_position_embeddings for chunked processing decisions
            max_pos_embeddings = self._get_max_position_embeddings()

            # Determine the effective max length for validation
            if self.max_embed_len is not None:
                # Use max_embed_len for validation instead of max_model_len
                length_type = "maximum embedding input length"
                max_length_value = self.max_embed_len
            else:
                # Fall back to max_model_len validation (original behavior)
                length_type = "maximum context length"
                max_length_value = self.max_model_len

            validation_error_msg = (
                "This model's {length_type} is {max_length_value} tokens. "
                "However, you requested {token_num} tokens in the input for "
                "embedding generation. Please reduce the length of the input."
            )

            chunked_processing_error_msg = (
                "This model's {length_type} is {max_length_value} tokens. "
                "However, you requested {token_num} tokens in the input for "
                "embedding generation. Please reduce the length of the input "
                "or enable chunked processing."
            )

            # Check if input exceeds max length
            if token_num > max_length_value:
                raise ValueError(
                    validation_error_msg.format(
                        length_type=length_type,
                        max_length_value=max_length_value,
                        token_num=token_num,
                    )
                )

            # Check for chunked processing
            # when exceeding max_position_embeddings
            if token_num > max_pos_embeddings:
                if enable_chunked:
                    # Allow long inputs when chunked processing is enabled
                    logger.info(
                        "Input length %s exceeds max_position_embeddings "
                        "%s, will use chunked processing",
                        token_num,
                        max_pos_embeddings,
                    )
                else:
                    raise ValueError(
                        chunked_processing_error_msg.format(
                            length_type="maximum position embeddings length",
                            max_length_value=max_pos_embeddings,
                            token_num=token_num,
                        )
                    )

            return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # For other request types, use the parent's implementation
        return super()._validate_input(request, input_ids, input_text)

    async def _create_single_prompt_generator(
        self,
        ctx: EmbeddingServeContext,
        engine_prompt: TokensPrompt,
        pooling_params: PoolingParams,
        trace_headers: Mapping[str, str] | None,
        prompt_index: int,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Create a generator for a single prompt using standard processing."""
        request_id_item = f"{ctx.request_id}-{prompt_index}"
        cache_hit_threshold = getattr(ctx.request, "cache_hit_threshold", None)

        self._log_inputs(
            request_id_item,
            engine_prompt,
            params=pooling_params,
            lora_request=ctx.lora_request,
            cache_hit_threshold=cache_hit_threshold,
        )

        # Return the original generator without wrapping
        return self.engine_client.encode(
            engine_prompt,
            pooling_params,
            request_id_item,
            lora_request=ctx.lora_request,
            trace_headers=trace_headers,
            priority=getattr(ctx.request, "priority", 0),
            cache_hit_threshold=cache_hit_threshold,
        )

    async def _prepare_generators(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """Override to support chunked processing."""
        ctx = cast(EmbeddingServeContext, ctx)

        # Check if we should use chunked processing
        use_chunked = self._should_use_chunked_processing(ctx.request)

        # If no chunked processing needed, delegate to parent class
        if not use_chunked:
            return await super()._prepare_generators(ctx)

        # Custom logic for chunked processing
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        try:
            trace_headers = (
                None
                if ctx.raw_request is None
                else await self._get_trace_headers(ctx.raw_request.headers)
            )

            pooling_params = self._create_pooling_params(ctx)
            if isinstance(pooling_params, ErrorResponse):
                return pooling_params

            # Verify and set the task for pooling params
            try:
                pooling_params.verify("embed", self.model_config)
            except ValueError as e:
                return self.create_error_response(str(e))

            if ctx.engine_prompts is None:
                return self.create_error_response("Engine prompts not available")

            max_pos_embeddings = self._get_max_position_embeddings()

            for i, engine_prompt in enumerate(ctx.engine_prompts):
                # Check if this specific prompt needs chunked processing
                if "prompt_token_ids" in engine_prompt:
                    prompt_token_ids = engine_prompt["prompt_token_ids"]
                    if len(prompt_token_ids) > max_pos_embeddings:
                        # Use chunked processing for this prompt
                        chunk_generators = await self._process_chunked_request(
                            ctx,
                            prompt_token_ids,
                            pooling_params,
                            trace_headers,
                            i,
                        )
                        generators.extend(chunk_generators)
                        continue

                # Normal processing for short prompts or non-token prompts
                generator = await self._create_single_prompt_generator(
                    ctx, engine_prompt, pooling_params, trace_headers, i
                )
                generators.append(generator)

            ctx.result_generator = merge_async_iterators(*generators)

            return None

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def _collect_batch(
        self,
        ctx: EmbeddingServeContext,
    ) -> ErrorResponse | None:
        """Collect and aggregate batch results
        with support for chunked processing.

        For chunked requests, performs online aggregation to
        minimize memory usage.
        For regular requests, collects results normally.
        """
        try:
            if ctx.engine_prompts is None:
                return self.create_error_response("Engine prompts not available")

            # Check if we used chunked processing
            use_chunked = self._should_use_chunked_processing(ctx.request)

            if not use_chunked:
                return await super()._collect_batch(ctx=ctx)

            if ctx.result_generator is None:
                return self.create_error_response("Result generator not available")

            # Online aggregation for chunked requests to
            # minimize memory usage
            # Track aggregation state for each prompt
            prompt_aggregators: dict[int, dict[str, Any]] = {}
            short_prompts_results: dict[int, PoolingRequestOutput] = {}

            async for result_idx, result in ctx.result_generator:
                if "-chunk-" in result.request_id:
                    # Extract prompt_idx from chunked request_id
                    parts = result.request_id.split("-")
                    try:
                        prompt_idx = int(parts[parts.index("prompt") + 1])
                    except (ValueError, IndexError):
                        # Fallback: extract from result_idx if parsing fails
                        prompt_idx = result_idx

                    # Initialize aggregator for this prompt if needed
                    if prompt_idx not in prompt_aggregators:
                        prompt_aggregators[prompt_idx] = {
                            "weighted_sum": None,
                            "total_weight": 0,
                            "chunk_count": 0,
                            "request_id": result.request_id.split("-chunk-")[0],
                        }

                    aggregator = prompt_aggregators[prompt_idx]

                    # MEAN pooling with online weighted averaging
                    # Ensure result is PoolingRequestOutput
                    # for embedding processing
                    if not isinstance(result, PoolingRequestOutput):
                        return self.create_error_response(
                            f"Expected PoolingRequestOutput for "
                            f"chunked embedding, got "
                            f"{type(result).__name__}"
                        )

                    # Handle both PoolingOutput and
                    # EmbeddingOutput types
                    if hasattr(result.outputs, "data"):
                        # PoolingOutput case
                        embedding_data = result.outputs.data
                    elif hasattr(result.outputs, "embedding"):
                        # EmbeddingOutput case -
                        # convert embedding list to tensor
                        embedding_data = result.outputs.embedding
                    else:
                        return self.create_error_response(
                            f"Unsupported output type: {type(result.outputs).__name__}"
                        )

                    if not isinstance(embedding_data, torch.Tensor):
                        embedding_data = torch.tensor(
                            embedding_data, dtype=torch.float32
                        )

                    if result.prompt_token_ids is None:
                        return self.create_error_response(
                            "prompt_token_ids cannot be None for chunked processing"
                        )
                    weight = len(result.prompt_token_ids)

                    weighted_embedding = embedding_data.to(dtype=torch.float32) * weight

                    if aggregator["weighted_sum"] is None:
                        # First chunk
                        aggregator["weighted_sum"] = weighted_embedding
                    else:
                        # Accumulate
                        aggregator["weighted_sum"] += weighted_embedding

                    aggregator["total_weight"] += weight
                    aggregator["chunk_count"] += 1
                else:
                    # Non-chunked result - extract prompt_idx from request_id
                    parts = result.request_id.split("-")
                    try:
                        # Last part should be prompt index
                        prompt_idx = int(parts[-1])
                    except (ValueError, IndexError):
                        prompt_idx = result_idx  # Fallback to result_idx

                    short_prompts_results[prompt_idx] = result

            # Finalize aggregated results
            final_res_batch: list[PoolingRequestOutput] = []
            num_prompts = len(ctx.engine_prompts)

            for prompt_idx in range(num_prompts):
                if prompt_idx in prompt_aggregators:
                    # Finalize MEAN aggregation for this chunked prompt
                    aggregator = prompt_aggregators[prompt_idx]

                    weighted_sum = aggregator["weighted_sum"]
                    total_weight = aggregator["total_weight"]

                    if (
                        weighted_sum is not None
                        and isinstance(weighted_sum, torch.Tensor)
                        and isinstance(total_weight, (int, float))
                        and total_weight > 0
                    ):
                        # Compute final mean embedding
                        final_embedding = weighted_sum / total_weight

                        # Create a PoolingRequestOutput
                        # for the aggregated result
                        pooling_output_data = PoolingOutput(data=final_embedding)

                        # Get original prompt token IDs for this prompt
                        original_prompt = ctx.engine_prompts[prompt_idx]
                        if "prompt_token_ids" not in original_prompt:
                            return self.create_error_response(
                                f"Chunked prompt {prompt_idx} does not contain "
                                "token IDs"
                            )

                        original_token_ids = original_prompt["prompt_token_ids"]

                        pooling_request_output = PoolingRequestOutput(
                            request_id=aggregator["request_id"],
                            prompt_token_ids=original_token_ids,
                            outputs=pooling_output_data,
                            num_cached_tokens=0,
                            finished=True,
                        )

                        final_res_batch.append(pooling_request_output)
                    else:
                        return self.create_error_response(
                            f"Failed to aggregate chunks for prompt {prompt_idx}"
                        )
                elif prompt_idx in short_prompts_results:
                    final_res_batch.append(short_prompts_results[prompt_idx])
                else:
                    return self.create_error_response(
                        f"Result not found for prompt {prompt_idx}"
                    )

            ctx.final_res_batch = final_res_batch

            return None

        except Exception as e:
            return self.create_error_response(str(e))

    async def create_embedding(
        self,
        request: EmbeddingRequest,
        raw_request: Request | None = None,
    ) -> EmbeddingResponse | ErrorResponse:
        """
        Embedding API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        model_name = self.models.model_name()
        request_id = (
            f"{self.request_id_prefix}-"
            f"{self._base_request_id(raw_request, request.request_id)}"
        )

        ctx = EmbeddingServeContext(
            request=request,
            raw_request=raw_request,
            model_name=model_name,
            request_id=request_id,
        )

        return await self.handle(ctx)  # type: ignore[return-value]

    def _create_pooling_params(
        self,
        ctx: EmbeddingServeContext,
    ) -> PoolingParams | ErrorResponse:
        pooling_params = super()._create_pooling_params(ctx)
        if isinstance(pooling_params, ErrorResponse):
            return pooling_params

        try:
            pooling_params.verify("embed", self.model_config)
        except ValueError as e:
            return self.create_error_response(str(e))

        return pooling_params
