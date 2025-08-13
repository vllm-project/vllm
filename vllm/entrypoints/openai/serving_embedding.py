# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
from collections.abc import AsyncGenerator
from typing import Any, Final, Literal, Optional, Union, cast

import numpy as np
import torch
from fastapi import Request
from typing_extensions import assert_never, override

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this docstring
# yapf: disable
from vllm.entrypoints.openai.protocol import (EmbeddingChatRequest,
                                              EmbeddingRequest,
                                              EmbeddingResponse,
                                              EmbeddingResponseData,
                                              ErrorResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (EmbeddingServeContext,
                                                    OpenAIServing,
                                                    ServeContext,
                                                    TextTokensPrompt)
# yapf: enable
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import EmbedsPrompt as EngineEmbedsPrompt
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.logger import init_logger
from vllm.outputs import (EmbeddingOutput, EmbeddingRequestOutput,
                          PoolingOutput, PoolingRequestOutput, RequestOutput)
from vllm.pooling_params import PoolingParams
from vllm.utils import chunk_list

logger = init_logger(__name__)


def _get_embedding(
    output: EmbeddingOutput,
    encoding_format: Literal["float", "base64"],
) -> Union[list[float], str]:
    if encoding_format == "float":
        return output.embedding
    elif encoding_format == "base64":
        # Force to use float32 for base64 encoding
        # to match the OpenAI python client behavior
        embedding_bytes = np.array(output.embedding, dtype="float32").tobytes()
        return base64.b64encode(embedding_bytes).decode("utf-8")

    assert_never(encoding_format)


class EmbeddingMixin(OpenAIServing):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Cache chunked processing support to avoid repeated attribute lookups
        pooler_config = getattr(self.model_config, 'pooler_config', None)
        self.supports_chunked_processing = (
            pooler_config is not None
            and getattr(pooler_config, 'enable_chunked_processing', False))

        # Cache max_embed_len to avoid repeated attribute lookups
        self.max_embed_len = (pooler_config.max_embed_len if pooler_config
                              and pooler_config.max_embed_len else None)

    @override
    async def _preprocess(
        self,
        ctx: ServeContext,
    ) -> Optional[ErrorResponse]:
        ctx = cast(EmbeddingServeContext, ctx)
        try:
            ctx.lora_request = self._maybe_get_adapters(ctx.request)

            tokenizer = await self.engine_client.get_tokenizer(ctx.lora_request
                                                               )

            if isinstance(ctx.request, EmbeddingChatRequest):
                (
                    _,
                    ctx.request_prompts,
                    ctx.engine_prompts,
                ) = await self._preprocess_chat(
                    ctx.request,
                    tokenizer,
                    ctx.request.messages,
                    chat_template=ctx.request.chat_template
                    or ctx.chat_template,
                    chat_template_content_format=ctx.
                    chat_template_content_format,
                    # In embedding requests, we are not generating tokens,
                    # so there is no need to append extra tokens to the input
                    add_generation_prompt=False,
                    continue_final_message=False,
                    truncate_prompt_tokens=ctx.truncate_prompt_tokens,
                    add_special_tokens=ctx.request.add_special_tokens,
                )
            else:
                (ctx.request_prompts,
                 ctx.engine_prompts) = await self._preprocess_completion(
                     ctx.request,
                     tokenizer,
                     ctx.request.input,
                     truncate_prompt_tokens=ctx.truncate_prompt_tokens,
                     add_special_tokens=ctx.request.add_special_tokens,
                 )
            return None
        except (ValueError, TypeError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

    @override
    def _build_response(
        self,
        ctx: ServeContext,
    ) -> Union[EmbeddingResponse, ErrorResponse]:
        items: list[EmbeddingResponseData] = []
        num_prompt_tokens = 0

        final_res_batch_checked = cast(list[PoolingRequestOutput],
                                       ctx.final_res_batch)

        for idx, final_res in enumerate(final_res_batch_checked):
            embedding_res = EmbeddingRequestOutput.from_base(final_res)

            item = EmbeddingResponseData(
                index=idx,
                embedding=_get_embedding(embedding_res.outputs,
                                         ctx.request.encoding_format),
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

    def _get_max_position_embeddings(self) -> int:
        """Get the model's effective maximum sequence length for chunking."""
        return self.model_config.max_model_len

    def _should_use_chunked_processing(self, request) -> bool:
        """Check if chunked processing should be used for this request."""
        return isinstance(
            request, EmbeddingRequest) and self.supports_chunked_processing

    async def _process_chunked_request(
        self,
        ctx: EmbeddingServeContext,
        original_prompt: TextTokensPrompt,
        pooling_params,
        trace_headers,
        prompt_idx: int,
    ) -> list[AsyncGenerator[PoolingRequestOutput, None]]:
        """Process a single prompt using chunked processing."""
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []
        token_ids = original_prompt["prompt_token_ids"]

        # Split into chunks using max_position_embeddings
        max_pos_embeddings = self._get_max_position_embeddings()
        # Process all chunks for MEAN aggregation
        for chunk_idx, chunk_tokens in enumerate(
                chunk_list(token_ids, max_pos_embeddings)):
            # Create a request ID for this chunk
            chunk_request_id = (f"{ctx.request_id}-prompt-{prompt_idx}-"
                                f"chunk-{chunk_idx}")

            # Create engine prompt for this chunk
            chunk_engine_prompt = EngineTokensPrompt(
                prompt_token_ids=chunk_tokens)

            # Create chunk request prompt for logging
            chunk_text = ""
            chunk_request_prompt = TextTokensPrompt(
                prompt=chunk_text, prompt_token_ids=chunk_tokens)

            # Log the chunk
            self._log_inputs(chunk_request_id,
                             chunk_request_prompt,
                             params=pooling_params,
                             lora_request=ctx.lora_request)

            # Create generator for this chunk
            generator = self.engine_client.encode(
                chunk_engine_prompt,
                pooling_params,
                chunk_request_id,
                lora_request=ctx.lora_request,
                trace_headers=trace_headers,
                priority=getattr(ctx.request, "priority", 0),
            )

            generators.append(generator)

        return generators

    def _validate_input(
        self,
        request,
        input_ids: list[int],
        input_text: str,
    ) -> TextTokensPrompt:
        """Override to support chunked processing for embedding requests."""
        token_num = len(input_ids)

        # Note: EmbeddingRequest doesn't have max_tokens
        if isinstance(request, EmbeddingRequest):
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
                "embedding generation. Please reduce the length of the input.")

            chunked_processing_error_msg = (
                "This model's {length_type} is {max_length_value} tokens. "
                "However, you requested {token_num} tokens in the input for "
                "embedding generation. Please reduce the length of the input "
                "or enable chunked processing.")

            # Check if input exceeds max length
            if token_num > max_length_value:
                raise ValueError(
                    validation_error_msg.format(
                        length_type=length_type,
                        max_length_value=max_length_value,
                        token_num=token_num))

            # Check for chunked processing
            # when exceeding max_position_embeddings
            if token_num > max_pos_embeddings:
                if enable_chunked:
                    # Allow long inputs when chunked processing is enabled
                    logger.info(
                        "Input length %s exceeds max_position_embeddings "
                        "%s, will use chunked processing", token_num,
                        max_pos_embeddings)
                else:
                    raise ValueError(
                        chunked_processing_error_msg.format(
                            length_type="maximum position embeddings length",
                            max_length_value=max_pos_embeddings,
                            token_num=token_num))

            return TextTokensPrompt(prompt=input_text,
                                    prompt_token_ids=input_ids)

        # For other request types, use the parent's implementation
        return super()._validate_input(request, input_ids, input_text)

    def _is_text_tokens_prompt(self, prompt) -> bool:
        """Check if a prompt is a TextTokensPrompt (has prompt_token_ids)."""
        return (isinstance(prompt, dict) and "prompt_token_ids" in prompt
                and "prompt_embeds" not in prompt)

    @override
    async def _prepare_generators(
        self,
        ctx: ServeContext,
    ) -> Optional[ErrorResponse]:
        """Override to support chunked processing."""
        ctx = cast(EmbeddingServeContext, ctx)
        generators: list[AsyncGenerator[Union[RequestOutput,
                                              PoolingRequestOutput],
                                        None]] = []

        try:
            trace_headers = (None if ctx.raw_request is None else await
                             self._get_trace_headers(ctx.raw_request.headers))

            if not hasattr(ctx.request, "to_pooling_params"):
                return self.create_error_response(
                    "Request type does not support pooling parameters")

            pooling_params = ctx.request.to_pooling_params()

            # Verify and set the task for pooling params
            try:
                pooling_params.verify("embed", self.model_config)
            except ValueError as e:
                return self.create_error_response(str(e))

            if ctx.engine_prompts is None:
                return self.create_error_response(
                    "Engine prompts not available")

            if ctx.request_prompts is None:
                return self.create_error_response(
                    "Request prompts not available")

            # Check if we should use chunked processing
            use_chunked = self._should_use_chunked_processing(ctx.request)

            for i, engine_prompt in enumerate(ctx.engine_prompts):
                request_prompt = ctx.request_prompts[i]

                # Check if this specific prompt needs chunked processing
                max_pos_embeddings = self._get_max_position_embeddings()
                if (use_chunked
                        and self._is_text_tokens_prompt(request_prompt)):
                    # Cast to TextTokensPrompt since we've
                    # verified prompt_token_ids
                    text_tokens_prompt = cast(TextTokensPrompt, request_prompt)
                    if len(text_tokens_prompt["prompt_token_ids"]
                           ) > max_pos_embeddings:
                        # Use chunked processing for this prompt
                        chunk_generators = await self._process_chunked_request(
                            ctx, text_tokens_prompt, pooling_params,
                            trace_headers, i)
                        generators.extend(chunk_generators)
                        continue

                # Normal processing for short prompts or non-token prompts
                request_id_item = f"{ctx.request_id}-{i}"

                self._log_inputs(request_id_item,
                                 request_prompt,
                                 params=pooling_params,
                                 lora_request=ctx.lora_request)

                # Mypy has an existing bug related to inferring the variance
                # of TypedDicts with `builtins.enumerate`:
                # https://github.com/python/mypy/issues/8586#issuecomment-2867698435
                engine_prompt = cast(
                    Union[EngineTokensPrompt, EngineEmbedsPrompt],
                    engine_prompt)
                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=ctx.lora_request,
                    trace_headers=trace_headers,
                    priority=getattr(ctx.request, "priority", 0),
                )

                generators.append(generator)

            from vllm.utils import merge_async_iterators
            ctx.result_generator = merge_async_iterators(*generators)

            return None

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    @override
    async def _collect_batch(
        self,
        ctx: ServeContext,
    ) -> Optional[ErrorResponse]:
        """Collect and aggregate batch results
        with support for chunked processing.
        
        For chunked requests, performs online aggregation to 
        minimize memory usage.
        For regular requests, collects results normally.
        """
        ctx = cast(EmbeddingServeContext, ctx)
        try:
            if ctx.engine_prompts is None:
                return self.create_error_response(
                    "Engine prompts not available")
            # Check if we used chunked processing
            use_chunked = self._should_use_chunked_processing(ctx.request)

            if not use_chunked:
                return await super()._collect_batch(ctx=ctx)
            if ctx.request_prompts is None:
                return self.create_error_response(
                    "Request prompts not available")

            if ctx.result_generator is None:
                return self.create_error_response(
                    "Result generator not available")

            # Check if we used chunked processing
            use_chunked = self._should_use_chunked_processing(ctx.request)

            if use_chunked:
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

                            # Initialize aggregator for this prompt if needed
                            if prompt_idx not in prompt_aggregators:
                                prompt_aggregators[prompt_idx] = {
                                    'weighted_sum':
                                    None,
                                    'total_weight':
                                    0,
                                    'chunk_count':
                                    0,
                                    'request_id':
                                    result.request_id.split("-chunk-")[0]
                                }

                            aggregator = prompt_aggregators[prompt_idx]

                            # MEAN pooling with online weighted averaging
                            # Ensure result is PoolingRequestOutput
                            # for embedding processing
                            if not isinstance(result, PoolingRequestOutput):
                                return self.create_error_response(
                                    f"Expected PoolingRequestOutput for "
                                    f"chunked embedding, got "
                                    f"{type(result).__name__}")

                            # Handle both PoolingOutput and
                            # EmbeddingOutput types
                            if hasattr(result.outputs, 'data'):
                                # PoolingOutput case
                                embedding_data = result.outputs.data
                            elif hasattr(result.outputs, 'embedding'):
                                # EmbeddingOutput case -
                                # convert embedding list to tensor
                                embedding_data = result.outputs.embedding
                            else:
                                return self.create_error_response(
                                    f"Unsupported output type: "
                                    f"{type(result.outputs).__name__}")

                            if not isinstance(embedding_data, torch.Tensor):
                                embedding_data = torch.tensor(
                                    embedding_data, dtype=torch.float32)

                            if result.prompt_token_ids is None:
                                return self.create_error_response(
                                    "prompt_token_ids cannot be None for "
                                    "chunked processing")
                            weight = len(result.prompt_token_ids)

                            weighted_embedding = embedding_data.to(
                                dtype=torch.float32) * weight

                            if aggregator['weighted_sum'] is None:
                                # First chunk
                                aggregator['weighted_sum'] = weighted_embedding
                            else:
                                # Accumulate
                                current_sum = aggregator['weighted_sum']
                                if isinstance(current_sum, torch.Tensor):
                                    aggregator['weighted_sum'] = (
                                        current_sum + weighted_embedding)

                            total_weight = aggregator['total_weight']
                            if isinstance(total_weight, (int, float)):
                                aggregator['total_weight'] = (total_weight +
                                                              weight)

                            chunk_count = aggregator['chunk_count']
                            if isinstance(chunk_count, int):
                                aggregator['chunk_count'] = chunk_count + 1

                        except (ValueError, IndexError):
                            return self.create_error_response(
                                f"Invalid chunk request ID format: "
                                f"{result.request_id}")
                    else:
                        # Non-chunked result
                        try:
                            prompt_idx = int(result.request_id.split("-")[-1])
                            short_prompts_results[prompt_idx] = cast(
                                PoolingRequestOutput, result)
                        except ValueError:
                            return self.create_error_response(
                                f"Invalid request ID "
                                f"format: {result.request_id}")

                # Finalize aggregated results
                final_res_batch: list[Union[PoolingRequestOutput,
                                            EmbeddingRequestOutput]] = []
                num_prompts = len(ctx.engine_prompts)

                for prompt_idx in range(num_prompts):
                    if prompt_idx in prompt_aggregators:
                        # Finalize MEAN aggregation for this chunked prompt
                        aggregator = prompt_aggregators[prompt_idx]

                        weighted_sum = aggregator['weighted_sum']
                        total_weight = aggregator['total_weight']

                        if (weighted_sum is not None
                                and isinstance(weighted_sum, torch.Tensor)
                                and isinstance(total_weight, (int, float))
                                and total_weight > 0):

                            # Compute final mean embedding
                            final_embedding = weighted_sum / total_weight

                            # Create a PoolingRequestOutput
                            # for the aggregated result
                            pooling_output_data = PoolingOutput(
                                data=final_embedding)

                            # Get original prompt token IDs for this prompt
                            original_prompt = ctx.request_prompts[prompt_idx]
                            if not self._is_text_tokens_prompt(
                                    original_prompt):
                                return self.create_error_response(
                                    f"Chunked prompt {prompt_idx} is not a "
                                    f"TextTokensPrompt")

                            original_token_ids = cast(
                                TextTokensPrompt,
                                original_prompt)["prompt_token_ids"]

                            pooling_request_output = PoolingRequestOutput(
                                request_id=aggregator['request_id'],
                                prompt_token_ids=original_token_ids,
                                outputs=pooling_output_data,
                                finished=True)

                            final_res_batch.append(pooling_request_output)
                        else:
                            return self.create_error_response(
                                f"Failed to aggregate chunks "
                                f"for prompt {prompt_idx}")
                    elif prompt_idx in short_prompts_results:
                        final_res_batch.append(
                            cast(PoolingRequestOutput,
                                 short_prompts_results[prompt_idx]))
                    else:
                        return self.create_error_response(
                            f"Result not found for prompt {prompt_idx}")

                ctx.final_res_batch = cast(
                    list[Union[RequestOutput, PoolingRequestOutput]],
                    final_res_batch)
            else:
                # Normal processing for non-chunked requests
                num_prompts = len(ctx.engine_prompts)
                normal_final_res_batch: list[
                    Optional[PoolingRequestOutput]] = [None] * num_prompts

                async for result_idx, result in ctx.result_generator:
                    if result_idx < num_prompts:
                        # Cast to PoolingRequestOutput for embedding results
                        normal_final_res_batch[result_idx] = cast(
                            PoolingRequestOutput, result)

                if None in normal_final_res_batch:
                    return self.create_error_response(
                        "Failed to generate results for all prompts")

                final_results = [
                    res for res in normal_final_res_batch if res is not None
                ]
                ctx.final_res_batch = cast(
                    list[Union[RequestOutput, PoolingRequestOutput]],
                    final_results)

            return None

        except Exception as e:
            return self.create_error_response(str(e))


class OpenAIServingEmbedding(EmbeddingMixin):
    request_id_prefix = "embd"

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

    async def create_embedding(
        self,
        request: EmbeddingRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[EmbeddingResponse, ErrorResponse]:
        """
        Embedding API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        model_name = self._get_model_name(request.model)
        request_id = (
            f"{self.request_id_prefix}-"
            f"{self._base_request_id(raw_request, request.request_id)}")

        ctx = EmbeddingServeContext(
            request=request,
            raw_request=raw_request,
            model_name=model_name,
            request_id=request_id,
            chat_template=self.chat_template,
            chat_template_content_format=self.chat_template_content_format,
        )

        return await super().handle(ctx)  # type: ignore

    @override
    def _validate_request(
        self,
        ctx: ServeContext[EmbeddingRequest],
    ) -> Optional[ErrorResponse]:
        if error := super()._validate_request(ctx):
            return error

        ctx.truncate_prompt_tokens = ctx.request.truncate_prompt_tokens

        return None

    @override
    def _create_pooling_params(
        self,
        ctx: ServeContext[EmbeddingRequest],
    ) -> Union[PoolingParams, ErrorResponse]:
        pooling_params = super()._create_pooling_params(ctx)
        if isinstance(pooling_params, ErrorResponse):
            return pooling_params

        try:
            pooling_params.verify("embed", self.model_config)
        except ValueError as e:
            return self.create_error_response(str(e))

        return pooling_params
