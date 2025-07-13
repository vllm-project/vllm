# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
from collections.abc import AsyncGenerator
from typing import Final, Literal, Optional, Union, cast

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
                                              EmbeddingCompletionRequest,
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

    async def _preprocess(
        self,
        ctx: ServeContext,
    ) -> Optional[ErrorResponse]:
        ctx = cast(EmbeddingServeContext, ctx)
        try:
            (
                ctx.lora_request,
                ctx.prompt_adapter_request,
            ) = self._maybe_get_adapters(ctx.request)

            tokenizer = await self.engine_client.get_tokenizer(ctx.lora_request
                                                               )

            if ctx.prompt_adapter_request is not None:
                raise NotImplementedError("Prompt adapter is not supported "
                                          "for embedding models")

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
        """Get the model's effective maximum sequence length for chunking.
        
        This uses the same logic as vLLM's _get_and_verify_max_len to determine
        the actual sequence length limit,
        considering both model config and tokenizer config.
        """
        hf_config = self.model_config.hf_config

        # Start with max_position_embeddings from model config
        derived_max_len = getattr(hf_config, 'max_position_embeddings', 2048)

        # Get tokenizer config for pooling models (embedding models)
        if self.model_config.runner_type == "pooling":
            from vllm.transformers_utils.config import try_get_tokenizer_config
            tokenizer_config = try_get_tokenizer_config(
                self.model_config.tokenizer,
                trust_remote_code=self.model_config.trust_remote_code,
                revision=self.model_config.tokenizer_revision)

            # Consider model_max_length in tokenizer_config
            # (same logic as _get_and_verify_max_len)
            if tokenizer_config:
                tokenizer_model_max_length = tokenizer_config.get(
                    'model_max_length', derived_max_len)
                derived_max_len = min(derived_max_len,
                                      tokenizer_model_max_length)

        return int(derived_max_len)

    def _should_use_chunked_processing(self, request) -> bool:
        """Check if chunked processing should be used for this request."""
        if not isinstance(request,
                          (EmbeddingChatRequest, EmbeddingCompletionRequest)):
            return False

        pooler_config = getattr(self.model_config, 'pooler_config', None)
        return (pooler_config is not None
                and getattr(pooler_config, 'enable_chunked_processing', False))

    def _chunk_token_ids(self, token_ids: list[int],
                         chunk_size: int) -> list[list[int]]:
        """Split token IDs into chunks of specified size."""
        if len(token_ids) <= chunk_size:
            return [token_ids]

        chunks = []
        for i in range(0, len(token_ids), chunk_size):
            chunk = token_ids[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

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
        chunks = self._chunk_token_ids(token_ids, max_pos_embeddings)

        logger.info(
            "Split input of %s tokens into %s chunks (max_chunk_size: %s)",
            len(token_ids), len(chunks), max_pos_embeddings)

        for chunk_idx, chunk_tokens in enumerate(chunks):
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
                             lora_request=ctx.lora_request,
                             prompt_adapter_request=ctx.prompt_adapter_request)

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

    async def _aggregate_chunked_results(
        self,
        ctx: EmbeddingServeContext,
        chunk_results: list[PoolingRequestOutput],
        original_token_count: int,
        original_prompt_token_ids: Optional[list[int]] = None,
    ) -> PoolingRequestOutput:
        """Aggregate results from multiple chunks
        using vLLM-compatible weighted averaging."""
        if len(chunk_results) == 1:
            return chunk_results[0]

        # Extract embeddings and use vLLM's token counting approach
        chunk_embeddings = []
        chunk_weights = []

        for result in chunk_results:
            # PoolingRequestOutput.outputs is a PoolingOutput object
            if hasattr(result, 'outputs') and hasattr(result.outputs, 'data'):
                # Get the embedding tensor from PoolingOutput.data
                embedding_data = result.outputs.data
                if not isinstance(embedding_data, torch.Tensor):
                    embedding_data = torch.tensor(embedding_data,
                                                  dtype=torch.float32)
                chunk_embeddings.append(embedding_data)

                # Use actual effective token count
                # this is what vLLM uses internally
                effective_token_count = len(result.prompt_token_ids)
                chunk_weights.append(effective_token_count)

        if not chunk_embeddings:
            raise ValueError("No valid embeddings found in chunk results")

        # Simple weighted averaging compatible with vLLM's approach
        # This is similar to what MeanPool does for multiple sequences
        device = chunk_embeddings[0].device
        # Use float32 for precision, as done in vLLM's PoolerHead
        dtype = torch.float32

        # Weighted sum following vLLM's internal logic
        weighted_sum = torch.zeros_like(chunk_embeddings[0],
                                        dtype=dtype,
                                        device=device)
        total_weight = 0

        for embedding, weight in zip(chunk_embeddings, chunk_weights):
            embedding = embedding.to(dtype=dtype, device=device)
            weighted_sum += embedding * weight
            total_weight += weight

        # Final averaged embedding - let vLLM handle the rest
        aggregated_embedding = weighted_sum / total_weight

        # NOTE: Don't manually normalize here
        # let vLLM's PoolerHead handle normalization
        # based on the model's pooler_config.normalize setting.
        # This ensures consistency with vLLM's standard pooling behavior.

        # Create aggregated result using vLLM's standard output structure
        first_result = chunk_results[0]

        # Create new PoolingOutput with aggregated embedding
        aggregated_output = PoolingOutput(data=aggregated_embedding)

        # Preserve original prompt token ids for consistency
        result_prompt_token_ids = (original_prompt_token_ids
                                   if original_prompt_token_ids is not None
                                   else first_result.prompt_token_ids)

        aggregated_result = PoolingRequestOutput(
            request_id=first_result.request_id,
            outputs=aggregated_output,
            prompt_token_ids=result_prompt_token_ids,
            finished=True,
        )

        return aggregated_result

    def _validate_input(
        self,
        request,
        input_ids: list[int],
        input_text: str,
    ) -> TextTokensPrompt:
        """Override to support chunked processing for embedding requests."""
        token_num = len(input_ids)

        # Note: EmbeddingRequest doesn't have max_tokens
        if isinstance(request,
                      (EmbeddingChatRequest, EmbeddingCompletionRequest)):
            # Check if chunked processing is enabled for pooling models
            pooler_config = getattr(self.model_config, 'pooler_config', None)
            enable_chunked = (pooler_config is not None and getattr(
                pooler_config, 'enable_chunked_processing', False))

            # Get max_embed_len from pooler config if set
            max_embed_len = (pooler_config.max_embed_len if pooler_config
                             and pooler_config.max_embed_len else None)

            # Use max_position_embeddings for chunked processing decisions
            max_pos_embeddings = self._get_max_position_embeddings()

            # Determine the effective max length for validation
            if max_embed_len is not None:
                # Use max_embed_len for validation instead of max_model_len
                effective_max_len = max_embed_len
                validation_error_msg = (
                    f"This model's maximum embedding input length is "
                    f"{max_embed_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for embedding "
                    f"generation. Please reduce the length of the input.")
            else:
                # Fall back to max_model_len validation (original behavior)
                effective_max_len = self.max_model_len
                validation_error_msg = (
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for embedding "
                    f"generation. Please reduce the length of the input.")

            # Check if input exceeds effective max length
            if token_num > effective_max_len:
                raise ValueError(validation_error_msg)

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
                        f"This model's maximum position embeddings length is "
                        f"{max_pos_embeddings} tokens. However, you requested "
                        f"{token_num} tokens in the input for embedding "
                        f"generation. Please reduce the length of the input or "
                        f"enable chunked processing.")

            return TextTokensPrompt(prompt=input_text,
                                    prompt_token_ids=input_ids)

        # For other request types, use the parent's implementation
        return super()._validate_input(request, input_ids, input_text)

    def _is_text_tokens_prompt(self, prompt) -> bool:
        """Check if a prompt is a TextTokensPrompt (has prompt_token_ids)."""
        return (isinstance(prompt, dict) and "prompt_token_ids" in prompt
                and "prompt_embeds" not in prompt)

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

                self._log_inputs(
                    request_id_item,
                    request_prompt,
                    params=pooling_params,
                    lora_request=ctx.lora_request,
                    prompt_adapter_request=ctx.prompt_adapter_request)

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

    async def _collect_batch(
        self,
        ctx: ServeContext,
    ) -> Optional[ErrorResponse]:
        """Override to support chunked processing."""
        ctx = cast(EmbeddingServeContext, ctx)
        try:
            if ctx.engine_prompts is None:
                return self.create_error_response(
                    "Engine prompts not available")

            if ctx.request_prompts is None:
                return self.create_error_response(
                    "Request prompts not available")

            if ctx.result_generator is None:
                return self.create_error_response(
                    "Result generator not available")

            # Check if we used chunked processing
            use_chunked = self._should_use_chunked_processing(ctx.request)

            if use_chunked:
                # Efficient single-pass processing for chunked requests
                from collections import defaultdict

                # Group results by original prompt index
                grouped_results = defaultdict(list)
                short_prompts_results = {}

                async for result_idx, result in ctx.result_generator:
                    if "-chunk-" in result.request_id:
                        # Extract prompt_idx from chunked request_id
                        # e.g., from "req-id-prompt-2-chunk-0" -> 2
                        parts = result.request_id.split("-")
                        try:
                            prompt_idx = int(parts[parts.index("prompt") + 1])
                            grouped_results[prompt_idx].append(
                                cast(PoolingRequestOutput, result))
                        except (ValueError, IndexError):
                            return self.create_error_response(
                                f"Invalid chunk request ID format: "
                                f"{result.request_id}")
                    else:
                        # Extract prompt_idx from non-chunked request_id
                        # e.g., from "req-id-2" -> 2
                        try:
                            prompt_idx = int(result.request_id.split("-")[-1])
                            short_prompts_results[prompt_idx] = cast(
                                PoolingRequestOutput, result)
                        except ValueError:
                            return self.create_error_response(
                                f"Invalid request ID format: "
                                f"{result.request_id}")

                # Build final result batch in prompt order
                final_res_batch = []

                for prompt_idx, request_prompt in enumerate(
                        ctx.request_prompts):
                    if prompt_idx in grouped_results:
                        # This was a chunked prompt - aggregate results
                        chunk_results = grouped_results[prompt_idx]
                        if self._is_text_tokens_prompt(request_prompt):
                            text_tokens_prompt = cast(TextTokensPrompt,
                                                      request_prompt)
                            original_token_count = len(
                                text_tokens_prompt["prompt_token_ids"])
                            aggregated_result = await \
                                self._aggregate_chunked_results(
                                    ctx, chunk_results, original_token_count,
                                    text_tokens_prompt["prompt_token_ids"])
                            final_res_batch.append(aggregated_result)
                        else:
                            return self.create_error_response(
                                f"Chunked prompt {prompt_idx} is not a "
                                f"text tokens prompt")
                    elif prompt_idx in short_prompts_results:
                        # This was a short prompt
                        final_res_batch.append(
                            short_prompts_results[prompt_idx])
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
        request_id = (f"{self.request_id_prefix}-"
                      f"{self._base_request_id(raw_request)}")

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

        pooling_params = ctx.request.to_pooling_params()

        try:
            pooling_params.verify(self.model_config)
        except ValueError as e:
            return self.create_error_response(str(e))

        return None
