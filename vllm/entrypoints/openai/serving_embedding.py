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
                          PoolingRequestOutput, RequestOutput)

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
        derived_max_len = getattr(hf_config, 'max_position_embeddings', 512)

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
        if not (pooler_config is not None and getattr(
                pooler_config, 'enable_chunked_processing', False)):
            return False

        # Check pooling type compatibility for chunked processing
        pooling_type = getattr(pooler_config, 'pooling_type', None)
        if pooling_type:
            pooling_type_upper = pooling_type.upper()

            # For LAST and CLS pooling, chunked processing doesn't make
            # semantic sense because only the last/first chunk
            # contains the relevant token position
            if pooling_type_upper in ['LAST', 'CLS']:
                # Check if user explicitly allowed non-mean chunking
                allow_non_mean = getattr(pooler_config,
                                         'allow_non_mean_chunking', False)
                if not allow_non_mean:
                    logger.warning(
                        "Chunked processing with pooling type '%s' "
                        "is not recommended as it may produce semantically "
                        "incorrect results. %s pooling relies on specific "
                        "token positions that lose their meaning when the "
                        "sequence is chunked. Consider using MEAN pooling "
                        "or disable chunked processing. Set "
                        "'allow_non_mean_chunking: true' ",
                        "to override this warning.", pooling_type,
                        pooling_type_upper)
                    return False  # Disable chunked processing by default
                else:
                    logger.info(
                        "Using chunked processing with %s pooling "
                        "(explicitly enabled). Note: only the %s chunk "
                        "will be processed to avoid computational waste.",
                        pooling_type_upper,
                        "last" if pooling_type_upper == "LAST" else "first")

            # Warn about non-MEAN pooling types (for other pooling types)
            elif pooling_type_upper != 'MEAN':
                # Check if user explicitly allowed non-mean chunking
                allow_non_mean = getattr(pooler_config,
                                         'allow_non_mean_chunking', False)
                if not allow_non_mean:
                    logger.warning(
                        "Chunked processing with pooling type '%s' "
                        "may produce different results than non-chunked "
                        "processing. Only MEAN pooling is mathematically "
                        "equivalent when using weighted averaging aggregation. "
                        "For other pooling types, different aggregation "
                        "strategies will be used that approximate the original "
                        "behavior. Set 'allow_non_mean_chunking: true' "
                        "in pooler config to suppress this warning.",
                        pooling_type)
                    # Still allow it but with warning
                else:
                    logger.info(
                        "Using chunked processing with pooling type "
                        "'%s' (explicitly enabled)", pooling_type)

        return True

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

        # Check pooling type to optimize chunk processing
        pooler_config = getattr(self.model_config, 'pooler_config', None)
        pooling_type = getattr(pooler_config, 'pooling_type', 'MEAN')
        if pooling_type:
            pooling_type = pooling_type.upper()

            # For LAST pooling, only process the last chunk
        # For CLS pooling, only process the first chunk
        if pooling_type == 'LAST':
            chunks_to_process = [chunks[-1]]
            chunk_indices = [len(chunks) - 1]
            logger.info("LAST pooling: processing only the last chunk")
        elif pooling_type == 'CLS':
            chunks_to_process = [chunks[0]]
            chunk_indices = [0]
            logger.info("CLS pooling: processing only the first chunk")
        else:
            # For MEAN and other pooling types, process all chunks
            chunks_to_process = chunks
            chunk_indices = list(range(len(chunks)))
            logger.info("Using chunked processing for MEAN pooling")

        for i, (chunk_idx, chunk_tokens) in enumerate(
                zip(chunk_indices, chunks_to_process)):
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
                                # Get pooling type to determine
                                # aggregation strategy
                                pooler_config = getattr(
                                    self.model_config, 'pooler_config', None)
                                pooling_type = getattr(pooler_config,
                                                       'pooling_type', 'MEAN')
                                if pooling_type:
                                    pooling_type = pooling_type.upper()

                                prompt_aggregators[prompt_idx] = {
                                    'pooling_type':
                                    pooling_type,
                                    'weighted_sum':
                                    None,
                                    'total_weight':
                                    0,
                                    'first_result':
                                    None,
                                    'last_result':
                                    None,
                                    'chunk_count':
                                    0,
                                    'request_id':
                                    result.request_id.split("-chunk-")[0]
                                }

                            aggregator = prompt_aggregators[prompt_idx]
                            pooling_type = aggregator['pooling_type']

                            # Handle different pooling types with
                            # online aggregation
                            if pooling_type == 'MEAN':
                                # Online weighted averaging
                                # Ensure result is PoolingRequestOutput
                                # for embedding processing
                                if not isinstance(result,
                                                  PoolingRequestOutput):
                                    return self.create_error_response(
                                        f"Expected PoolingRequestOutput for "
                                        f"chunked embedding, got "
                                        f"{type(result).__name__}")

                                embedding_data = result.outputs.data
                                if not isinstance(embedding_data,
                                                  torch.Tensor):
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
                                    aggregator[
                                        'weighted_sum'] = weighted_embedding
                                else:
                                    # Accumulate
                                    current_sum = aggregator['weighted_sum']
                                    if isinstance(current_sum, torch.Tensor):
                                        aggregator['weighted_sum'] = (
                                            current_sum + weighted_embedding)

                                total_weight = aggregator['total_weight']
                                if isinstance(total_weight, (int, float)):
                                    aggregator['total_weight'] = (
                                        total_weight + weight)

                            elif pooling_type == 'LAST':
                                # Keep only the
                                # last result (highest chunk index)
                                if not isinstance(result,
                                                  PoolingRequestOutput):
                                    return self.create_error_response(
                                        f"Expected PoolingRequestOutput for "
                                        f"chunked embedding, got "
                                        f"{type(result).__name__}")

                                chunk_idx = int(parts[parts.index("chunk") +
                                                      1])
                                last_chunk_idx = aggregator.get(
                                    'last_chunk_idx', -1)
                                # Ensure last_chunk_idx is an integer
                                # for comparison
                                if not isinstance(last_chunk_idx, int):
                                    last_chunk_idx = -1
                                if (aggregator['last_result'] is None
                                        or chunk_idx > last_chunk_idx):
                                    aggregator['last_result'] = result
                                    aggregator['last_chunk_idx'] = chunk_idx

                            elif pooling_type == 'CLS':
                                # Keep only the first result (chunk index 0)
                                if not isinstance(result,
                                                  PoolingRequestOutput):
                                    return self.create_error_response(
                                        f"Expected PoolingRequestOutput for "
                                        f"chunked embedding, got "
                                        f"{type(result).__name__}")

                                chunk_idx = int(parts[parts.index("chunk") +
                                                      1])
                                if chunk_idx == 0:
                                    aggregator['first_result'] = result

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
                                f"Invalid request ID format: "
                                f"{result.request_id}")

                # Build final result batch
                final_res_batch = []

                for prompt_idx, request_prompt in enumerate(
                        ctx.request_prompts):
                    if prompt_idx in prompt_aggregators:
                        # Finalize aggregation for this chunked prompt
                        aggregator = prompt_aggregators[prompt_idx]
                        pooling_type = aggregator['pooling_type']

                        if pooling_type == 'MEAN':
                            # Finalize weighted average
                            weighted_sum = aggregator['weighted_sum']
                            total_weight = aggregator['total_weight']
                            if (weighted_sum is not None
                                    and isinstance(weighted_sum, torch.Tensor)
                                    and isinstance(total_weight, (int, float))
                                    and total_weight > 0):
                                final_embedding = weighted_sum / total_weight

                                # Create aggregated result
                                from vllm.outputs import PoolingOutput
                                aggregated_output = PoolingOutput(
                                    data=final_embedding)

                                # Get original prompt token ids
                                if self._is_text_tokens_prompt(request_prompt):
                                    text_tokens_prompt = cast(
                                        TextTokensPrompt, request_prompt)
                                    original_token_ids = text_tokens_prompt[
                                        "prompt_token_ids"]
                                else:
                                    return self.create_error_response(
                                        f"Chunked prompt {prompt_idx} is not a "
                                        f"text tokens prompt")

                                # Ensure request_id is string
                                request_id = aggregator['request_id']
                                if not isinstance(request_id, str):
                                    return self.create_error_response(
                                        f"Invalid request_id type: "
                                        f"{type(request_id)}")

                                aggregated_result = PoolingRequestOutput(
                                    request_id=request_id,
                                    outputs=aggregated_output,
                                    prompt_token_ids=original_token_ids,
                                    finished=True,
                                )
                                final_res_batch.append(aggregated_result)
                            else:
                                return self.create_error_response(
                                    f"No valid aggregation data for prompt "
                                    f"{prompt_idx}")

                        elif pooling_type == 'LAST':
                            if aggregator['last_result'] is not None:
                                # Use the last chunk result
                                last_result = aggregator['last_result']
                                if not isinstance(last_result,
                                                  PoolingRequestOutput):
                                    return self.create_error_response(
                                        f"Expected PoolingRequestOutput for "
                                        f"last_result, got "
                                        f"{type(last_result).__name__}")

                                if self._is_text_tokens_prompt(request_prompt):
                                    text_tokens_prompt = cast(
                                        TextTokensPrompt, request_prompt)
                                    original_token_ids = text_tokens_prompt[
                                        "prompt_token_ids"]

                                    # Ensure request_id is string
                                    request_id = aggregator['request_id']
                                    if not isinstance(request_id, str):
                                        return self.create_error_response(
                                            f"Invalid request_id type: "
                                            f"{type(request_id)}")

                                    aggregated_result = PoolingRequestOutput(
                                        request_id=request_id,
                                        outputs=last_result.outputs,
                                        prompt_token_ids=original_token_ids,
                                        finished=True,
                                    )
                                    final_res_batch.append(aggregated_result)
                                else:
                                    return self.create_error_response(
                                        f"Chunked prompt {prompt_idx} is not a "
                                        f"text tokens prompt")
                            else:
                                return self.create_error_response(
                                    f"No LAST result found for prompt "
                                    f"{prompt_idx}")

                        elif pooling_type == 'CLS':
                            if aggregator['first_result'] is not None:
                                # Use the first chunk result
                                first_result = aggregator['first_result']
                                if not isinstance(first_result,
                                                  PoolingRequestOutput):
                                    return self.create_error_response(
                                        f"Expected PoolingRequestOutput for "
                                        f"first_result, got "
                                        f"{type(first_result).__name__}")

                                if self._is_text_tokens_prompt(request_prompt):
                                    text_tokens_prompt = cast(
                                        TextTokensPrompt, request_prompt)
                                    original_token_ids = text_tokens_prompt[
                                        "prompt_token_ids"]

                                    # Ensure request_id is string
                                    request_id = aggregator['request_id']
                                    if not isinstance(request_id, str):
                                        return self.create_error_response(
                                            f"Invalid request_id type: "
                                            f"{type(request_id)}")

                                    aggregated_result = PoolingRequestOutput(
                                        request_id=request_id,
                                        outputs=first_result.outputs,
                                        prompt_token_ids=original_token_ids,
                                        finished=True,
                                    )
                                    final_res_batch.append(aggregated_result)
                                else:
                                    return self.create_error_response(
                                        f"Chunked prompt {prompt_idx} is not a "
                                        f"text tokens prompt")
                            else:
                                return self.create_error_response(
                                    f"No CLS result found for prompt "
                                    f"{prompt_idx}")
                        else:
                            return self.create_error_response(
                                f"Unsupported pooling type for chunked "
                                f"processing: {pooling_type}")

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
