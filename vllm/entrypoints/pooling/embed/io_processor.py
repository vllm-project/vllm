# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any, cast

import torch

from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
)
from vllm.entrypoints.pooling.typing import EngineInputs, PoolingServeContext
from vllm.inputs.data import ProcessorInputs, PromptType, token_inputs
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.utils.collection_utils import chunk_list


class EmbedIOProcessor(PoolingIOProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model_config.pooler_config is not None

        self.pooler_config = self.model_config.pooler_config
        self.enable_chunked_processing = self.pooler_config.enable_chunked_processing

    #################################################################
    # online APIs

    def pre_process_online(self, ctx: PoolingServeContext):
        request: EmbeddingCompletionRequest | EmbeddingChatRequest = ctx.request

        if isinstance(request, EmbeddingChatRequest):
            self._validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )
            _, engine_prompts = self._preprocess_chat_online(
                request,
                request.messages,
                default_template=self.chat_template,
                default_template_content_format=self.chat_template_content_format,
                default_template_kwargs=None,
            )
        elif isinstance(request, EmbeddingCompletionRequest):
            engine_prompts = self._preprocess_completion_online(
                request,
                prompt_input=request.input,
                prompt_embeds=None,
            )
        else:
            raise ValueError("Invalid classification request type")
        return self._maybe_apply_chunked_processing_pre_process_online(
            ctx, engine_prompts
        )

    def post_process_online(
        self,
        ctx: PoolingServeContext,
    ) -> list[PoolingRequestOutput]:
        return self._maybe_apply_chunked_processing_post_process_online(ctx)

    #################################################################
    # offline APIs

    def pre_process_offline(
        self,
        prompts: PromptType | Sequence[PromptType],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[ProcessorInputs]:
        return self._preprocess_completion_offline(
            prompts=prompts, tokenization_kwargs=tokenization_kwargs
        )

    #################################################################
    # Long Text Embedding with Chunked Processing
    # PTAL: examples/pooling/embed/openai_embedding_long_text

    def _maybe_apply_chunked_processing_pre_process_online(
        self, ctx: PoolingServeContext, engine_prompts: list[ProcessorInputs]
    ):
        engine_inputs = [
            EngineInputs(engine_prompt=prompt) for prompt in engine_prompts
        ]

        if not self.enable_chunked_processing:
            ctx.engine_inputs = engine_inputs
            return None

        ctx.intermediates = engine_inputs
        request_id = ctx.request_id
        max_model_len = self.model_config.max_model_len
        chunked_engine_inputs: list[EngineInputs] = []
        for prompt_idx, engine_prompt in enumerate(engine_prompts):
            token_ids = engine_prompt.get("prompt_token_ids", None)
            if token_ids is None:
                raise NotImplementedError(
                    "Long Text Embedding with Chunked Processing does "
                    "not support EmbedsPrompt and EncoderDecoderInputs."
                )

            prompt_token_ids = cast(list[int], token_ids)

            for chunk_idx, chunk_tokens in enumerate(
                chunk_list(prompt_token_ids, max_model_len)
            ):
                chunked_engine_inputs.append(
                    EngineInputs(
                        engine_prompt=token_inputs(prompt_token_ids=chunk_tokens),
                        request_id_item=f"{request_id}-prompt-{prompt_idx}-chunk-{chunk_idx}",
                    )
                )
        ctx.engine_inputs = chunked_engine_inputs
        return None

    def _maybe_apply_chunked_processing_post_process_online(
        self,
        ctx: PoolingServeContext,
    ) -> list[PoolingRequestOutput]:
        if ctx.final_res_batch is None:
            raise ValueError("Final response batch not available")

        if not self.enable_chunked_processing:
            return ctx.final_res_batch

        # Online aggregation for chunked requests to
        # minimize memory usage
        # Track aggregation state for each prompt
        prompt_aggregators: dict[int, dict[str, Any]] = {}
        short_prompts_results: dict[int, PoolingRequestOutput] = {}
        for result_idx, result in enumerate(ctx.final_res_batch):
            if "-chunk-" not in result.request_id:
                # Non-chunked result - extract prompt_idx from request_id
                parts = result.request_id.split("-")
                try:
                    # Last part should be prompt index
                    prompt_idx = int(parts[-1])
                except (ValueError, IndexError):
                    prompt_idx = result_idx  # Fallback to result_idx

                short_prompts_results[prompt_idx] = result
            else:
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
                    raise ValueError(
                        f"Expected PoolingRequestOutput for "
                        f"chunked embedding, got "
                        f"{type(result).__name__}"
                    )

                embedding_data = result.outputs.data

                if result.prompt_token_ids is None:
                    raise ValueError(
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

        if ctx.intermediates is None:
            raise ValueError("Original engine inputs not available")

        original_engine_inputs = cast(list[EngineInputs], ctx.intermediates)
        num_prompts = len(original_engine_inputs)

        # Finalize aggregated results
        final_res_batch: list[PoolingRequestOutput] = []
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
                    original_prompt = original_engine_inputs[prompt_idx].engine_prompt
                    token_ids = original_prompt.get("prompt_token_ids", None)
                    if token_ids is None:
                        raise NotImplementedError(
                            "Long Text Embedding with Chunked Processing does "
                            "not support EmbedsPrompt and EncoderDecoderInputs."
                        )

                    original_token_ids = cast(list[int], token_ids)
                    pooling_request_output = PoolingRequestOutput(
                        request_id=aggregator["request_id"],
                        prompt_token_ids=original_token_ids,
                        outputs=pooling_output_data,
                        num_cached_tokens=0,
                        finished=True,
                    )

                    final_res_batch.append(pooling_request_output)
                else:
                    raise ValueError(
                        f"Failed to aggregate chunks for prompt {prompt_idx}"
                    )
            elif prompt_idx in short_prompts_results:
                final_res_batch.append(short_prompts_results[prompt_idx])
            else:
                raise ValueError(f"Result not found for prompt {prompt_idx}")

        return final_res_batch
