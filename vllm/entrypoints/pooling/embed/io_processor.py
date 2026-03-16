# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from typing import Any, cast

import torch

from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.inputs.data import ProcessorInputs, token_inputs
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.utils.collection_utils import chunk_list

logger = logging.getLogger(__name__)


class EmbedIOProcessor(PoolingIOProcessor):
    name = "embedding"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model_config.pooler_config is not None

        self.pooler_config = self.model_config.pooler_config
        self.enable_chunked_processing = self.pooler_config.enable_chunked_processing

        # Load task instructions from HF config or sentence-transformers config
        self.task_instructions: dict[str, str] | None = self._load_task_instructions(
            self.model_config.hf_config
        ) or self._load_st_prompts(self.model_config.model, self.model_config.revision)
        if self.task_instructions:
            logger.info(
                "Loaded prompt prefixes for input_type: %s",
                list(self.task_instructions.keys()),
            )

    @staticmethod
    def _load_task_instructions(hf_config: Any) -> dict[str, str] | None:
        """Extract ``task_instructions`` from the HF model config."""
        ti = getattr(hf_config, "task_instructions", None)
        if not isinstance(ti, dict) or not ti:
            return None
        return {k: v for k, v in ti.items() if isinstance(v, str)}

    @staticmethod
    def _load_st_prompts(
        model: str | Any,
        revision: str | None,
    ) -> dict[str, str] | None:
        """Load ``prompts`` from ``config_sentence_transformers.json``."""
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        try:
            cfg = get_hf_file_to_dict(
                "config_sentence_transformers.json", str(model), revision
            )
        except (ValueError, OSError):
            return None

        if cfg is None:
            return None
        prompts = cfg.get("prompts")
        if not isinstance(prompts, dict) or not prompts:
            return None
        return {k: v for k, v in prompts.items() if isinstance(v, str)}

    #################################################################
    # Long Text Embedding with Chunked Processing
    # PTAL: examples/pooling/embed/openai_embedding_long_text

    def pre_process_online(self, ctx: PoolingServeContext):
        super().pre_process_online(ctx)

        if not self.enable_chunked_processing:
            return None

        if ctx.engine_prompts is None:
            raise ValueError("Engine prompts not available")

        ctx.intermediates = ctx.engine_prompts
        request_id = ctx.request_id
        max_model_len = self.model_config.max_model_len
        chunked_engine_prompts: list[ProcessorInputs] = []
        prompt_request_ids: list[str] = []
        for prompt_idx, engine_prompt in enumerate(ctx.engine_prompts):
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
                chunked_engine_prompts.append(
                    token_inputs(prompt_token_ids=chunk_tokens)
                )
                prompt_request_ids.append(
                    f"{request_id}-prompt-{prompt_idx}-chunk-{chunk_idx}"
                )

        ctx.engine_prompts = chunked_engine_prompts
        ctx.prompt_request_ids = prompt_request_ids
        return None

    def post_process_online(
        self,
        ctx: PoolingServeContext,
    ):
        if ctx.final_res_batch is None:
            raise ValueError("Final response batch not available")

        if not self.enable_chunked_processing:
            return super().post_process_online(ctx)

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
                if result.prompt_token_ids is None:
                    raise ValueError(
                        "prompt_token_ids cannot be None for chunked processing"
                    )

                weight = len(result.prompt_token_ids)
                embedding_data = result.outputs.data
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
            raise ValueError("Original prompts inputs not available")

        original_engine_prompts = cast(list[ProcessorInputs], ctx.intermediates)
        num_prompts = len(original_engine_prompts)

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
                    original_prompt = original_engine_prompts[prompt_idx]
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

        ctx.final_res_batch = final_res_batch
        return None
