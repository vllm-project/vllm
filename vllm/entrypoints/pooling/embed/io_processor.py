# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL

from vllm import PoolingParams
from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    CustomChatCompletionMessageParam,
)
from vllm.inputs import EngineInput, tokens_input
from vllm.logger import init_logger
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.renderers import merge_kwargs
from vllm.renderers.hf import resolve_chat_template
from vllm.utils.collection_utils import chunk_list
from vllm.utils.mistral import is_mistral_tokenizer

from ..base.io_processor import PoolingIOProcessor
from ..scoring.io_processor import JinaRankingIOProcessorMixin
from ..typing import (
    ALLOfflineInputsContext,
    ChunkedEmbeddingMetadata,
    OfflineEncodeInputsContext,
    PoolingChatLikeRequest,
    PoolingCompletionLikeRequest,
    PoolingServeContext,
    RequestFactory,
)
from .protocol import (
    CohereEmbedContent,
    CohereEmbedInput,
    CohereEmbedRequest,
    EmbeddingBatchChatInputRequest,
    EmbeddingBatchChatRequest,
    EmbeddingChatInputRequest,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
)

logger = init_logger(__name__)


@dataclass
class _ChunkedPromptAggregator:
    weighted_sum: torch.Tensor | None = None
    total_weight: int = 0


class EmbedIOProcessor(PoolingIOProcessor):
    name = "embed"

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

    def pre_process_online(self, ctx: PoolingServeContext):
        if isinstance(ctx.request, CohereEmbedRequest):
            self._pre_process_cohere_online(ctx)
        elif isinstance(
            ctx.request,
            (
                EmbeddingChatRequest,
                EmbeddingBatchChatRequest,
                EmbeddingChatInputRequest,
                EmbeddingBatchChatInputRequest,
            ),
        ):
            self._pre_process_openai_chat_online(ctx)
        else:
            super().pre_process_online(ctx)

        if self.enable_chunked_processing:
            self._pre_process_chunked(ctx)

    def post_process_online(
        self,
        ctx: PoolingServeContext,
    ):
        if ctx.final_res_batch is None:
            raise ValueError("Final response batch not available")

        if not self.enable_chunked_processing:
            self._enforce_cohere_max_tokens(ctx)
            return super().post_process_online(ctx)

        self._post_process_chunked(ctx)
        self._enforce_cohere_max_tokens(ctx)

    #################################################################
    # Long Text Embedding with Chunked Processing
    # PTAL: examples/pooling/embed/openai_embedding_long_text
    #################################################################

    def _pre_process_chunked(self, ctx: PoolingServeContext) -> None:
        if ctx.engine_inputs is None:
            raise ValueError("Engine prompts not available")

        ctx.original_engine_inputs = ctx.engine_inputs
        request_id = ctx.request_id
        max_model_len = self.model_config.max_model_len
        chunked_engine_inputs: list[EngineInput] = []
        prompt_request_ids: list[str] = []
        chunked_embedding_metadata: list[ChunkedEmbeddingMetadata] = []
        for prompt_idx, engine_input in enumerate(ctx.engine_inputs):
            token_ids = engine_input.get("prompt_token_ids", None)
            if token_ids is None:
                raise NotImplementedError(
                    "Long Text Embedding with Chunked Processing does "
                    "not support EmbedsPrompt and EncoderDecoderInput."
                )

            prompt_token_ids = cast(list[int], token_ids)

            for chunk_idx, chunk_tokens in enumerate(
                chunk_list(prompt_token_ids, max_model_len)
            ):
                chunked_engine_inputs.append(
                    tokens_input(prompt_token_ids=chunk_tokens)
                )
                prompt_request_ids.append(
                    f"{request_id}-prompt-{prompt_idx}-chunk-{chunk_idx}"
                )
                chunked_embedding_metadata.append(
                    ChunkedEmbeddingMetadata(
                        prompt_index=prompt_idx,
                        chunk_index=chunk_idx,
                    )
                )

        ctx.engine_inputs = chunked_engine_inputs
        ctx.prompt_request_ids = prompt_request_ids
        ctx.chunked_embedding_metadata = chunked_embedding_metadata

        return None

    def _post_process_chunked(self, ctx: PoolingServeContext) -> None:
        # Online aggregation for chunked requests to
        # minimize memory usage
        # Track aggregation state for each prompt
        if ctx.chunked_embedding_metadata is None:
            raise ValueError("Chunked embedding metadata not available")
        if len(ctx.chunked_embedding_metadata) != len(ctx.final_res_batch):
            raise ValueError(
                "Chunked embedding metadata count does not match result count"
            )

        prompt_aggregators: dict[int, _ChunkedPromptAggregator] = {}
        for result, chunk_metadata in zip(
            ctx.final_res_batch, ctx.chunked_embedding_metadata
        ):
            prompt_idx = chunk_metadata.prompt_index
            aggregator = prompt_aggregators.setdefault(
                prompt_idx, _ChunkedPromptAggregator()
            )

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

            if aggregator.weighted_sum is None:
                # First chunk
                aggregator.weighted_sum = weighted_embedding
            else:
                # Accumulate
                aggregator.weighted_sum += weighted_embedding

            aggregator.total_weight += weight

        if ctx.original_engine_inputs is None:
            raise ValueError("Original engine inputs not available")

        original_engine_inputs = ctx.original_engine_inputs
        num_prompts = len(original_engine_inputs)

        # Finalize aggregated results
        final_res_batch: list[PoolingRequestOutput] = []
        for prompt_idx in range(num_prompts):
            if prompt_idx in prompt_aggregators:
                # Finalize MEAN aggregation for this chunked prompt
                aggregator = prompt_aggregators[prompt_idx]

                weighted_sum = aggregator.weighted_sum
                total_weight = aggregator.total_weight

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
                    original_prompt = original_engine_inputs[prompt_idx]
                    token_ids = original_prompt.get("prompt_token_ids", None)
                    if token_ids is None:
                        raise NotImplementedError(
                            "Long Text Embedding with Chunked Processing does "
                            "not support EmbedsPrompt and EncoderDecoderInput."
                        )

                    original_token_ids = cast(list[int], token_ids)
                    pooling_request_output = PoolingRequestOutput(
                        request_id=f"{ctx.request_id}-prompt-{prompt_idx}",
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
            else:
                raise ValueError(f"Result not found for prompt {prompt_idx}")

        ctx.final_res_batch = final_res_batch

        return None

    #################################################################
    # Cohere Request Preprocessing & Postprocessing
    #################################################################

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
        """Load ``task_instructions`` from ``config_sentence_transformers.json``."""
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

    @staticmethod
    def _mixed_input_to_messages(
        inp: CohereEmbedInput,
        *,
        task_prefix: str | None = None,
    ) -> list[ChatCompletionMessageParam]:
        """Build chat messages from a mixed text+image input.

        When *task_prefix* is given, it is used as the system prompt.
        """
        messages: list[ChatCompletionMessageParam] = []
        if task_prefix is not None:
            messages.append(
                CustomChatCompletionMessageParam(
                    role="system",
                    content=[
                        ChatCompletionContentPartTextParam(
                            type="text", text=task_prefix
                        )
                    ],
                )
            )

        parts: list[ChatCompletionContentPartParam] = []
        for item in inp.content:
            if item.type == "text" and item.text is not None:
                parts.append(
                    ChatCompletionContentPartTextParam(type="text", text=item.text)
                )
            elif item.type == "image_url" and item.image_url is not None:
                parts.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(url=item.image_url["url"]),
                    )
                )
        messages.append(CustomChatCompletionMessageParam(role="user", content=parts))
        return messages

    @staticmethod
    def _check_cohere_max_tokens(
        outputs: list[PoolingRequestOutput],
        max_tokens_check: int | None,
    ) -> None:
        """Raise if any output exceeds *max_tokens_check* tokens.

        Used to enforce ``truncate=NONE`` with an explicit ``max_tokens``:
        the pipeline runs without truncation and we reject afterwards.
        """
        if max_tokens_check is None:
            return
        for out in outputs:
            n = len(out.prompt_token_ids)
            if n > max_tokens_check:
                raise ValueError(
                    f"Input of {n} tokens exceeds max_tokens={max_tokens_check} "
                    "with truncate=NONE. Set truncate to END or START to "
                    "allow truncation."
                )

    @staticmethod
    def _resolve_cohere_truncation(
        request: CohereEmbedRequest,
    ) -> tuple[int | None, Literal["left", "right"] | None]:
        """Return ``(truncate_prompt_tokens, truncation_side)``."""
        if request.truncate == "NONE":
            return None, None
        if request.truncate == "START":
            tokens = request.max_tokens if request.max_tokens is not None else -1
            return tokens, "left"
        if request.max_tokens is not None:
            return request.max_tokens, None
        return -1, None

    def create_pooling_params(self, request):
        if isinstance(request, CohereEmbedRequest):
            return PoolingParams(
                task="embed",
                dimensions=request.output_dimension,
            )
        return super().create_pooling_params(request)

    def _pre_process_openai_chat_online(
        self,
        ctx: PoolingServeContext[
            EmbeddingChatRequest
            | EmbeddingBatchChatRequest
            | EmbeddingChatInputRequest
            | EmbeddingBatchChatInputRequest
        ],
    ) -> None:
        request = ctx.request
        self._validate_chat_template(
            request_chat_template=request.chat_template,
            chat_template_kwargs=request.chat_template_kwargs,
            trust_request_chat_template=self.trust_request_chat_template,
        )

        if isinstance(
            request, (EmbeddingBatchChatRequest, EmbeddingBatchChatInputRequest)
        ):
            all_messages = request.messages
        else:
            all_messages = [request.messages]
        ctx.engine_inputs = self._batch_render_openai_chat(request, all_messages)

    def _batch_render_openai_chat(
        self,
        request: (
            EmbeddingChatRequest
            | EmbeddingBatchChatRequest
            | EmbeddingChatInputRequest
            | EmbeddingBatchChatInputRequest
        ),
        all_messages: Sequence[list[ChatCompletionMessageParam]],
    ) -> list[EngineInput]:
        renderer = self.renderer
        mm_config = self.model_config.multimodal_config

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            self.chat_template,
            self.chat_template_content_format,
        ).with_defaults(
            merge_kwargs(
                None,
                dict(
                    tools=None,
                    tokenize=is_mistral_tokenizer(renderer.tokenizer),
                ),
            ),
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
        )

        _, engine_inputs = renderer.render_chat(
            all_messages,
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )
        return engine_inputs

    def _pre_process_cohere_online(self, ctx: PoolingServeContext) -> None:
        """Convert a ``CohereEmbedRequest`` into engine prompts.

        If a model has a chat template the task instruction are rendered
        as a system prompt. Otherwise they are just prepended to the input text.

        Images and mixed inputs are always batch-rendered through the chat
        template in one ``render_chat`` call.
        """
        request = ctx.request
        assert isinstance(request, CohereEmbedRequest)

        if request.texts is None and request.images is None and request.inputs is None:
            raise ValueError("One of texts, images, or inputs must be provided")

        truncate_prompt_tokens, truncation_side = self._resolve_cohere_truncation(
            request
        )
        input_type = request.input_type
        self._validate_input_type(input_type)

        if request.images is not None:
            input: list[CohereEmbedInput] = [
                CohereEmbedInput(
                    content=[
                        CohereEmbedContent(type="image_url", image_url={"url": uri})
                    ]
                )
                for uri in request.images
            ]
        elif request.inputs is not None:
            input = request.inputs
        else:
            texts = request.texts or []
            task_prefix = self._get_task_instruction_prefix(input_type)

            if task_prefix is None:
                ctx.engine_inputs = self._preprocess_cohere_text_completion(
                    request,
                    texts,
                    truncate_prompt_tokens,
                    truncation_side,
                )
                return

            all_messages = [
                self._mixed_input_to_messages(
                    CohereEmbedInput(
                        content=[CohereEmbedContent(type="text", text=text)]
                    ),
                    task_prefix=task_prefix,
                )
                for text in texts
            ]
            if self._has_chat_template():
                ctx.engine_inputs = self._batch_render_chat(
                    request,
                    all_messages,
                    truncate_prompt_tokens,
                    truncation_side,
                )
            else:
                ctx.engine_inputs = self._preprocess_cohere_text_completion(
                    request,
                    self._apply_task_instruction(texts, input_type),
                    truncate_prompt_tokens,
                    truncation_side,
                )
            return

        task_prefix = self._get_task_instruction_prefix(input_type)
        all_messages = [
            self._mixed_input_to_messages(inp, task_prefix=task_prefix) for inp in input
        ]
        ctx.engine_inputs = self._batch_render_chat(
            request, all_messages, truncate_prompt_tokens, truncation_side
        )

    def _has_chat_template(self) -> bool:
        return (
            resolve_chat_template(
                self.renderer.tokenizer,
                chat_template=self.chat_template,
                tools=None,
                model_config=self.model_config,
            )
            is not None
        )

    def _preprocess_cohere_text_completion(
        self,
        request: CohereEmbedRequest,
        texts: list[str],
        truncate_prompt_tokens: int | None,
        truncation_side: Literal["left", "right"] | None,
    ) -> list[EngineInput]:
        proxy = EmbeddingCompletionRequest(
            model=request.model,
            input=texts,
            dimensions=request.output_dimension,
            encoding_format="float",
            truncate_prompt_tokens=truncate_prompt_tokens,
            truncation_side=truncation_side,
        )
        return self._preprocess_cmpl_online(
            proxy, prompt_input=proxy.input, prompt_embeds=None
        )

    def _batch_render_chat(
        self,
        request: CohereEmbedRequest,
        all_messages: Sequence[list[ChatCompletionMessageParam]],
        truncate_prompt_tokens: int | None,
        truncation_side: Literal["left", "right"] | None,
    ) -> list[EngineInput]:
        """Batch-render multiple conversations through the chat template."""
        if not all_messages:
            return []

        proxy = EmbeddingChatRequest(
            model=request.model,
            messages=list(all_messages[0]),
            dimensions=request.output_dimension,
            encoding_format="float",
            truncate_prompt_tokens=truncate_prompt_tokens,
            truncation_side=truncation_side,
        )

        renderer = self.renderer
        mm_config = self.model_config.multimodal_config

        tok_params = proxy.build_tok_params(self.model_config)
        chat_params = proxy.build_chat_params(
            self.chat_template,
            self.chat_template_content_format,
        ).with_defaults(
            merge_kwargs(
                None,
                dict(
                    tools=None,
                    tokenize=is_mistral_tokenizer(renderer.tokenizer),
                ),
            ),
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
        )

        _, engine_inputs = renderer.render_chat(all_messages, chat_params, tok_params)
        return engine_inputs

    def _validate_input_type(self, input_type: str | None) -> None:
        """Raise if *input_type* is not supported by this model."""
        if input_type is None:
            return
        if self.task_instructions is None:
            raise ValueError(
                f"Unsupported input_type {input_type!r}. "
                "This model does not define any input_type task instructions."
            )
        if input_type not in self.task_instructions:
            supported = ", ".join(sorted(self.task_instructions))
            raise ValueError(
                f"Unsupported input_type {input_type!r}. Supported values: {supported}"
            )

    def _apply_task_instruction(
        self,
        texts: list[str],
        input_type: str | None,
    ) -> list[str]:
        """Prepend the task-instruction prefix for *input_type*.

        Returns *texts* unchanged when no matching prefix is configured.
        """
        prefix = self._get_task_instruction_prefix(input_type)
        if not prefix:
            return texts
        return [prefix + t for t in texts]

    def _get_task_instruction_prefix(self, input_type: str | None) -> str | None:
        """Return the task-instruction prefix for *input_type*, or ``None``."""
        if not self.task_instructions or input_type is None:
            return None
        return self.task_instructions.get(input_type) or None

    def _enforce_cohere_max_tokens(self, ctx: PoolingServeContext) -> None:
        if isinstance(ctx.request, CohereEmbedRequest):
            request = ctx.request
            if request.truncate == "NONE" and request.max_tokens is not None:
                self._check_cohere_max_tokens(ctx.final_res_batch, request.max_tokens)


class TokenEmbedIOProcessor(PoolingIOProcessor):
    name = "token_embed"


class JinaRankingTokenEmbedIOProcessor(
    TokenEmbedIOProcessor, JinaRankingIOProcessorMixin
):
    def pre_process_online(self, ctx: PoolingServeContext):
        request = ctx.request
        if isinstance(request, PoolingCompletionLikeRequest):
            prompts = request.input
            if not isinstance(prompts, Sequence) or len(prompts) < 2:
                raise ValueError("The JinaForRanking model requires at least 2 inputs.")

            text_prompts = self.ensure_str(prompts)

            # The JinaForRanking model concatenates docs first, then query.
            # Let's stay consistent with this novel design.
            prompt_input = self.format_docs_prompts_func(
                query=text_prompts[-1], docs=text_prompts[:-1]
            )

            engine_inputs = self._preprocess_cmpl_online(
                request,
                prompt_input=prompt_input,
                prompt_embeds=None,
            )
        elif isinstance(request, PoolingChatLikeRequest):
            raise ValueError("The JinaForRanking does not support chat Request.")
        else:
            raise ValueError(f"Invalid {self.name} request type")

        ctx.engine_inputs = engine_inputs

    def get_request_factory_offline(
        self, ctx: ALLOfflineInputsContext
    ) -> tuple[RequestFactory, int]:
        assert isinstance(ctx, OfflineEncodeInputsContext)
        if not isinstance(ctx.prompts, Sequence) or len(ctx.prompts) < 2:
            raise ValueError("The JinaForRanking model requires at least 2 inputs.")

        text_prompts = self.ensure_str(ctx.prompts)

        # The JinaForRanking model concatenates docs first, then query.
        # Let's stay consistent with this novel design.
        ctx.prompts = self.format_docs_prompts_func(
            query=text_prompts[-1], docs=text_prompts[:-1]
        )

        return super().get_request_factory_offline(ctx)
