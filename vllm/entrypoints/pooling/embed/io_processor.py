# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.pooling.base.io_processor import EngineInputs, PoolingIOProcessor
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
)
from vllm.inputs.data import ProcessorInputs, PromptType, TokensPrompt
from vllm.outputs import PoolingRequestOutput
from vllm.renderers.inputs import DecoderOnlyTokPrompt, TokPrompt
from vllm.utils.collection_utils import chunk_list


class EmbedIOProcessor(PoolingIOProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model_config.pooler_config is not None

        self.pooler_config = self.model_config.pooler_config
        self.enable_chunked_processing = self.pooler_config.enable_chunked_processing

    def pre_process_online(
        self, request: EmbeddingCompletionRequest | EmbeddingChatRequest
    ) -> list[EngineInputs] | None:
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
            request_id=request.request_id, engine_prompts=engine_prompts
        )

    def post_process(
        self, outputs: list[PoolingRequestOutput]
    ) -> list[PoolingRequestOutput]:
        return self._maybe_apply_chunked_processing_post_process_online(outputs)

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
        self,
        request_id: str,
        engine_prompts: list[TokPrompt],
    ) -> list[EngineInputs]:
        if not self.enable_chunked_processing:
            return [EngineInputs(engine_prompt=prompt) for prompt in engine_prompts]

        max_model_len = self.model_config.max_model_len
        chunked_engine_prompts: list[EngineInputs] = []
        for prompt_idx, engine_prompt in enumerate(engine_prompts):
            # "EncoderDecoderInputs" has no key "prompt_token_ids"
            assert isinstance(engine_prompt, DecoderOnlyTokPrompt)
            prompt_token_ids = engine_prompt["prompt_token_ids"]

            for chunk_idx, chunk_tokens in enumerate(
                chunk_list(prompt_token_ids, max_model_len)
            ):
                chunked_engine_prompts.append(
                    EngineInputs(
                        engine_prompt=TokensPrompt(prompt_token_ids=chunk_tokens),
                        request_id_item=f"{request_id}-prompt-{prompt_idx}-chunk-{chunk_idx}",
                    )
                )
        return chunked_engine_prompts

    def _maybe_apply_chunked_processing_post_process_online(
        self, outputs: list[PoolingRequestOutput]
    ) -> list[PoolingRequestOutput]:
        if not self.enable_chunked_processing:
            return outputs
        return outputs
