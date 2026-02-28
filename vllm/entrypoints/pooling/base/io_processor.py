# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from typing import Final, TypeAlias

from vllm import TokensPrompt
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationChatRequest,
    ClassificationCompletionRequest,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
    PoolingChatRequest,
    PoolingCompletionRequest,
)
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest,
    ScoreDataRequest,
    ScoreQueriesDocumentsRequest,
    ScoreRequest,
    ScoreTextRequest,
)
from vllm.exceptions import VLLMValidationError
from vllm.renderers import BaseRenderer

PoolingCompletionLikeRequest: TypeAlias = (
    EmbeddingCompletionRequest
    | ClassificationCompletionRequest
    | RerankRequest
    | ScoreRequest
    | PoolingCompletionRequest
)

PoolingChatLikeRequest: TypeAlias = (
    EmbeddingChatRequest | ClassificationChatRequest | PoolingChatRequest
)

AnyPoolingRequest: TypeAlias = (
    PoolingCompletionLikeRequest | PoolingChatLikeRequest | IOProcessorRequest
)


class PoolingIOProcessor:
    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        *,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        trust_request_chat_template: bool = False,
    ):
        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)

        self.model_config = model_config
        self.renderer = renderer

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template

    async def pre_process(self, *args, **kwargs):
        raise NotImplementedError

    async def post_process(self, *args, **kwargs):
        pass

    def create_pooling_params(self, request):
        return request.to_pooling_params()

    def _validate_input(
        self,
        request: AnyPoolingRequest,
        input_ids: list[int],
        input_text: str,
    ) -> TokensPrompt:
        token_num = len(input_ids)
        max_model_len = self.model_config.max_model_len
        # Note: input length can be up to the entire model context length
        # since these requests don't generate tokens.
        if token_num > max_model_len:
            operations: dict[type[AnyPoolingRequest], str] = {
                ScoreDataRequest: "score",
                ScoreTextRequest: "score",
                ScoreQueriesDocumentsRequest: "score",
                ClassificationCompletionRequest: "classification",
                ClassificationChatRequest: "classification",
            }
            operation = operations.get(type(request), "embedding generation")
            raise VLLMValidationError(
                f"This model's maximum context length is "
                f"{max_model_len} tokens. However, you requested "
                f"{token_num} tokens in the input for {operation}. "
                f"Please reduce the length of the input.",
                parameter="input_tokens",
                value=token_num,
            )
        return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)
