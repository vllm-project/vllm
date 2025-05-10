# SPDX-License-Identifier: Apache-2.0

import base64
from typing import Final, Literal, Optional, Union, cast

import numpy as np
from fastapi import Request
from typing_extensions import assert_never, override

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (EmbeddingChatRequest,
                                              EmbeddingRequest,
                                              EmbeddingResponse,
                                              EmbeddingResponseData,
                                              ErrorResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (EmbeddingServeContext,
                                                    OpenAIServing,
                                                    ServeContext)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.outputs import (EmbeddingOutput, EmbeddingRequestOutput,
                          PoolingRequestOutput)

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
