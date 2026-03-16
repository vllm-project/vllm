# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import logging
from collections.abc import Callable
from functools import partial
from typing import Literal, TypeAlias, cast

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from typing_extensions import assert_never

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatTemplateConfig,
    CustomChatCompletionMessageParam,
)
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.serving import PoolingServing
from vllm.entrypoints.pooling.embed.io_processor import EmbedIOProcessor
from vllm.entrypoints.pooling.embed.protocol import (
    CohereBilledUnits,
    CohereEmbedInput,
    CohereEmbedRequest,
    CohereEmbedResponse,
    CohereMeta,
    EmbeddingBytesResponse,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    build_typed_embeddings,
)
from vllm.entrypoints.pooling.typing import AnyPoolingRequest, PoolingServeContext
from vllm.entrypoints.pooling.utils import (
    encode_pooling_bytes,
    encode_pooling_output_base64,
    encode_pooling_output_float,
    get_json_response_cls,
)
from vllm.outputs import PoolingRequestOutput
from vllm.renderers import BaseRenderer
from vllm.utils.serial_utils import EmbedDType, Endianness

logger = logging.getLogger(__name__)

JSONResponseCLS = get_json_response_cls()

EmbeddingServeContext: TypeAlias = PoolingServeContext[EmbeddingRequest]


class ServingEmbedding(PoolingServing):
    """Embedding API supporting both OpenAI and Cohere formats."""

    request_id_prefix = "embd"
    io_processor: EmbedIOProcessor

    def init_io_processor(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> EmbedIOProcessor:
        return EmbedIOProcessor(
            model_config=model_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )

    @property
    def task_instructions(self) -> dict[str, str] | None:
        return self.io_processor.task_instructions

    async def _process(
        self,
        request: AnyPoolingRequest,
        raw_request: Request | None = None,
    ) -> PoolingServeContext:
        if isinstance(request, CohereEmbedRequest):
            return await self._process_cohere(request, raw_request)
        return await super()._process(request, raw_request)

    async def _build_response(
        self,
        ctx: PoolingServeContext,
    ) -> Response:
        if isinstance(ctx.request, CohereEmbedRequest):
            return self._build_cohere_response_from_ctx(ctx)
        return await self._build_openai_response(ctx)

    async def _process_cohere(
        self,
        request: CohereEmbedRequest,
        raw_request: Request | None = None,
    ) -> PoolingServeContext:
        if request.texts is None and request.images is None and request.inputs is None:
            raise ValueError("One of texts, images, or inputs must be provided")

        truncate_prompt_tokens, truncation_side = self._resolve_cohere_truncation(
            request
        )
        input_type = request.input_type
        self._validate_input_type(input_type)

        max_tokens_check = (
            request.max_tokens
            if request.truncate == "NONE" and request.max_tokens is not None
            else None
        )

        image_tokens = 0
        texts_echo: list[str] | None = None

        if request.images is not None:
            all_floats, resp_id, total_tokens = await self._cohere_process_chat_batch(
                request,
                [
                    [
                        CustomChatCompletionMessageParam(
                            role="user",
                            content=[{"type": "image_url", "image_url": {"url": uri}}],
                        )
                    ]
                    for uri in request.images
                ],
                raw_request,
                truncate_prompt_tokens,
                truncation_side,
                max_tokens_check=max_tokens_check,
            )
            image_tokens = total_tokens

        elif request.inputs is not None:
            task_prefix = self._get_task_instruction_prefix(input_type)
            all_floats, resp_id, total_tokens = await self._cohere_process_chat_batch(
                request,
                [
                    self._mixed_input_to_messages(inp, task_prefix=task_prefix)
                    for inp in request.inputs
                ],
                raw_request,
                truncate_prompt_tokens,
                truncation_side,
                max_tokens_check=max_tokens_check,
            )

        else:
            texts_echo = request.texts
            prefixed = self._apply_task_instruction(request.texts or [], input_type)
            all_floats, resp_id, total_tokens = await self._cohere_process_completion(
                request,
                prefixed,
                raw_request,
                truncate_prompt_tokens,
                truncation_side,
                max_tokens_check=max_tokens_check,
            )

        model_name = self.models.model_name()
        ctx = PoolingServeContext(
            request=request,
            raw_request=raw_request,
            model_name=model_name,
            request_id=resp_id,
        )
        ctx.intermediates = {
            "all_floats": all_floats,
            "total_tokens": total_tokens,
            "image_tokens": image_tokens,
            "texts_echo": texts_echo,
        }
        return ctx

    async def _cohere_process_completion(
        self,
        request: CohereEmbedRequest,
        texts: list[str],
        raw_request: Request | None,
        truncate_prompt_tokens: int | None,
        truncation_side: Literal["left", "right"] | None,
        max_tokens_check: int | None = None,
    ) -> tuple[list[list[float]], str, int]:
        """Process raw texts via a single batched completion request."""
        comp_req = EmbeddingCompletionRequest(
            model=request.model,
            input=texts,
            dimensions=request.output_dimension,
            encoding_format="float",
            truncate_prompt_tokens=truncate_prompt_tokens,
            truncation_side=truncation_side,
        )
        ctx = await super()._process(comp_req, raw_request)
        self._check_cohere_max_tokens(ctx.final_res_batch, max_tokens_check)
        all_floats = [encode_pooling_output_float(out) for out in ctx.final_res_batch]
        total_tokens = sum(len(out.prompt_token_ids) for out in ctx.final_res_batch)
        return all_floats, ctx.request_id, total_tokens

    async def _cohere_process_chat_batch(
        self,
        request: CohereEmbedRequest,
        all_messages: list[list[ChatCompletionMessageParam]],
        raw_request: Request | None,
        truncate_prompt_tokens: int | None,
        truncation_side: Literal["left", "right"] | None,
        max_tokens_check: int | None = None,
    ) -> tuple[list[list[float]], str, int]:
        """Process a batch of chat requests in parallel."""
        if not all_messages:
            return [], "", 0

        chat_reqs = [
            EmbeddingChatRequest(
                model=request.model,
                messages=msgs,
                dimensions=request.output_dimension,
                encoding_format="float",
                truncate_prompt_tokens=truncate_prompt_tokens,
                truncation_side=truncation_side,
            )
            for msgs in all_messages
        ]
        # Each _process call derives a request ID from raw_request. When
        # X-Request-Id is set, passing the same raw_request to every call
        # would produce duplicate IDs and crash the engine scheduler.
        # Pass raw_request only to the first call; the rest get None so
        # they each generate a unique ID.
        contexts = await asyncio.gather(
            super()._process(chat_reqs[0], raw_request),
            *[super()._process(req, None) for req in chat_reqs[1:]],
        )

        all_floats: list[list[float]] = []
        total_tokens = 0
        for ctx in contexts:
            self._check_cohere_max_tokens(ctx.final_res_batch, max_tokens_check)
            for output in ctx.final_res_batch:
                all_floats.append(encode_pooling_output_float(output))
                total_tokens += len(output.prompt_token_ids)
        return all_floats, contexts[0].request_id, total_tokens

    async def _build_openai_response(
        self,
        ctx: EmbeddingServeContext,
    ) -> JSONResponse | StreamingResponse:
        encoding_format = ctx.request.encoding_format
        embed_dtype = ctx.request.embed_dtype
        endianness = ctx.request.endianness

        if encoding_format == "float" or encoding_format == "base64":
            return self._openai_json_response(
                ctx.final_res_batch,
                ctx.request_id,
                ctx.created_time,
                ctx.model_name,
                encoding_format,
                embed_dtype,
                endianness,
            )

        if encoding_format == "bytes" or encoding_format == "bytes_only":
            return self._openai_bytes_response(
                ctx.final_res_batch,
                ctx.request_id,
                ctx.created_time,
                ctx.model_name,
                encoding_format,
                embed_dtype,
                endianness,
            )

        assert_never(encoding_format)

    def _openai_json_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["float", "base64"],
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> JSONResponse:
        encode_fn = cast(
            Callable[[PoolingRequestOutput], list[float] | str],
            (
                encode_pooling_output_float
                if encoding_format == "float"
                else partial(
                    encode_pooling_output_base64,
                    embed_dtype=embed_dtype,
                    endianness=endianness,
                )
            ),
        )

        items: list[EmbeddingResponseData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
            item = EmbeddingResponseData(
                index=idx,
                embedding=encode_fn(final_res),
            )
            prompt_token_ids = final_res.prompt_token_ids

            items.append(item)
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        response = EmbeddingResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=usage,
        )
        return JSONResponseCLS(content=response.model_dump())

    def _openai_bytes_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["bytes", "bytes_only"],
        embed_dtype: EmbedDType,
        endianness: Endianness,
    ) -> StreamingResponse:
        content, items, usage = encode_pooling_bytes(
            pooling_outputs=final_res_batch,
            embed_dtype=embed_dtype,
            endianness=endianness,
        )

        headers = (
            None
            if encoding_format == "bytes_only"
            else {
                "metadata": json.dumps(
                    {
                        "id": request_id,
                        "created": created_time,
                        "model": model_name,
                        "data": items,
                        "usage": usage,
                    }
                )
            }
        )

        response = EmbeddingBytesResponse(content=content, headers=headers)
        return StreamingResponse(
            content=response.content,
            headers=response.headers,
            media_type=response.media_type,
        )

    @staticmethod
    def _build_cohere_response_from_ctx(
        ctx: PoolingServeContext,
    ) -> JSONResponse:
        request = ctx.request
        assert isinstance(request, CohereEmbedRequest)
        intermediates = ctx.intermediates
        assert isinstance(intermediates, dict)

        all_floats: list[list[float]] = intermediates["all_floats"]
        total_tokens: int = intermediates["total_tokens"]
        image_tokens: int = intermediates["image_tokens"]
        texts_echo: list[str] | None = intermediates["texts_echo"]

        embedding_types = request.embedding_types or ["float"]
        embeddings_obj = build_typed_embeddings(all_floats, embedding_types)

        input_tokens = total_tokens - image_tokens
        response = CohereEmbedResponse(
            id=ctx.request_id,
            embeddings=embeddings_obj,
            texts=texts_echo,
            meta=CohereMeta(
                billed_units=CohereBilledUnits(
                    input_tokens=input_tokens,
                    image_tokens=image_tokens,
                ),
            ),
        )
        return JSONResponse(content=response.model_dump(exclude_none=True))

    @staticmethod
    def _mixed_input_to_messages(
        inp: CohereEmbedInput,
        *,
        task_prefix: str | None = None,
    ) -> list[ChatCompletionMessageParam]:
        """Build chat messages from a mixed text+image input.

        When *task_prefix* is given, it is prepended to each text part.
        """
        parts: list[ChatCompletionContentPartParam] = []
        for item in inp.content:
            if item.type == "text" and item.text is not None:
                text = task_prefix + item.text if task_prefix else item.text
                parts.append(ChatCompletionContentPartTextParam(type="text", text=text))
            elif item.type == "image_url" and item.image_url is not None:
                parts.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(url=item.image_url["url"]),
                    )
                )
        return [CustomChatCompletionMessageParam(role="user", content=parts)]

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
        """Prepend the task instruction prompt prefix for *input_type*.

        Returns *texts* unchanged when no matching prefix is configured.
        """
        prefix = self._get_task_instruction_prefix(input_type)
        if not prefix:
            return texts
        return [prefix + t for t in texts]

    def _get_task_instruction_prefix(self, input_type: str | None) -> str | None:
        """Return the task instruction prompt prefix for *input_type*."""
        if not self.task_instructions or input_type is None:
            return None
        return self.task_instructions.get(input_type) or None

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
