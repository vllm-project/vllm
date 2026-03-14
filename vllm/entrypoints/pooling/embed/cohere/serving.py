# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translates Cohere /v2/embed requests into the existing OpenAI embedding
pipeline and converts the response back into the Cohere format.

Any embedding model works.  The ``input_type`` field is used to select a
prompt prefix for the model:

* For models whose HuggingFace ``config.json`` contains a
  ``task_instructions`` dict, the ``input_type`` value is matched against its
  keys and the corresponding prefix is prepended to each text.
* Otherwise, if the model's ``config_sentence_transformers.json`` contains a
  ``prompts`` dict, those keys are used instead.
* Models without either config reject all ``input_type`` values.

Instead of going through the full HTTP response cycle (which would require
an unnecessary JSON serialization / deserialization roundtrip), this module
calls :meth:`PoolingServing._process` directly to obtain raw
:class:`PoolingRequestOutput` objects from the engine.  Multimodal (chat)
requests are dispatched in parallel via :func:`asyncio.gather`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from fastapi import Request
from fastapi.responses import JSONResponse
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL

from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    CustomChatCompletionMessageParam,
)
from vllm.entrypoints.pooling.embed.cohere.protocol import (
    CohereEmbedInput,
    CohereEmbedRequest,
    CohereEmbedResponse,
    CohereMeta,
    CohereTokens,
    build_typed_embeddings,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
)
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.pooling.utils import encode_pooling_output_float

logger = logging.getLogger(__name__)


def _load_st_prompts(
    model: str | Any,
    revision: str | None,
) -> dict[str, str] | None:
    """Load the ``prompts`` dict from ``config_sentence_transformers.json``.

    Returns ``None`` when the file doesn't exist or has no ``prompts`` key.
    """
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    cfg = get_hf_file_to_dict("config_sentence_transformers.json", str(model), revision)
    if cfg is None:
        return None
    prompts = cfg.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        return None
    return {k: v for k, v in prompts.items() if isinstance(v, str)}


def _mixed_input_to_messages(
    inp: CohereEmbedInput,
    *,
    st_prefix: str | None = None,
) -> list[ChatCompletionMessageParam]:
    """Build chat messages from a mixed text+image input.

    When *st_prefix* is given, it is prepended to each text part.
    """
    parts: list[ChatCompletionContentPartParam] = []
    for item in inp.content:
        if item.type == "text" and item.text is not None:
            text = st_prefix + item.text if st_prefix else item.text
            parts.append(ChatCompletionContentPartTextParam(type="text", text=text))
        elif item.type == "image_url" and item.image_url is not None:
            parts.append(
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(url=item.image_url["url"]),
                )
            )
    return [CustomChatCompletionMessageParam(role="user", content=parts)]


class CohereServingEmbedding:
    """Translates Cohere /v2/embed requests into the existing OpenAI
    embedding pipeline and converts the response back."""

    def __init__(self, openai_handler: ServingEmbedding) -> None:
        self.openai_handler = openai_handler
        model_config = openai_handler.model_config
        hf_config = model_config.hf_config
        self.st_prompts: dict[str, str] | None = self._load_task_instructions(
            hf_config
        ) or _load_st_prompts(model_config.model, model_config.revision)
        if self.st_prompts:
            logger.info(
                "Loaded prompt prefixes for input_type: %s",
                list(self.st_prompts.keys()),
            )

    @staticmethod
    def _load_task_instructions(hf_config: Any) -> dict[str, str] | None:
        """Extract ``task_instructions`` from the HF model config."""
        ti = getattr(hf_config, "task_instructions", None)
        if not isinstance(ti, dict) or not ti:
            return None
        return {k: v for k, v in ti.items() if isinstance(v, str)}

    async def create_embedding(
        self,
        request: CohereEmbedRequest,
        raw_request: Request | None = None,
    ) -> JSONResponse:
        if request.texts is None and request.images is None and request.inputs is None:
            raise ValueError("One of texts, images, or inputs must be provided")

        truncate_prompt_tokens, truncation_side = self._resolve_truncation(request)
        input_type = request.input_type
        self._validate_input_type(input_type)
        texts_echo: list[str] | None = None

        # When truncate="NONE" with an explicit max_tokens, the pipeline
        # should NOT truncate but must reject inputs that exceed the limit.
        max_tokens_check = (
            request.max_tokens
            if request.truncate == "NONE" and request.max_tokens is not None
            else None
        )

        if request.images is not None:
            all_floats, resp_id, total_tokens = await self._process_chat_batch(
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

        elif request.inputs is not None:
            st_prefix = self._get_st_prefix(input_type)
            all_floats, resp_id, total_tokens = await self._process_chat_batch(
                request,
                [
                    _mixed_input_to_messages(inp, st_prefix=st_prefix)
                    for inp in request.inputs
                ],
                raw_request,
                truncate_prompt_tokens,
                truncation_side,
                max_tokens_check=max_tokens_check,
            )

        else:
            texts = request.texts or []
            texts_echo = list(texts) if texts else None
            prefixed = self._apply_st_prompt(texts, input_type)
            all_floats, resp_id, total_tokens = await self._process_completion(
                request,
                prefixed,
                raw_request,
                truncate_prompt_tokens,
                truncation_side,
                max_tokens_check=max_tokens_check,
            )

        return self._build_response(
            request,
            all_floats,
            resp_id=resp_id,
            total_tokens=total_tokens,
            texts_echo=texts_echo,
        )

    async def _process_completion(
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
        ctx = await self.openai_handler._process(comp_req, raw_request)
        self._check_max_tokens(ctx.final_res_batch, max_tokens_check)
        all_floats = [encode_pooling_output_float(out) for out in ctx.final_res_batch]
        total_tokens = sum(len(out.prompt_token_ids) for out in ctx.final_res_batch)
        return all_floats, ctx.request_id, total_tokens

    async def _process_chat_batch(
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
            self.openai_handler._process(chat_reqs[0], raw_request),
            *[self.openai_handler._process(req, None) for req in chat_reqs[1:]],
        )

        all_floats: list[list[float]] = []
        total_tokens = 0
        for ctx in contexts:
            self._check_max_tokens(ctx.final_res_batch, max_tokens_check)
            for output in ctx.final_res_batch:
                all_floats.append(encode_pooling_output_float(output))
                total_tokens += len(output.prompt_token_ids)
        return all_floats, contexts[0].request_id, total_tokens

    @staticmethod
    def _check_max_tokens(
        outputs: list[Any],
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
        if self.st_prompts is None:
            raise ValueError(
                f"Unsupported input_type {input_type!r}. "
                "This model does not define any input_type prompt prefixes."
            )
        if input_type not in self.st_prompts:
            supported = ", ".join(sorted(self.st_prompts))
            raise ValueError(
                f"Unsupported input_type {input_type!r}. Supported values: {supported}"
            )

    def _apply_st_prompt(
        self,
        texts: list[str],
        input_type: str | None,
    ) -> list[str]:
        """Prepend the sentence-transformer prompt prefix for *input_type*.

        Returns *texts* unchanged when no matching prefix is configured.
        """
        if not self.st_prompts or input_type is None:
            return texts
        prefix = self.st_prompts.get(input_type)
        if not prefix:
            return texts
        return [prefix + t for t in texts]

    def _get_st_prefix(self, input_type: str | None) -> str | None:
        """Return the ST prompt prefix for *input_type*, or ``None``."""
        if not self.st_prompts or input_type is None:
            return None
        return self.st_prompts.get(input_type) or None

    @staticmethod
    def _resolve_truncation(
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

    @staticmethod
    def _build_response(
        request: CohereEmbedRequest,
        all_floats: list[list[float]],
        resp_id: str,
        total_tokens: int,
        texts_echo: list[str] | None,
    ) -> JSONResponse:
        embedding_types = request.embedding_types or ["float"]

        embeddings_obj = build_typed_embeddings(
            all_floats,
            list(embedding_types),
        )

        response = CohereEmbedResponse(
            id=resp_id,
            embeddings=embeddings_obj,
            texts=texts_echo,
            meta=CohereMeta(
                tokens=CohereTokens(input_tokens=total_tokens),
            ),
        )
        return JSONResponse(content=response.model_dump(exclude_none=True))
