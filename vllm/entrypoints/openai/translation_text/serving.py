# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving handler for the text-to-text ``/v1/translations`` endpoint.

The endpoint picks one of two strategies per model:

* **Encoder-decoder MT models (direct-generate).** MarianMT / NLLB take the
  source text as the encoder's single "text" modality and generate directly
  through the engine. Selected when ``model_config.is_encoder_decoder`` is true.
  The target language is forwarded as the decoder prompt so the model chooses the
  output language: NLLB/M2M100 resolve the code to a ``forced_bos_token_id``,
  while bilingual Marian implies the target and ignores it. An unknown code is
  rejected with a 400 before generation (:meth:`_validate_target_language`).
* **Decoder-only / instruct models (chat-delegation).** The request is wrapped
  in an instruction prompt and handed to :class:`OpenAIServingChat`, reusing the
  full generation pipeline; the chat response is reshaped into the translation
  schema.
"""

import json
import time
from collections.abc import AsyncGenerator
from http import HTTPStatus

from fastapi import Request

from vllm import SamplingParams
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.translation_text.protocol import (
    TranslationDelta,
    TranslationRequest,
    TranslationResponse,
    TranslationStreamResponse,
    TranslationStreamResponseChoice,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.logger import init_logger

logger = init_logger(__name__)

# Default instruction templates. Placeholders: {source_language},
# {target_language}, {text}. Kept configurable so operators can tune them per
# model family (translation quality varies across Granite/Llama/Mistral).
DEFAULT_TRANSLATION_PROMPT_TEMPLATE = (
    "Translate the following text from {source_language} to {target_language}. "
    "Output only the translation, with no additional commentary.\n\n{text}"
)
DEFAULT_TRANSLATION_DETECT_PROMPT_TEMPLATE = (
    "Detect the source language of the following text and translate it to "
    "{target_language}. Output only the translation, with no additional "
    "commentary.\n\n{text}"
)


class OpenAIServingTextTranslation(BaseServing):
    def __init__(
        self,
        chat_serving: OpenAIServingChat,
        *,
        request_logger=None,
        default_prompt_template: str | None = None,
        default_detect_prompt_template: str | None = None,
    ) -> None:
        # Reuse the model/config wiring already resolved by the chat handler.
        super().__init__(
            models=chat_serving.models,
            model_config=chat_serving.model_config,
            request_logger=request_logger,
        )
        self.chat_serving = chat_serving
        # The chat handler is a ``GenerateBaseServing`` and already holds the
        # engine client; the direct-generate (encoder-decoder) path reuses it so
        # no extra wiring is needed in the API-server startup.
        self.engine_client = getattr(chat_serving, "engine_client", None)
        self.default_prompt_template = (
            default_prompt_template or DEFAULT_TRANSLATION_PROMPT_TEMPLATE
        )
        self.default_detect_prompt_template = (
            default_detect_prompt_template or DEFAULT_TRANSLATION_DETECT_PROMPT_TEMPLATE
        )

    def _build_prompt(self, request: TranslationRequest) -> str:
        if request.prompt_template:
            return request.prompt_template.format(
                source_language=request.source_language or "the source language",
                target_language=request.target_language,
                text=request.text,
            )
        if request.source_language:
            return self.default_prompt_template.format(
                source_language=request.source_language,
                target_language=request.target_language,
                text=request.text,
            )
        return self.default_detect_prompt_template.format(
            target_language=request.target_language,
            text=request.text,
        )

    async def create_translation(
        self,
        request: TranslationRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | TranslationResponse | ErrorResponse:
        error = await self._check_model(request)
        if error is not None:
            return error

        # Encoder-decoder MT models generate directly; decoder-only / instruct
        # models fall through to chat delegation.
        if self.model_config.is_encoder_decoder:
            return await self._create_translation_direct(request, raw_request)

        prompt = self._build_prompt(request)
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[{"role": "user", "content": prompt}],
            stream=request.stream,
            temperature=request.temperature,
            top_p=request.top_p,
            max_completion_tokens=request.max_tokens,
            seed=request.seed,
        )

        result = await self.chat_serving.create_chat_completion(
            chat_request, raw_request
        )

        if isinstance(result, ErrorResponse):
            return result
        if isinstance(result, ChatCompletionResponse):
            return self._to_translation_response(request, result)
        # Streaming: ``result`` is an async generator of SSE strings.
        return self._translation_stream_generator(request, result)

    async def _create_translation_direct(
        self,
        request: TranslationRequest,
        raw_request: Request | None = None,
    ) -> TranslationResponse | ErrorResponse:
        """Direct-generate path for encoder-decoder MT models: the source text
        becomes the encoder's single "text" modality and generation runs straight
        through the engine. Streaming is not supported here and is rejected
        cleanly rather than silently returning a non-streamed body.
        """
        if self.engine_client is None:
            return self.create_error_response(
                "Translation is not available: no engine client is configured "
                "for the encoder-decoder generation path.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        if request.stream:
            return self.create_error_response(
                "Streaming is not yet supported for encoder-decoder translation "
                'models; retry with "stream": false.',
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        # Reject an unsupported target language before generation: otherwise the
        # engine's renderer silently tokenizes an invalid code to subwords and
        # produces garbage instead of a 400.
        lang_error = self._validate_target_language(request)
        if lang_error is not None:
            return lang_error

        request_id = self._base_request_id(raw_request) or f"transl-{time.time()}"
        # Greedy by default: best translation quality, and matches the parity tests.
        sampling_params = SamplingParams(
            temperature=request.temperature if request.temperature is not None else 0.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            max_tokens=request.max_tokens,
            seed=request.seed,
        )

        # Source is the encoder's single "text" modality; the target language is
        # the decoder prompt, which the model's create_decoder_prompt hook turns
        # into conditioning (forced_bos for NLLB/M2M100, empty for Marian). The
        # runtime prepends decoder_start_token_id ahead of the hook's output.
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {"text": request.text},
            },
            "decoder_prompt": request.target_language,
        }

        try:
            generator = self.engine_client.generate(prompt, sampling_params, request_id)
            final_res = None
            async for res in generator:
                final_res = res
        except ValueError as exc:
            # Backstop: the model's create_decoder_prompt hook rejected the
            # target language (pre-validation above covers the common case).
            return self.create_error_response(
                str(exc),
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Error during encoder-decoder translation")
            return self.create_error_response(
                f"Translation generation failed: {exc}",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        if final_res is None or not final_res.outputs:
            return self.create_error_response(
                "Translation generation produced no output.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        output = final_res.outputs[0]
        num_prompt_tokens = len(final_res.prompt_token_ids or [])
        encoder_prompt_ids = getattr(final_res, "encoder_prompt_token_ids", None)
        if encoder_prompt_ids:
            num_prompt_tokens += len(encoder_prompt_ids)
        num_completion_tokens = len(output.token_ids)

        response_id = (
            request_id if request_id.startswith("transl-") else f"transl-{request_id}"
        )
        return TranslationResponse(
            id=response_id,
            model=request.model,
            translated_text=output.text.strip(),
            source_language=request.source_language,
            target_language=request.target_language,
            usage=UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            ),
        )

    def _validate_target_language(
        self, request: TranslationRequest
    ) -> ErrorResponse | None:
        """Reject an unsupported target language with a 400 before generation.

        Resolves the code through the model's own ``create_decoder_prompt`` hook
        (the one the engine calls), so an unknown NLLB/M2M100 code becomes a clean
        400 rather than being silently tokenized into subwords. Returns ``None``
        (letting the request proceed) whenever the check cannot run: no renderer,
        no processor, or a bilingual model whose hook accepts anything.
        """
        renderer = getattr(self.engine_client, "renderer", None)
        if renderer is None:
            return None
        try:
            mm_processor = renderer.get_mm_processor()
        except Exception:
            return None
        create_decoder_prompt = getattr(mm_processor, "create_decoder_prompt", None)
        if create_decoder_prompt is None:
            return None
        try:
            create_decoder_prompt(request.target_language, None)
        except ValueError as exc:
            return self.create_error_response(
                str(exc),
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST,
            )
        except Exception:
            logger.debug(
                "Target-language pre-validation skipped (unexpected error).",
                exc_info=True,
            )
        return None

    def _to_translation_response(
        self,
        request: TranslationRequest,
        chat_response: ChatCompletionResponse,
    ) -> TranslationResponse:
        content = ""
        if chat_response.choices:
            content = chat_response.choices[0].message.content or ""
        return TranslationResponse(
            id=chat_response.id.replace("chatcmpl-", "transl-"),
            created=chat_response.created,
            model=chat_response.model,
            translated_text=content.strip(),
            source_language=request.source_language,
            target_language=request.target_language,
            usage=chat_response.usage,
        )

    async def _translation_stream_generator(
        self,
        request: TranslationRequest,
        chat_generator: AsyncGenerator[str, None],
    ) -> AsyncGenerator[str, None]:
        async for chunk in chat_generator:
            # Pass through anything that is not a JSON data line unchanged.
            if not chunk.startswith("data:"):
                yield chunk
                continue
            payload = chunk[len("data:") :].strip()
            if payload == "[DONE]":
                yield "data: [DONE]\n\n"
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                # Keep-alive comments etc. -- forward verbatim.
                yield chunk
                continue

            choices = data.get("choices") or []
            delta_text = None
            finish_reason = None
            if choices:
                delta_text = (choices[0].get("delta") or {}).get("content")
                finish_reason = choices[0].get("finish_reason")

            translation_chunk = TranslationStreamResponse(
                id=str(data.get("id", "")).replace("chatcmpl-", "transl-"),
                created=data.get("created") or int(time.time()),
                model=data.get("model") or request.model,
                choices=[
                    TranslationStreamResponseChoice(
                        index=0,
                        delta=TranslationDelta(translated_text=delta_text),
                        finish_reason=finish_reason,
                    )
                ],
                usage=data.get("usage"),
            )
            yield f"data: {translation_chunk.model_dump_json(exclude_none=True)}\n\n"
