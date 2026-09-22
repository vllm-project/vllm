# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving handler for the text-to-text ``/v1/translations`` endpoint.

The endpoint picks one of two strategies per model:

* **Encoder-decoder MT models (direct-generate).** MarianMT / NLLB take the
  source text as the encoder's single "text" modality and generate directly
  through the engine. Selected when ``model_config.is_encoder_decoder`` is true.
  The target language is resolved to decoder token ids in the API process via
  :meth:`_resolve_decoder_prompt` and forwarded directly so the engine renderer
  does not re-tokenize the code: NLLB/M2M100 normalize an ISO code / name to the
  checkpoint's tag and resolve it to a ``forced_bos_token_id``, while bilingual
  Marian implies the target and takes an empty decoder prompt. An unknown code is
  rejected with a 400 before generation. Both a non-streaming response and
  incremental SSE streaming are supported.
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
from vllm.sampling_params import RequestOutputKind

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
    ) -> AsyncGenerator[str, None] | TranslationResponse | ErrorResponse:
        """Direct-generate path for encoder-decoder MT models: the source text
        becomes the encoder's single "text" modality and generation runs straight
        through the engine. Non-streaming returns a :class:`TranslationResponse`;
        ``stream=True`` returns an async generator of SSE lines. Setup errors (no
        engine, unsupported target language) are returned synchronously as an
        :class:`ErrorResponse` before any streaming begins.
        """
        if self.engine_client is None:
            return self.create_error_response(
                "Translation is not available: no engine client is configured "
                "for the encoder-decoder generation path.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        # Resolve the target language to decoder token ids in this process. The
        # engine's renderer tokenizes a decoder-prompt *string* before the model's
        # create_decoder_prompt hook runs, so forwarding the raw code would both
        # (a) tokenize an invalid code to subword garbage instead of erroring and
        # (b) leak the tokenized code into a bilingual model's decoder. Resolving
        # to ids here (NLLB -> forced-BOS token; MarianMT -> empty) sidesteps both;
        # an unknown code returns a clean 400.
        decoder_prompt, lang_error = self._resolve_decoder_prompt(request)
        if lang_error is not None:
            return lang_error

        request_id = self._base_request_id(raw_request) or f"transl-{time.time()}"
        # Greedy by default: best translation quality, and matches the parity
        # tests. Streaming asks the engine for incremental deltas (each output
        # carries only the new text); non-streaming keeps cumulative output.
        sampling_params = SamplingParams(
            temperature=request.temperature if request.temperature is not None else 0.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            max_tokens=request.max_tokens,
            seed=request.seed,
            output_kind=(
                RequestOutputKind.DELTA
                if request.stream
                else RequestOutputKind.CUMULATIVE
            ),
        )

        # Source as the encoder's single "text" modality; decoder_prompt is the
        # value resolved above -- pre-computed token ids ({"prompt_token_ids":
        # [...]}) when the processor was available (NLLB -> forced-BOS token;
        # MarianMT -> empty), else the raw target string as a best-effort
        # fallback. The runtime prepends decoder_start_token_id ahead of it.
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {"text": request.text},
            },
            "decoder_prompt": decoder_prompt,
        }

        if request.stream:
            return self._translation_direct_stream_generator(
                request, prompt, sampling_params, request_id
            )

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

    async def _translation_direct_stream_generator(
        self,
        request: TranslationRequest,
        prompt: dict,
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        """Stream the direct-generate path as translation SSE chunks.

        ``sampling_params.output_kind`` is ``DELTA``, so each ``RequestOutput``
        carries only the newly generated text; we forward it verbatim as the
        chunk delta (no per-chunk stripping -- that would corrupt token spacing).
        The stream always terminates with ``data: [DONE]``. A generation error is
        surfaced as a single error data line so the client is not left hanging.
        """
        response_id = (
            request_id if request_id.startswith("transl-") else f"transl-{request_id}"
        )
        created = int(time.time())
        try:
            async for res in self.engine_client.generate(
                prompt, sampling_params, request_id
            ):
                if not res.outputs:
                    continue
                output = res.outputs[0]
                delta_text = output.text
                finish_reason = output.finish_reason
                # Skip empty non-final deltas (keep-alive noise); always emit the
                # final chunk so the client sees ``finish_reason``.
                if not delta_text and finish_reason is None:
                    continue
                chunk = TranslationStreamResponse(
                    id=response_id,
                    created=created,
                    model=request.model,
                    choices=[
                        TranslationStreamResponseChoice(
                            index=0,
                            delta=TranslationDelta(translated_text=delta_text or None),
                            finish_reason=finish_reason,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Error during streaming encoder-decoder translation")
            err = self.create_error_response(
                f"Translation generation failed: {exc}",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            yield f"data: {err.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    def _resolve_decoder_prompt(
        self, request: TranslationRequest
    ) -> tuple[object, ErrorResponse | None]:
        """Resolve the target language into the decoder prompt.

        Returns ``(decoder_prompt, None)`` on success, or ``(None, error)`` when
        the target language is unrecognized (a clean 400).

        Calls the model's ``create_decoder_prompt`` hook here, in the API process,
        to turn the code into decoder token ids (NLLB/M2M100 -> forced-BOS token,
        after normalizing an ISO code / name to the checkpoint's tag; MarianMT ->
        ``[]``, target implied) and forwards them as ``{"prompt_token_ids": [...]}``
        so the engine renderer does not re-tokenize the code. Best-effort: if the
        renderer / processor / hook is unavailable, falls back to the raw target
        string and lets the engine be the backstop.
        """
        fallback = request.target_language
        renderer = getattr(self.engine_client, "renderer", None)
        if renderer is None:
            return fallback, None
        try:
            mm_processor = renderer.get_mm_processor()
        except Exception:
            return fallback, None
        create_decoder_prompt = getattr(mm_processor, "create_decoder_prompt", None)
        if create_decoder_prompt is None:
            return fallback, None
        try:
            decoder_ids = create_decoder_prompt(request.target_language, None)
        except ValueError as exc:
            return None, self.create_error_response(
                str(exc),
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST,
            )
        except Exception:
            # Resolution is best-effort; never fail a request because the
            # pre-computation itself hit an unexpected error -- let the engine try
            # with the raw code.
            logger.debug(
                "Decoder-prompt pre-resolution skipped (unexpected error).",
                exc_info=True,
            )
            return fallback, None
        # ``create_decoder_prompt`` returns decoder token ids (possibly empty for
        # a bilingual model). Forward them as an explicit token prompt so the
        # renderer passes them through without re-tokenizing. A non-list return
        # (defensive) is forwarded as-is.
        if isinstance(decoder_ids, (list, tuple)):
            return {"prompt_token_ids": list(decoder_ids)}, None
        return decoder_ids, None

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
