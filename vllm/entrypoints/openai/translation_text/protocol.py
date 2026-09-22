# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request/response schemas for the text-to-text ``/v1/translations`` endpoint.

This endpoint is distinct from the audio ``/v1/audio/translations`` (Whisper)
endpoint: it takes source *text* plus structured language parameters and returns
translated text, following OpenAI-style response conventions (``id``/``object``/
``created``/``model``) even though OpenAI does not define this endpoint.
"""

import time

from pydantic import Field

from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.utils import random_uuid


class TranslationRequest(OpenAIBaseModel):
    # The model to use for translation (must match the served model or a LoRA).
    model: str
    # Source content to translate.
    text: str
    # Target language, e.g. an ISO 639-1 ("de") or 639-3 ("deu") code, or a
    # natural-language name ("German"). Passed through to the prompt template.
    target_language: str
    # Source language. If omitted, the model is asked to auto-detect it.
    source_language: str | None = None
    # Stream the translation back as SSE chunks.
    stream: bool | None = False

    # --- sampling passthrough (forwarded to the underlying generation) ---
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None

    # Optional per-request override of the instruction template. Supports the
    # ``{source_language}``, ``{target_language}`` and ``{text}`` placeholders.
    prompt_template: str | None = None


class TranslationResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"transl-{random_uuid()}")
    object: str = "translation"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    translated_text: str
    # Echoed when provided by the client; ``None`` when auto-detect was used.
    source_language: str | None = None
    target_language: str
    usage: UsageInfo | None = None


class TranslationDelta(OpenAIBaseModel):
    translated_text: str | None = None


class TranslationStreamResponseChoice(OpenAIBaseModel):
    index: int = 0
    delta: TranslationDelta
    finish_reason: str | None = None


class TranslationStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"transl-{random_uuid()}")
    object: str = "translation.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[TranslationStreamResponseChoice]
    usage: UsageInfo | None = None
