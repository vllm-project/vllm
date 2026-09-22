# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the text-to-text ``/v1/translations`` endpoint.

These tests are GPU-free: the underlying chat handler is stubbed, so they
exercise the translation request/response schemas, prompt templating, response
mapping, the SSE stream transform, and the HTTP route wiring in isolation.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.translation_text.api_router import attach_router
from vllm.entrypoints.openai.translation_text.protocol import (
    TranslationRequest,
    TranslationResponse,
)
from vllm.entrypoints.openai.translation_text.serving import (
    OpenAIServingTextTranslation,
)
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse, UsageInfo


def _make_handler() -> OpenAIServingTextTranslation:
    # BaseServing.__init__ only stores ``models``/``model_config``.
    stub_chat = SimpleNamespace(models="MODELS", model_config="CONFIG")
    return OpenAIServingTextTranslation(stub_chat)


# --------------------------------------------------------------------------- #
# Prompt templating
# --------------------------------------------------------------------------- #
def test_prompt_with_source_language():
    h = _make_handler()
    prompt = h._build_prompt(
        TranslationRequest(
            model="m", text="Hello", source_language="en", target_language="de"
        )
    )
    assert "from en to de" in prompt
    assert prompt.endswith("Hello")


def test_prompt_auto_detect_when_source_omitted():
    h = _make_handler()
    prompt = h._build_prompt(
        TranslationRequest(model="m", text="Bonjour", target_language="en")
    )
    assert "Detect the source language" in prompt
    assert "to en" in prompt


def test_prompt_custom_template_override():
    h = _make_handler()
    prompt = h._build_prompt(
        TranslationRequest(
            model="m",
            text="Ciao",
            target_language="en",
            prompt_template="TL {target_language}: {text}",
        )
    )
    assert prompt == "TL en: Ciao"


# --------------------------------------------------------------------------- #
# Response mapping
# --------------------------------------------------------------------------- #
def test_response_mapping_strips_and_carries_usage():
    h = _make_handler()
    chat_response = ChatCompletionResponse(
        id="chatcmpl-xyz",
        model="m",
        choices=[
            ChatCompletionResponseChoice(
                index=0, message=ChatMessage(role="assistant", content="  Hallo Welt  ")
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )
    req = TranslationRequest(
        model="m", text="Hello world", source_language="en", target_language="de"
    )
    resp = h._to_translation_response(req, chat_response)
    assert isinstance(resp, TranslationResponse)
    assert resp.translated_text == "Hallo Welt"  # stripped
    assert resp.id.startswith("transl-")
    assert resp.source_language == "en"
    assert resp.target_language == "de"
    assert resp.usage.total_tokens == 7


# --------------------------------------------------------------------------- #
# Streaming transform
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stream_transform_remaps_chat_chunks():
    h = _make_handler()
    req = TranslationRequest(model="m", text="Hello", target_language="de", stream=True)

    async def fake_chat_stream():
        yield (
            'data: {"id":"chatcmpl-abc","created":123,"model":"m",'
            '"choices":[{"delta":{"content":"Hallo"},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"id":"chatcmpl-abc","created":123,"model":"m",'
            '"choices":[{"delta":{"content":" Welt"},"finish_reason":"stop"}]}\n\n'
        )
        yield "data: [DONE]\n\n"

    lines = [
        c.strip()
        async for c in h._translation_stream_generator(req, fake_chat_stream())
    ]
    assert lines[-1] == "data: [DONE]"

    parsed = [
        json.loads(line[len("data:") :]) for line in lines if "[DONE]" not in line
    ]
    assert parsed[0]["object"] == "translation.chunk"
    assert parsed[0]["id"].startswith("transl-")
    assert parsed[0]["choices"][0]["delta"]["translated_text"] == "Hallo"
    assert parsed[1]["choices"][0]["delta"]["translated_text"] == " Welt"
    assert parsed[1]["choices"][0]["finish_reason"] == "stop"


# --------------------------------------------------------------------------- #
# Direct-generate path (encoder-decoder MT models) — PR 3.1
# --------------------------------------------------------------------------- #
class _FakeOutput:
    def __init__(self, text, token_ids, finish_reason="stop"):
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason


class _FakeRequestOutput:
    def __init__(self, output, prompt_token_ids, encoder_prompt_token_ids=None):
        self.outputs = [output]
        self.prompt_token_ids = prompt_token_ids
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.finished = True


class _FakeEngine:
    """Stub engine client whose ``generate`` yields one final output."""

    def __init__(self, request_output):
        self._request_output = request_output
        self.captured = {}

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.captured["prompt"] = prompt
        self.captured["sampling_params"] = sampling_params
        self.captured["request_id"] = request_id
        yield self._request_output


def _make_direct_handler(engine, is_encoder_decoder=True):
    stub_chat = SimpleNamespace(
        models="MODELS",
        model_config=SimpleNamespace(is_encoder_decoder=is_encoder_decoder),
        engine_client=engine,
    )
    return OpenAIServingTextTranslation(stub_chat)


def _default_engine():
    return _FakeEngine(
        _FakeRequestOutput(
            _FakeOutput(text="  Hallo Welt  ", token_ids=[10, 11, 2]),
            prompt_token_ids=[],
            encoder_prompt_token_ids=[7, 8, 9, 2],
        )
    )


@pytest.mark.asyncio
async def test_direct_generate_builds_enc_dec_prompt_and_response():
    engine = _default_engine()
    h = _make_direct_handler(engine)
    req = TranslationRequest(model="opus-mt", text="Hello world", target_language="de")
    resp = await h._create_translation_direct(req, None)

    assert isinstance(resp, TranslationResponse)
    # Text stripped; response id carries the translation prefix.
    assert resp.translated_text == "Hallo Welt"
    assert resp.id.startswith("transl-")
    assert resp.target_language == "de"
    # Source text is fed to the encoder's single "text" modality; empty decoder
    # prompt (target-language conditioning is PR 3.2).
    prompt = engine.captured["prompt"]
    assert prompt["encoder_prompt"]["multi_modal_data"]["text"] == "Hello world"
    assert prompt["decoder_prompt"] == ""
    # Deterministic (greedy) by default.
    assert engine.captured["sampling_params"].temperature == 0.0
    # Usage counts encoder prompt + completion tokens.
    assert resp.usage.completion_tokens == 3
    assert resp.usage.prompt_tokens == 4
    assert resp.usage.total_tokens == 7


@pytest.mark.asyncio
async def test_direct_generate_rejects_streaming():
    h = _make_direct_handler(_default_engine())
    req = TranslationRequest(
        model="opus-mt", text="Hello", target_language="de", stream=True
    )
    resp = await h._create_translation_direct(req, None)
    assert isinstance(resp, ErrorResponse)
    assert resp.error.code == 400


@pytest.mark.asyncio
async def test_direct_generate_errors_without_engine():
    stub_chat = SimpleNamespace(
        models="MODELS",
        model_config=SimpleNamespace(is_encoder_decoder=True),
        # no engine_client attribute -> getattr default None
    )
    h = OpenAIServingTextTranslation(stub_chat)
    req = TranslationRequest(model="opus-mt", text="Hello", target_language="de")
    resp = await h._create_translation_direct(req, None)
    assert isinstance(resp, ErrorResponse)
    assert resp.error.code == 500


@pytest.mark.asyncio
async def test_create_translation_dispatches_to_direct_for_enc_dec():
    engine = _default_engine()
    h = _make_direct_handler(engine, is_encoder_decoder=True)

    async def _ok(_request):
        return None

    h._check_model = _ok  # bypass model-registry validation
    req = TranslationRequest(model="opus-mt", text="Hello", target_language="de")
    resp = await h.create_translation(req, None)
    assert isinstance(resp, TranslationResponse)
    assert resp.translated_text == "Hallo Welt"
    # The chat handler was NOT used for an encoder-decoder model.
    assert "prompt" in engine.captured


@pytest.mark.asyncio
async def test_create_translation_uses_chat_for_decoder_only():
    called = {}

    async def fake_create_chat_completion(chat_request, raw_request):
        called["chat"] = chat_request
        return ChatCompletionResponse(
            id="chatcmpl-abc",
            model="instruct",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hallo Welt"),
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    stub_chat = SimpleNamespace(
        models="MODELS",
        model_config=SimpleNamespace(is_encoder_decoder=False),
        engine_client=object(),
        create_chat_completion=fake_create_chat_completion,
    )
    h = OpenAIServingTextTranslation(stub_chat)

    async def _ok(_request):
        return None

    h._check_model = _ok
    req = TranslationRequest(model="instruct", text="Hello", target_language="de")
    resp = await h.create_translation(req, None)
    assert isinstance(resp, TranslationResponse)
    # Decoder-only models go through chat delegation, not direct generate.
    assert "chat" in called
    assert resp.translated_text == "Hallo Welt"


# --------------------------------------------------------------------------- #
# HTTP route wiring
# --------------------------------------------------------------------------- #
class _FakeHandler:
    def __init__(self, mode):
        self.mode = mode

    async def create_translation(self, request, raw_request=None):
        if self.mode == "ok":
            return TranslationResponse(
                model=request.model,
                translated_text="Hallo Welt",
                source_language=request.source_language,
                target_language=request.target_language,
            )
        if self.mode == "err":
            return ErrorResponse(
                error=ErrorInfo(message="no model", type="NotFoundError", code=404)
            )
        raise AssertionError("unexpected mode")


def _client(handler):
    app = FastAPI()
    attach_router(app)
    app.state.openai_serving_text_translation = handler
    app.state.enable_server_load_tracking = False
    return TestClient(app)


def test_route_success():
    client = _client(_FakeHandler("ok"))
    resp = client.post(
        "/v1/translations",
        json={"model": "m", "text": "Hello world", "target_language": "de"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "translation"
    assert body["translated_text"] == "Hallo Welt"


def test_route_error_passthrough():
    client = _client(_FakeHandler("err"))
    resp = client.post(
        "/v1/translations",
        json={"model": "m", "text": "Hello", "target_language": "de"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "NotFoundError"


def test_route_missing_target_language_is_422():
    client = _client(_FakeHandler("ok"))
    resp = client.post("/v1/translations", json={"model": "m", "text": "Hello"})
    assert resp.status_code == 422
