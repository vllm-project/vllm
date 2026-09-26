# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the text-to-text ``/v1/translations`` endpoint.

These tests are GPU-free: the underlying chat handler and engine are stubbed, so
they exercise the request/response schemas, prompt templating, response mapping,
the SSE stream transform, the encoder-decoder dispatch, and the HTTP route.
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
@pytest.mark.parametrize(
    ("kwargs", "substrings", "exact"),
    [
        # Source given -> translate-from template.
        (dict(text="Hello", source_language="en"), ["from en to de", "Hello"], None),
        # Source omitted -> auto-detect template.
        (dict(text="Bonjour"), ["Detect the source language", "to de"], None),
        # Custom template overrides both defaults.
        (
            dict(text="Ciao", prompt_template="TL {target_language}: {text}"),
            None,
            "TL de: Ciao",
        ),
    ],
)
def test_build_prompt(kwargs, substrings, exact):
    prompt = _make_handler()._build_prompt(
        TranslationRequest(model="m", target_language="de", **kwargs)
    )
    if exact is not None:
        assert prompt == exact
    else:
        assert all(s in prompt for s in substrings)


# --------------------------------------------------------------------------- #
# Response mapping (chat-delegation path)
# --------------------------------------------------------------------------- #
def test_response_mapping_strips_and_carries_usage():
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
    resp = _make_handler()._to_translation_response(req, chat_response)
    assert resp.translated_text == "Hallo Welt"  # stripped
    assert resp.id.startswith("transl-")  # chatcmpl- -> transl-
    assert resp.source_language == "en"
    assert resp.target_language == "de"
    assert resp.usage.total_tokens == 7


# --------------------------------------------------------------------------- #
# Streaming transform (chat SSE -> translation.chunk SSE)
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

    gen = h._translation_stream_generator(req, fake_chat_stream())
    lines = [c.strip() async for c in gen]
    assert lines[-1] == "data: [DONE]"

    parsed = [json.loads(ln[len("data:") :]) for ln in lines if "[DONE]" not in ln]
    assert parsed[0]["object"] == "translation.chunk"
    assert parsed[0]["id"].startswith("transl-")
    assert parsed[0]["choices"][0]["delta"]["translated_text"] == "Hallo"
    assert parsed[1]["choices"][0]["delta"]["translated_text"] == " Welt"
    assert parsed[1]["choices"][0]["finish_reason"] == "stop"


# --------------------------------------------------------------------------- #
# Direct-generate path (encoder-decoder MT models)
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
        self.captured.update(
            prompt=prompt, sampling_params=sampling_params, request_id=request_id
        )
        yield self._request_output


def _make_direct_handler(engine, is_encoder_decoder=True):
    stub_chat = SimpleNamespace(
        models="MODELS",
        model_config=SimpleNamespace(is_encoder_decoder=is_encoder_decoder),
        engine_client=engine,
    )
    h = OpenAIServingTextTranslation(stub_chat)

    async def _ok(_request):  # bypass model-registry validation
        return None

    h._check_model = _ok
    return h


def _default_engine():
    return _FakeEngine(
        _FakeRequestOutput(
            _FakeOutput(text="  Hallo Welt  ", token_ids=[10, 11, 2]),
            prompt_token_ids=[],
            encoder_prompt_token_ids=[7, 8, 9, 2],
        )
    )


@pytest.mark.asyncio
async def test_direct_generate_dispatch_prompt_and_response():
    """An encoder-decoder model dispatches to the direct path (not chat), builds
    the enc-dec prompt, decodes greedily, and reshapes the output."""
    engine = _default_engine()
    h = _make_direct_handler(engine)
    req = TranslationRequest(model="opus-mt", text="Hello world", target_language="de")
    resp = await h.create_translation(req, None)

    assert isinstance(resp, TranslationResponse)
    assert resp.translated_text == "Hallo Welt"  # stripped
    assert resp.id.startswith("transl-")
    assert resp.target_language == "de"
    # Source -> encoder "text" modality; target language -> decoder prompt for
    # the model's own conditioning hook.
    prompt = engine.captured["prompt"]
    assert prompt["encoder_prompt"]["multi_modal_data"]["text"] == "Hello world"
    assert prompt["decoder_prompt"] == "de"
    assert engine.captured["sampling_params"].temperature == 0.0  # greedy
    # Usage counts encoder-prompt + completion tokens.
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (4, 3)
    assert resp.usage.total_tokens == 7


class _StreamingEngine:
    """Stub engine whose ``generate`` yields incremental (DELTA) outputs."""

    def __init__(self, deltas):
        # deltas: list of (text, finish_reason) tuples.
        self._deltas = deltas
        self.captured = {}

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.captured["prompt"] = prompt
        self.captured["sampling_params"] = sampling_params
        self.captured["request_id"] = request_id
        for i, (text, finish) in enumerate(self._deltas):
            yield _FakeRequestOutput(
                _FakeOutput(text=text, token_ids=[i], finish_reason=finish),
                prompt_token_ids=[],
            )


@pytest.mark.asyncio
async def test_direct_generate_streams_sse_chunks():
    engine = _StreamingEngine([("Hallo", None), (" Welt", None), ("", "stop")])
    h = _make_direct_handler(engine)
    req = TranslationRequest(
        model="opus-mt", text="Hello world", target_language="de", stream=True
    )
    gen = await h._create_translation_direct(req, None)

    lines = [c.strip() async for c in gen]
    assert lines[-1] == "data: [DONE]"
    # The engine is asked for incremental deltas, not cumulative text.
    from vllm.sampling_params import RequestOutputKind

    assert engine.captured["sampling_params"].output_kind == RequestOutputKind.DELTA

    parsed = [
        json.loads(line[len("data:") :]) for line in lines if "[DONE]" not in line
    ]
    assert parsed[0]["object"] == "translation.chunk"
    assert parsed[0]["id"].startswith("transl-")
    assert parsed[0]["choices"][0]["delta"]["translated_text"] == "Hallo"
    assert parsed[1]["choices"][0]["delta"]["translated_text"] == " Welt"
    # Final chunk carries the finish reason (empty delta is dropped).
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"
    # Concatenated deltas reconstruct the full translation.
    text = "".join(
        p["choices"][0]["delta"].get("translated_text") or "" for p in parsed
    )
    assert text == "Hallo Welt"


@pytest.mark.asyncio
async def test_direct_generate_streaming_rejects_unknown_language():
    # Pre-validation runs before streaming begins: an unknown target still yields
    # a synchronous 400 (ErrorResponse), never a half-open stream.
    engine = _StreamingEngine([("x", "stop")])
    engine.renderer = _FakeRenderer(_FakeProcessor(known={"deu_Latn"}))
    h = _make_direct_handler(engine)
    req = TranslationRequest(
        model="nllb", text="Hello", target_language="xx_Yyyy", stream=True
    )
    resp = await h._create_translation_direct(req, None)
    assert isinstance(resp, ErrorResponse)
    assert resp.error.code == 400
    assert "prompt" not in engine.captured


@pytest.mark.asyncio
async def test_direct_generate_errors_without_engine():
    h = _make_direct_handler(engine=None)  # no engine_client -> 500
    req = TranslationRequest(model="opus-mt", text="Hello", target_language="de")
    resp = await h._create_translation_direct(req, None)
    assert isinstance(resp, ErrorResponse)
    assert resp.error.code == 500


@pytest.mark.asyncio
async def test_decoder_only_uses_chat_delegation():
    called = {}

    async def fake_create_chat_completion(chat_request, raw_request):
        called["chat"] = chat_request
        return ChatCompletionResponse(
            id="chatcmpl-abc",
            model="instruct",
            choices=[
                ChatCompletionResponseChoice(
                    index=0, message=ChatMessage(role="assistant", content="Hallo Welt")
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
    assert "chat" in called  # went through chat delegation, not direct generate
    assert resp.translated_text == "Hallo Welt"


# --------------------------------------------------------------------------- #
# Target-language conditioning (encoder-decoder direct path)
# --------------------------------------------------------------------------- #
class _FakeProcessor:
    """Stub multimodal processor exposing ``create_decoder_prompt``.

    Mimics NLLB/M2M100: an unknown target code raises ``ValueError``; a known
    code resolves to a single forced-BOS token id.
    """

    def __init__(self, known):
        self._known = set(known)

    def create_decoder_prompt(self, prompt, mm_items):
        if prompt not in self._known:
            raise ValueError(f"Unsupported target language: {prompt!r}")
        return [42]


class _FakeRenderer:
    def __init__(self, processor):
        self._processor = processor

    def get_mm_processor(self):
        return self._processor


def _engine_with_renderer(processor):
    engine = _default_engine()
    engine.renderer = _FakeRenderer(processor)
    return engine


@pytest.mark.asyncio
async def test_direct_generate_forwards_target_language_code():
    engine = _default_engine()
    h = _make_direct_handler(engine)
    req = TranslationRequest(model="nllb", text="Hello", target_language="deu_Latn")
    await h._create_translation_direct(req, None)
    assert engine.captured["prompt"]["decoder_prompt"] == "deu_Latn"


@pytest.mark.asyncio
async def test_pre_validation_rejects_unknown_language():
    engine = _engine_with_renderer(_FakeProcessor(known={"deu_Latn"}))
    h = _make_direct_handler(engine)
    req = TranslationRequest(model="nllb", text="Hello", target_language="xxx_Zzzz")
    resp = await h._create_translation_direct(req, None)
    assert isinstance(resp, ErrorResponse)
    assert resp.error.code == 400
    # Rejected before generation -- the engine was never asked to generate.
    assert "prompt" not in engine.captured


@pytest.mark.asyncio
async def test_pre_validation_allows_known_language():
    engine = _engine_with_renderer(_FakeProcessor(known={"deu_Latn"}))
    h = _make_direct_handler(engine)
    req = TranslationRequest(model="nllb", text="Hello", target_language="deu_Latn")
    resp = await h._create_translation_direct(req, None)
    assert isinstance(resp, TranslationResponse)
    assert engine.captured["prompt"]["decoder_prompt"] == "deu_Latn"


@pytest.mark.asyncio
async def test_pre_validation_skipped_without_renderer():
    # A model without a renderer/processor (e.g. bilingual MarianMT) proceeds
    # to generation; the target code is passed through unvalidated.
    engine = _default_engine()  # no ``renderer`` attribute
    h = _make_direct_handler(engine)
    req = TranslationRequest(model="opus-mt", text="Hello", target_language="anything")
    resp = await h._create_translation_direct(req, None)
    assert isinstance(resp, TranslationResponse)


@pytest.mark.asyncio
async def test_direct_generate_maps_value_error_to_400():
    class _RaisingEngine:
        def __init__(self):
            self.captured = {}

        async def generate(self, prompt, sampling_params, request_id, **kwargs):
            raise ValueError("Unsupported target language: 'xx'")
            yield  # pragma: no cover - makes this an async generator

    h = _make_direct_handler(_RaisingEngine())
    req = TranslationRequest(model="nllb", text="Hello", target_language="xx")
    resp = await h._create_translation_direct(req, None)
    assert isinstance(resp, ErrorResponse)
    assert resp.error.code == 400


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
        return ErrorResponse(
            error=ErrorInfo(message="no model", type="NotFoundError", code=404)
        )


def _client(handler):
    app = FastAPI()
    attach_router(app)
    app.state.openai_serving_text_translation = handler
    app.state.enable_server_load_tracking = False
    return TestClient(app)


@pytest.mark.parametrize(
    ("mode", "payload", "status", "check"),
    [
        # Success -> TranslationResponse body.
        (
            "ok",
            {"model": "m", "text": "Hello world", "target_language": "de"},
            200,
            lambda b: (
                b["object"] == "translation" and b["translated_text"] == "Hallo Welt"
            ),
        ),
        # Handler ErrorResponse is passed through with its status code.
        (
            "err",
            {"model": "m", "text": "Hello", "target_language": "de"},
            404,
            lambda b: b["error"]["type"] == "NotFoundError",
        ),
        # Missing required target_language -> 422 from request validation.
        ("ok", {"model": "m", "text": "Hello"}, 422, lambda b: True),
    ],
)
def test_route(mode, payload, status, check):
    resp = _client(_FakeHandler(mode)).post("/v1/translations", json=payload)
    assert resp.status_code == status
    assert check(resp.json())
