# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise logprob representation over HTTP with a scripted engine."""

import httpx
import pytest
from fastapi import FastAPI
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2Config, TokenizersBackend

from tests.entrypoints.scale_out.token_in_token_out.test_generate_stream import (
    _make_request_output,
)
from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.scale_out.derender.api_router import router as derender_router
from vllm.entrypoints.scale_out.derender.serving import ServingDerender
from vllm.entrypoints.scale_out.render.api_router import router as render_router
from vllm.entrypoints.scale_out.render.serving import ServingRender
from vllm.entrypoints.scale_out.token_in_token_out.api_router import (
    router as generate_router,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.logprobs import Logprob
from vllm.parser import Parser
from vllm.renderers import renderer_from_config
from vllm.renderers.online_derenderer import OnlineDerenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.v1.engine.input_processor import InputProcessor


class _PassThroughParser(Parser):
    def parse(self, model_output, request, **kwargs):
        return None, model_output, None

    def parse_delta(
        self, delta_text, delta_token_ids, request, prompt_token_ids=None, *, finished
    ):
        return DeltaMessage(content=delta_text)


class _ScriptedEngine:
    """External inference service with deterministic token/logprob output."""

    errored = False

    def __init__(self, config):
        self.vllm_config = config
        self.model_config = config.model_config
        self.renderer = renderer_from_config(config)
        self.input_processor = InputProcessor(config, self.renderer)

    def check_admission(self, n):
        pass

    async def generate(self, prompt, params, request_id, **kwargs):
        yield _make_request_output(
            request_id,
            [2],
            text="é",
            finished=True,
            finish_reason="length",
            logprobs=[
                {
                    2: Logprob(-0.5, rank=1, decoded_token="é"),
                    3: Logprob(-1.5, rank=2, decoded_token="e"),
                }
            ],
        )


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("token-id-routes")
    backend = Tokenizer(
        models.WordLevel({"<unk>": 0, "Hi": 1, "é": 2, "e": 3}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.decoder = decoders.WordPiece()
    tokenizer = TokenizersBackend(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    tokenizer.save_pretrained(tmp_path)
    GPT2Config(
        vocab_size=4, n_positions=100, eos_token_id=None, bos_token_id=None
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path), max_model_len=100, max_logprobs=20, dtype="float32"
    )
    config = VllmConfig(model_config=model_config)
    engine = _ScriptedEngine(config)
    registry = OpenAIServingModels(
        engine, [BaseModelPath(name="tiny", model_path=str(tmp_path))]
    )
    online_renderer = OnlineRenderer(
        model_config=model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    derenderer = OnlineDerenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    serving = ServingTokens(
        engine,
        registry,
        online_renderer,
        online_derenderer=derenderer,
        request_logger=None,
    )
    app = FastAPI()
    app.include_router(render_router)
    app.include_router(generate_router)
    app.include_router(derender_router)
    app.state.serving_tokens = serving
    app.state.serving_render = ServingRender(serving.models, serving.online_renderer)
    app.state.serving_derender = ServingDerender(serving.models, derenderer)
    yield app
    engine.renderer._executor.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("server_flag", [False, True])
@pytest.mark.parametrize("request_flag", [None, False, True])
@pytest.mark.parametrize(
    ("mode", "parsed"),
    [
        ("derender", "plain"),
        ("derender", "parsed"),
        ("derender", "hidden"),
        ("text", "plain"),
        ("tokens", "plain"),
    ],
)
async def test_render_generate_logprob_override(
    app, server_flag, request_flag, mode, parsed
):
    app.state.serving_tokens.online_derenderer.parser = (
        _PassThroughParser if parsed != "plain" else None
    )
    app.state.serving_tokens.return_tokens_as_token_ids = server_flag
    chat = {
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": True,
        "top_logprobs": 2,
        "max_completion_tokens": 1,
    }
    if parsed == "hidden":
        chat["include_reasoning"] = False
    if request_flag is not None:
        chat["return_tokens_as_token_ids"] = request_flag
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        rendered = await client.post("/v1/chat/completions/render", json=chat)
        assert rendered.status_code == 200, rendered.text
        payload = rendered.json()
        payload["output_mode"] = mode
        response = await client.post("/inference/v1/generate", json=payload)
        assert response.status_code == 200, response.text
        if mode == "derender" and parsed == "hidden":
            choice = response.json()["choices"][0]
            assert choice["logprobs"] is None
            assert choice["token_ids"] is None
            assert choice["message"]["content"] == "é"
            return
        entry = response.json()["choices"][0]["logprobs"]["content"][0]
        if mode == "tokens":
            assert entry["token_id"] == 2
            assert "token" not in entry and "bytes" not in entry
            return
        placeholders = (
            server_flag if mode == "text" or request_flag is None else request_flag
        )
        assert (entry["token"], entry["bytes"]) == (
            ("token_id:2", None) if placeholders else ("é", [195, 169])
        )
        assert [t["token"] for t in entry["top_logprobs"]] == (
            ["token_id:2", "token_id:3"] if placeholders else ["é", "e"]
        )
        choice = response.json()["choices"][0]
        assert (
            choice["message"]["content"] if mode == "derender" else choice["text"]
        ) == "é"
        if mode == "derender" and request_flag is None and parsed == "plain":
            payload.pop("parse_context")
            legacy = await client.post("/inference/v1/generate", json=payload)
            assert legacy.status_code == 200, legacy.text
            assert legacy.json()["choices"] == response.json()["choices"]
        if mode == "derender":
            payload["stream"] = True
            rejected = await client.post("/inference/v1/generate", json=payload)
            assert rejected.status_code == 400
            assert "does not support stream=true" in rejected.text


@pytest.mark.asyncio
@pytest.mark.parametrize("server_flag", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("route", "request_flag"),
    [
        (route, flag)
        for route in ("completion", "chat", "parsed", "hidden")
        for flag in ("no_request", None, False, True)
        if flag != "no_request" or route in ("completion", "chat")
    ],
)
async def test_derender_logprob_override(app, server_flag, request_flag, stream, route):
    chat = route != "completion"
    parsed = route in ("parsed", "hidden")
    app.state.serving_derender.online_derenderer.parser = (
        _PassThroughParser if parsed else None
    )
    app.state.serving_derender.online_derenderer.return_tokens_as_token_ids = (
        server_flag
    )
    original = (
        {"messages": [], "logprobs": True, "top_logprobs": 1}
        if chat
        else {"prompt": "Hi", "logprobs": 1}
    )
    if request_flag not in (None, "no_request"):
        original["return_tokens_as_token_ids"] = request_flag
    if route == "hidden":
        original["include_reasoning"] = False
    generated = {
        "request_id": "scripted",
        "choices": [
            {
                "index": 0,
                "token_ids": [2],
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {
                            "token_id": 2,
                            "logprob": -0.5,
                            "top_logprobs": [
                                {"token_id": 2, "logprob": -0.5},
                                {"token_id": 3, "logprob": -1.5},
                            ],
                        }
                    ]
                },
            }
        ],
    }
    payload = {
        "stream": stream,
        "prompt_token_ids": [1] if stream or chat else [[1]],
        "chat_request" if chat else "completion_request": original,
    }
    payload[
        "generate_chunk"
        if stream
        else "generate_response"
        if chat
        else "generate_responses"
    ] = generated if stream or chat else [generated]
    if request_flag == "no_request":
        payload.pop("chat_request" if chat else "completion_request")
    path = "/v1/chat/completions/derender" if chat else "/v1/completions/derender"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(path, json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    choice = (data["chunk"] if stream else data)["choices"][0]
    placeholders = server_flag if request_flag in (None, "no_request") else request_flag
    token = "token_id:2" if placeholders else "é"
    if route == "hidden":
        assert choice["logprobs"] is None
        assert choice["delta" if stream else "message"]["content"] == " é"
    elif chat:
        entry = choice["logprobs"]["content"][0]
        assert (entry["token"], entry["bytes"]) == (
            token,
            None if placeholders else [195, 169],
        )
        assert len(entry["top_logprobs"]) == (2 if request_flag == "no_request" else 1)
        assert entry["top_logprobs"][0]["token"] == token
        assert choice["delta" if stream else "message"]["content"] == " é"
    else:
        assert choice["text"] == " é"
        assert choice["logprobs"]["tokens"] == [token]
        assert choice["logprobs"]["top_logprobs"] == [
            {token: -0.5, "token_id:3" if placeholders else "e": -1.5}
        ]
