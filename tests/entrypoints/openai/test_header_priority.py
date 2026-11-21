# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority ingress through real HTTP routes and serving, with an external engine."""

from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.entrypoints.common.cli_args import FrontendArgs
from vllm.entrypoints.openai.chat_completion.api_router import (
    router as chat_router,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.api_router import router as completion_router
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.scale_out.token_in_token_out.api_router import (
    router as token_router,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_derenderer import OnlineDerenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.input_processor import InputProcessor

MODEL = "openai-community/gpt2"
CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


@pytest.fixture(scope="module")
def vllm_config():
    model = ModelConfig(model=MODEL, max_model_len=100)
    return VllmConfig(
        model_config=model,
        scheduler_config=SchedulerConfig(max_model_len=100, is_encoder_decoder=False),
    )


@pytest.fixture
def priority_app(vllm_config):
    engine = MagicMock(spec=AsyncLLM)
    engine.errored = False
    engine.model_config = vllm_config.model_config
    engine.vllm_config = vllm_config
    engine.renderer = HfRenderer(
        vllm_config, cached_tokenizer_from_config(vllm_config.model_config)
    )
    engine.input_processor = InputProcessor(vllm_config, engine.renderer)
    priorities = []

    async def generate(prompt, sampling_params, request_id, *, priority, **kwargs):
        priorities.append(priority)
        yield RequestOutput(
            request_id=request_id,
            prompt="Hello",
            prompt_token_ids=[15496],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=" world",
                    token_ids=[995],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="length",
                )
            ],
            finished=True,
        )

    engine.generate = generate
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name=MODEL, model_path=MODEL)],
    )
    renderer = OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="string",
    )
    app = FastAPI()
    app.state.args = FrontendArgs()
    app.state.vllm_config = vllm_config
    app.state.openai_serving_chat = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        online_renderer=renderer,
        request_logger=None,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="string",
    )
    app.state.openai_serving_completion = OpenAIServingCompletion(
        engine, models, online_renderer=renderer, request_logger=None
    )
    app.include_router(chat_router)
    app.include_router(completion_router)
    init_exception_handler(app)
    return app, priorities


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
@pytest.mark.parametrize(
    "policy,enabled,body,headers,expected",
    [
        ("priority", None, {}, [("X-Priority", "0")], 0),
        ("priority", None, {}, [("X-Priority", "-9")], -9),
        ("priority", True, {}, [("Priority", "u=7, i")], 70),
        ("priority", None, {}, [("Priority", "u=0")], 0),
        ("priority", None, {}, [("Priority", "u=3;x=token, ext=(1 2)")], 30),
        ("priority", None, {"priority": 0}, [("X-Priority", "-9")], 0),
        ("priority", None, {"priority": 0}, [("Priority", "u=7")], 0),
        ("priority", False, {"priority": -5}, [("X-Priority", "0")], -5),
        ("priority", None, {"priority": -5}, [("Priority", "u=0")], -5),
        ("priority", None, {}, [("X-Priority", "12"), ("Priority", "u=0")], 12),
        ("priority", None, {}, [("X-Priority", "bad"), ("Priority", "u=2")], 20),
        ("priority", None, {}, [("X-Priority", "1.0")], 0),
        ("priority", None, {}, [("X-Priority", "1_0")], 0),
        ("priority", None, {}, [("X-Priority", " \t+12 \t")], 12),
        ("priority", None, {}, [("Priority", 'u="1"')], 0),
        ("priority", None, {}, [("Priority", "u=?1")], 0),
        ("priority", None, {}, [("Priority", "u=1.0")], 0),
        ("priority", None, {}, [("Priority", "u=8")], 0),
        ("priority", None, {}, [("Priority", "u=-1")], 0),
        ("priority", None, {}, [("Priority", 'u=2, ext="bad\\q"')], 0),
        ("priority", False, {}, [("X-Priority", "-9")], 0),
        ("priority", False, {}, [("Priority", "u=7")], 0),
        ("fcfs", None, {}, [("X-Priority", "-9")], 0),
        ("fcfs", None, {}, [("Priority", "u=7")], 0),
        ("fcfs", True, {}, [("X-Priority", "-9")], -9),
        ("priority", None, {}, [("X-Priority", "1"), ("X-Priority", "2")], 0),
        ("priority", None, {}, [("Priority", "u=1"), ("Priority", "u=6")], 60),
        ("priority", None, {}, [("Priority", "u=1, u=?0")], 0),
        ("priority", False, {"priority": 0}, [("X-Vllm-Priority", "-4")], -4),
        ("fcfs", None, {"priority": 3}, [("X-Vllm-Priority", "0")], 0),
        ("priority", None, {"priority": 5}, [("X-Vllm-Priority", "bad")], 5),
        (
            "priority",
            None,
            {},
            [("X-Vllm-Priority", "bad"), ("X-Priority", "10")],
            0,
        ),
        (
            "priority",
            None,
            {},
            [("X-Vllm-Priority", "-4"), ("X-Vllm-Priority", "0")],
            -4,
        ),
        (
            "priority",
            None,
            {"priority": 0},
            [("X-Vllm-Priority", "-4"), ("X-Priority", "10")],
            -4,
        ),
        ("priority", None, {}, [], 0),
    ],
)
async def test_header_priority_engine_handoff(
    priority_app, endpoint, policy, enabled, body, headers, expected
):
    app, priorities = priority_app
    app.state.vllm_config.scheduler_config.policy = policy
    app.state.args.allow_header_priority = enabled
    payload = {"model": MODEL, "max_tokens": 1, **body}
    if endpoint == "completions":
        payload["prompt"] = "Hello"
    else:
        payload["messages"] = [{"role": "user", "content": "Hello"}]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(f"/v1/{endpoint}", json=payload, headers=headers)
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    text = choice["text"] if endpoint == "completions" else choice["message"]["content"]
    assert text == " world"
    assert priorities == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
@pytest.mark.parametrize(
    "value,expected",
    [
        (-(1 << 31), -(1 << 31)),
        ((1 << 31) - 1, (1 << 31) - 1),
        (-(1 << 31) - 1, 20),
        (1 << 31, 20),
        (1 << 64, 20),
    ],
)
async def test_x_priority_wire_range(priority_app, endpoint, value, expected):
    await test_header_priority_engine_handoff(
        priority_app,
        endpoint,
        "priority",
        True,
        {},
        [("X-Priority", str(value)), ("Priority", "u=2")],
        expected,
    )


@pytest.mark.asyncio
async def test_token_generate_uses_only_legacy_priority_header(priority_app):
    app, priorities = priority_app
    app.state.args.allow_header_priority = True
    completion = app.state.openai_serving_completion
    app.state.serving_tokens = ServingTokens(
        completion.engine_client,
        completion.models,
        completion.online_renderer,
        online_derenderer=OnlineDerenderer(
            model_config=completion.model_config,
            renderer=completion.renderer,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="string",
        ),
        request_logger=None,
    )
    app.include_router(token_router)
    payload = {
        "model": MODEL,
        "token_ids": [15496],
        "sampling_params": {"max_tokens": 1},
    }
    headers = {"X-Priority": "-9", "Priority": "u=7"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for extra in ({}, {"X-Vllm-Priority": "-4"}):
            response = await client.post(
                "/inference/v1/generate", json=payload, headers=headers | extra
            )
            assert response.status_code == 200, response.text
        response = await client.post(
            "/inference/v1/generate",
            json=payload,
            headers=headers | {"X-Vllm-Priority": str(1 << 31)},
        )
    assert response.status_code == 400, response.text
    assert "X-Vllm-Priority" in response.json()["error"]["message"]
    assert priorities == [0, -4]


@pytest.mark.parametrize(
    "field,expected",
    [
        ("u=0", 0),
        ("u=7", 7),
        ("u=3, i", 3),
        ("u=2 ,  i;  ext=token , other=1", 2),
        (" u=2\t,\ti", 2),
        ('u=2;foo="x", ext=(1 2);a=?1, other=:YWJj:, token=a/b', 2),
        ('u=1, ext="a,b;u=7"', 1),
        ('u=1, ext="a\\"b\\\\c"', 1),
        ("u=1, ext=:YQ:", 1),
        ("u=1, ext=()", 1),
        ("u=1, ext=( 1 2  )", 1),
        ("u=2, ext=(foo; p=token bar; q=other);a=baz", 2),
        ('u=2, ext=("a" :YQ: ?1 1.2)', 2),
        ("u=1, u=6", 6),
        ("u=1, u=?1", None),
        ("u", None),
        ("i", None),
        ("u=?1", None),
        ('u="1"', None),
        ("u=1.0", None),
        ("u=-1", None),
        ("u=8", None),
        ("u=+1", None),
        ("u=1,", None),
        ("u=1 garbage", None),
        ("u =1", None),
        ("u=1, broken=", None),
        ('u=1, ext="bad\\q"', None),
        ("u=1, ext=1234567890123456", None),
        ("u=1, ext=1.1234", None),
        ("u=1, ext=:a:", None),
        ("u=1, ext=(1\t2)", None),
        ("u=1;\tx=1", None),
    ],
)
def test_priority_structured_dictionary(field, expected):
    from vllm.entrypoints.serve.utils.request_priority import parse_priority_urgency

    assert parse_priority_urgency(field) == expected
