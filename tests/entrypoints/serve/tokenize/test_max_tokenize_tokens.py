# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from typing import cast
from unittest.mock import MagicMock

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

import vllm.envs as envs
from tests.utils import RemoteOpenAIServer
from vllm.config import ModelConfig
from vllm.entrypoints.anthropic.api_router import (
    attach_router as attach_anthropic_router,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.common.cli_args import BaseFrontendArgs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.serve.exception_handling.register import (
    init_exception_handler,
)
from vllm.entrypoints.serve.tokenize.api_router import (
    attach_router as attach_tokenize_router,
)
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
)
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.exceptions import VLLMValidationError
from vllm.inputs import TokensPrompt
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.renderers.params import (
    apply_max_tokenize_tokens,
    resolve_max_tokenize_tokens,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.mark.parametrize(
    ("env_value", "cli_value", "expected"),
    [
        pytest.param("5", None, 5, id="environment-variable"),
        pytest.param(None, "7", 7, id="cli-flag"),
        pytest.param("5", "7", 7, id="cli-overrides-environment"),
        pytest.param("0", None, None, id="environment-disables-limit"),
        pytest.param("5", "0", None, id="cli-disables-environment-limit"),
        pytest.param(None, None, 131072, id="default-ceiling"),
    ],
)
def test_max_tokenize_tokens_env_and_cli(monkeypatch, env_value, cli_value, expected):
    # Environment variables may have been cached by earlier tests or server setup.
    envs.disable_envs_cache()
    if env_value is None:
        monkeypatch.delenv("VLLM_MAX_TOKENIZE_TOKENS", raising=False)
    else:
        monkeypatch.setenv("VLLM_MAX_TOKENIZE_TOKENS", env_value)

    parser = BaseFrontendArgs.add_cli_args(FlexibleArgumentParser())
    cli_args = [] if cli_value is None else ["--max-tokenize-tokens", cli_value]
    args = parser.parse_args(cli_args)

    model_config = ModelConfig(max_model_len=100)
    resolved = resolve_max_tokenize_tokens(
        args.max_tokenize_tokens, model_config.max_model_len
    )
    tok_params = apply_max_tokenize_tokens(
        TokenizeCompletionRequest(model="test-model", prompt="hello").build_tok_params(
            model_config
        ),
        resolved,
    )
    assert resolved == expected
    assert tok_params.max_total_tokens == expected


@pytest.mark.parametrize(
    ("max_model_len", "expected"),
    [
        pytest.param(100, 131072, id="default-floor"),
        pytest.param(70000, 140000, id="default-uses-model-length"),
    ],
)
def test_max_tokenize_tokens_default_uses_model_length_floor(
    monkeypatch, max_model_len, expected
):
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_MAX_TOKENIZE_TOKENS", raising=False)
    assert resolve_max_tokenize_tokens(None, max_model_len) == expected


def test_tokenize_request_models_respect_max_total_tokens():
    model_config = ModelConfig(max_model_len=100)

    # 1. TokenizeCompletionRequest
    req_cmpl = TokenizeCompletionRequest(model="test-model", prompt="hello")
    params_cmpl = req_cmpl.build_tok_params(model_config)
    assert apply_max_tokenize_tokens(params_cmpl, 42).max_total_tokens == 42
    assert (
        apply_max_tokenize_tokens(
            replace(params_cmpl, max_total_tokens=10), 42
        ).max_total_tokens
        == 10
    )

    # 2. TokenizeChatRequest
    req_chat = TokenizeChatRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )
    params_chat = req_chat.build_tok_params(model_config)
    assert apply_max_tokenize_tokens(params_chat, 42).max_total_tokens == 42
    assert (
        apply_max_tokenize_tokens(
            replace(params_chat, max_total_tokens=10), 42
        ).max_total_tokens
        == 10
    )

    # 3. DetokenizeRequest
    req_detok = DetokenizeRequest(model="test-model", tokens=[1, 2, 3])
    params_detok = req_detok.build_tok_params(model_config)
    assert apply_max_tokenize_tokens(params_detok, 42).max_total_tokens == 42
    assert (
        apply_max_tokenize_tokens(
            replace(params_detok, max_total_tokens=10), 42
        ).max_total_tokens
        == 10
    )

    # 4. Exceeding max_tokenize_tokens raises VLLMValidationError
    with pytest.raises(VLLMValidationError):
        apply_max_tokenize_tokens(replace(params_cmpl, max_total_tokens=50), 42)


def test_tokenize_request_rejects_over_limit_requests():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-tokenize-tokens",
        "5",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        response = requests.post(
            remote_server.url_for("tokenize"),
            json={"model": MODEL_NAME, "prompt": "hello " * 10},
        )

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == "input_tokens"
    assert "maximum context length" in error["message"]


def _build_harmony_online_renderer() -> OnlineRenderer:
    model_config = MagicMock()
    model_config.hf_config.model_type = "gpt_oss"
    model_config.model = "openai/gpt-oss-20b"
    model_config.served_model_name = "gpt-oss"
    model_config.multimodal_config = None
    model_config.max_model_len = 8192
    model_config.max_tokenize_tokens = None

    renderer = MagicMock()
    renderer.tokenizer = MagicMock()
    renderer.get_tokenizer.return_value = renderer.tokenizer

    return OnlineRenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


def test_harmony_tokenize_and_count_tokens_max_limit():
    model_config = MagicMock()
    model_config.hf_config.model_type = "gpt_oss"
    model_config.model = "openai/gpt-oss-20b"
    model_config.served_model_name = "gpt-oss"
    model_config.multimodal_config = None
    model_config.max_model_len = 8192
    model_config.max_tokenize_tokens = None

    renderer = MagicMock()
    renderer.tokenizer = MagicMock()
    renderer.get_tokenizer.return_value = renderer.tokenizer

    online_renderer = OnlineRenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    assert online_renderer.use_harmony is True

    models = OpenAIModelRegistry(
        model_config=model_config,
        base_model_paths=[
            BaseModelPath(name="gpt-oss", model_path="openai/gpt-oss-20b")
        ],
    )

    serving_tok = ServingTokenization(
        models=models,
        online_renderer=online_renderer,
        chat_template=None,
        chat_template_content_format="auto",
        max_tokenize_tokens=3,
    )

    engine_client = MagicMock()
    engine_client.errored = False

    serving_anthropic = AnthropicServingMessages(
        engine_client=engine_client,
        models=models,
        response_role="assistant",
        online_renderer=online_renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        max_tokenize_tokens=3,
    )

    parser = FlexibleArgumentParser()
    BaseFrontendArgs.add_cli_args(parser)

    app = FastAPI()
    init_exception_handler(app)
    app.state.args = parser.parse_args([])
    app.state.serving_tokenization = serving_tok
    app.state.anthropic_serving_messages = serving_anthropic
    attach_tokenize_router(app)
    attach_anthropic_router(app)

    client = TestClient(app)
    payload = {
        "model": "gpt-oss",
        "messages": [
            {"role": "user", "content": "Hello world, testing harmony limit."}
        ],
    }

    tok_resp = client.post("/tokenize", json=payload)
    assert tok_resp.status_code == 400
    assert "maximum context length is 3 tokens" in tok_resp.text

    count_resp = client.post("/v1/messages/count_tokens", json=payload)
    assert count_resp.status_code == 400
    assert "maximum context length is 3 tokens" in count_resp.text


@pytest.mark.parametrize("blank_env", ["", "   "])
def test_blank_env_var_startup_and_cli_precedence(
    monkeypatch: pytest.MonkeyPatch, blank_env: str
):
    # Blank or whitespace-only env vars should fall back to None without crashing
    monkeypatch.setenv("VLLM_MAX_TOKENIZE_TOKENS", blank_env)
    envs.enable_envs_cache()
    try:
        assert envs.VLLM_MAX_TOKENIZE_TOKENS is None

        # CLI flag should take precedence over the environment variable
        parser = FlexibleArgumentParser()
        BaseFrontendArgs.add_cli_args(parser)
        args = parser.parse_args(["--max-tokenize-tokens", "7"])
        assert args.max_tokenize_tokens == 7
    finally:
        envs.disable_envs_cache()


@pytest.mark.asyncio
async def test_harmony_chat_without_tok_params_does_not_truncate():
    online_renderer = _build_harmony_online_renderer()
    request = ChatCompletionRequest(
        model="gpt-oss",
        messages=[{"role": "user", "content": "Hello world, testing harmony chat."}],
        truncate_prompt_tokens=3,
    )

    _, engine_inputs = await online_renderer.render_chat(request)
    assert len(engine_inputs) == 1
    engine_input = cast(TokensPrompt, engine_inputs[0])
    # Ordinary chat rendering (tok_params=None) must preserve all encoded tokens
    # rather than truncating to 3 tokens.
    assert len(engine_input["prompt_token_ids"]) > 3
