# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import requests

import vllm.envs as envs
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.common.cli_args import BaseFrontendArgs
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
)
from vllm.exceptions import VLLMValidationError
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

    model_config = SimpleNamespace(max_model_len=100)
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
    model_config = SimpleNamespace(max_model_len=100)

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
