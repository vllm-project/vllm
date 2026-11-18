# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import requests

import vllm.envs as envs
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.common.cli_args import BaseFrontendArgs
from vllm.entrypoints.serve.tokenize.protocol import TokenizeCompletionRequest
from vllm.renderers.params import resolve_max_tokenize_tokens
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

    # The explicit ModelConfig field is populated from FrontendArgs when set;
    # otherwise the resolver consults the environment and then the fallback.
    model_config = SimpleNamespace(
        max_tokenize_tokens=args.max_tokenize_tokens,
        max_model_len=100,
    )
    tok_params = TokenizeCompletionRequest(
        model="test-model", prompt="hello"
    ).build_tok_params(model_config)

    assert resolve_max_tokenize_tokens(model_config) == expected
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
    model_config = SimpleNamespace(
        max_tokenize_tokens=None, max_model_len=max_model_len
    )
    assert resolve_max_tokenize_tokens(model_config) == expected


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
