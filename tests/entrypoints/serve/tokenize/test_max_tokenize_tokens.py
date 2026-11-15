# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.serve.tokenize.protocol import TokenizeCompletionRequest
from vllm.renderers.params import resolve_max_tokenize_tokens
from vllm.utils.argparse_utils import FlexibleArgumentParser


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
    # Environment variables may have been cached by other tests or server setup.
    envs.disable_envs_cache()
    if env_value is None:
        monkeypatch.delenv("VLLM_MAX_TOKENIZE_TOKENS", raising=False)
    else:
        monkeypatch.setenv("VLLM_MAX_TOKENIZE_TOKENS", env_value)

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    cli_args = [] if cli_value is None else ["--max-tokenize-tokens", cli_value]
    args = EngineArgs.from_cli_args(parser.parse_args(cli_args))

    # The explicit ModelConfig field is populated from EngineArgs when set;
    # otherwise the resolver consults the environment and then the fallback.
    model_config = SimpleNamespace(max_tokenize_tokens=args.max_tokenize_tokens,
                                   max_model_len=100)
    tok_params = TokenizeCompletionRequest(
        model="test-model", prompt="hello"
    ).build_tok_params(model_config)

    assert resolve_max_tokenize_tokens(model_config) == expected
    assert tok_params.max_total_tokens == expected
