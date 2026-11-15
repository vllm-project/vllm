# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.serve.tokenize.protocol import TokenizeCompletionRequest
from vllm.renderers.params import resolve_max_tokenize_tokens
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _parse_engine_args(*args: str) -> EngineArgs:
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    return EngineArgs.from_cli_args(parser.parse_args(list(args)))


def _model_config(max_tokenize_tokens: int | None):
    # The resolver only needs these config fields; avoid initializing a model.
    return SimpleNamespace(max_tokenize_tokens=max_tokenize_tokens, max_model_len=100)


@pytest.mark.parametrize(
    ("env_value", "cli_value", "expected"),
    [
        ("5", None, 5),  # env var supplies the limit
        ("5", "7", 7),  # CLI value takes precedence over the environment
        ("5", "0", None),  # CLI can explicitly disable an env-provided limit
        ("0", None, None),  # env var can disable the default limit
        (None, None, 131072),  # unset falls back to the documented default
    ],
)
def test_max_tokenize_tokens_env_and_cli(monkeypatch, env_value, cli_value, expected):
    if env_value is None:
        monkeypatch.delenv("VLLM_MAX_TOKENIZE_TOKENS", raising=False)
    else:
        monkeypatch.setenv("VLLM_MAX_TOKENIZE_TOKENS", env_value)
    # Re-read the environment-backed setting after monkeypatching.
    envs.environment_variables["VLLM_MAX_TOKENIZE_TOKENS"] = lambda: (
        int(env_value) if env_value is not None else None
    )

    cli_args = [] if cli_value is None else ["--max-tokenize-tokens", cli_value]
    parsed_args = _parse_engine_args(*cli_args)
    model_config = _model_config(parsed_args.max_tokenize_tokens)

    assert resolve_max_tokenize_tokens(model_config) == expected
    request = TokenizeCompletionRequest(model="test-model", prompt="hello")
    assert request.build_tok_params(model_config).max_total_tokens == expected
