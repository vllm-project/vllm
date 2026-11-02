# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration guards of `enable_return_last_hidden_states`."""

from types import SimpleNamespace

import pytest

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs


def _config(**overrides) -> SimpleNamespace:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_return_last_hidden_states=True,
            runner_type="generate",
            is_diffusion=False,
        ),
        use_v2_model_runner=True,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    for key, value in overrides.items():
        target, _, attr = key.rpartition("__")
        setattr(getattr(config, target) if target else config, attr, value)
    return config


def test_supported_config_passes():
    VllmConfig._verify_last_hidden_states_config(_config())


def test_disabled_flag_skips_every_check():
    VllmConfig._verify_last_hidden_states_config(
        _config(
            model_config__enable_return_last_hidden_states=False,
            use_v2_model_runner=False,
            speculative_config=object(),
        )
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"use_v2_model_runner": False}, "Model Runner V2"),
        ({"model_config__runner_type": "pooling"}, "autoregressive"),
        ({"model_config__is_diffusion": True}, "autoregressive"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"parallel_config__pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"parallel_config__decode_context_parallel_size": 2}, "context parallelism"),
        (
            {"parallel_config__prefill_context_parallel_size": 2},
            "context parallelism",
        ),
    ],
)
def test_unsupported_config_is_rejected(overrides: dict, match: str):
    with pytest.raises(ValueError, match=match):
        VllmConfig._verify_last_hidden_states_config(_config(**overrides))


def test_engine_arg_reaches_model_config():
    enabled = EngineArgs(
        model="facebook/opt-125m", enable_return_last_hidden_states=True
    )
    default = EngineArgs(model="facebook/opt-125m")
    assert enabled.create_model_config().enable_return_last_hidden_states
    assert not default.create_model_config().enable_return_last_hidden_states
