# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for AliceAI configuration."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.config import AliceAIForCausalLMConfig


def _make_config(**overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "kda_allow_negative_eigenvalues": False,
            "router_score_function": "sigmoid",
            "router_bias_correction": True,
            **overrides,
        }
    )


def test_model_config_warns_when_mtp_is_ignored(caplog) -> None:
    config = _make_config(mtp_num_hidden_layers=1)

    with caplog.at_level("WARNING"):
        AliceAIForCausalLMConfig.verify_and_update_model_config(
            SimpleNamespace(hf_config=config)
        )
    assert "ignores the MTP configuration and weights" in caplog.text


@pytest.mark.parametrize(
    "overrides,error_type,error",
    [
        ({}, None, None),
        (
            {"kda_allow_negative_eigenvalues": True},
            NotImplementedError,
            "kda_allow_negative_eigenvalues=false",
        ),
        (
            {"router_score_function": "softmax"},
            ValueError,
            "router_score_function='sigmoid'",
        ),
        (
            {"router_bias_correction": False},
            ValueError,
            "router_bias_correction=true",
        ),
    ],
)
def test_model_config_rejects_unsupported_alice_contract(
    overrides, error_type, error
) -> None:
    model_config = SimpleNamespace(hf_config=_make_config(**overrides))

    if error is None:
        AliceAIForCausalLMConfig.verify_and_update_model_config(model_config)
    else:
        with pytest.raises(error_type, match=error):
            AliceAIForCausalLMConfig.verify_and_update_model_config(model_config)


@pytest.mark.parametrize(
    "pipeline_parallel_size,speculative_method,error",
    [
        (1, None, None),
        (2, None, "pipeline_parallel_size=1"),
        (1, "eagle3", "eagle3"),
        (1, "dflash", "dflash"),
        (1, "dspark", "dspark"),
        (1, "extract_hidden_states", "extract_hidden_states"),
    ],
)
def test_runtime_config_rejects_unsupported_alice_modes(
    pipeline_parallel_size, speculative_method, error
) -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size),
        speculative_config=(
            None
            if speculative_method is None
            else SimpleNamespace(method=speculative_method)
        ),
    )

    if error is None:
        AliceAIForCausalLMConfig.verify_and_update_config(vllm_config)
    else:
        with pytest.raises(NotImplementedError, match=error):
            AliceAIForCausalLMConfig.verify_and_update_config(vllm_config)
