# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for AliceAI MTP configuration."""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.alice_ai_mtp import (
    AliceAIMTP,
    AliceAIMultiTokenPredictor,
)
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    AliceAIForCausalLMConfig,
)
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.alice_ai import AliceAIConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)


def test_native_config_preserves_parent_overrides() -> None:
    """Transformers must preserve the parent's config initialization."""
    config = AliceAIConfig(
        decoder_sparse_step=2,
        mlp_only_layers=[0],
        rope_theta=2_000_000.0,
    )

    assert config.decoder_sparse_step == 2
    assert config.mlp_only_layers == [0]
    assert config.rope_parameters["rope_type"] == "default"
    assert config.rope_parameters["rope_theta"] == 2_000_000.0


def test_alice_config_defaults_match_release_contract() -> None:
    config = AliceAIConfig()

    assert config.decoder_sparse_step == 1
    assert config.mlp_only_layers == []
    assert config.vocab_size == 129024
    assert config.max_position_embeddings == 262144
    assert config.linear_conv_kernel_dim == 4
    assert config.linear_key_head_dim == 128
    assert config.linear_value_head_dim == 128
    assert config.linear_num_key_heads == 32
    assert config.linear_num_value_heads == 32
    assert config.router_score_function == "sigmoid"
    assert config.router_bias_correction is True
    assert config.block_attn_res_block_size == 4
    assert config.kda_allow_negative_eigenvalues is False
    assert config.mtp_num_hidden_layers == 1
    assert config.number_of_conv_states == 3
    assert config.rope_parameters["rope_type"] == "default"
    assert config.rope_parameters["rope_theta"] == 1_000_000.0


def test_native_config_loads_with_auto_map_and_selects_mtp(tmp_path) -> None:
    cfg = {
        "model_type": "alice_ai",
        "architectures": ["AliceAIForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_alice_ai.AliceAIConfig",
            "AutoModelForCausalLM": "modeling_alice_ai.AliceAIForCausalLM",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    config = get_config(tmp_path, trust_remote_code=True)

    assert isinstance(config, AliceAIConfig)
    config = SpeculativeConfig.hf_config_override(config)
    assert config.model_type == "alice_ai_mtp"
    assert config.architectures == ["AliceAIMTP"]
    assert config.n_predict == config.num_nextn_predict_layers == 1
    convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS[config.model_type]
    assert convertor_cls(config, config).get_num_hidden_layers() == 1


@pytest.mark.parametrize(
    "overrides,error_type,error",
    [
        ({}, None, None),
        (
            {"mtp_num_hidden_layers": 2},
            NotImplementedError,
            "mtp_num_hidden_layers=1",
        ),
        (
            {"mtp_num_hidden_layers": True},
            NotImplementedError,
            "mtp_num_hidden_layers=1",
        ),
        (
            {"mtp_num_hidden_layers": 1.0},
            NotImplementedError,
            "mtp_num_hidden_layers=1",
        ),
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
    model_config = SimpleNamespace(hf_config=AliceAIConfig(**overrides))

    if error is None:
        AliceAIForCausalLMConfig.verify_and_update_model_config(model_config)
    else:
        with pytest.raises(error_type, match=error):
            AliceAIForCausalLMConfig.verify_and_update_model_config(model_config)


@pytest.mark.parametrize(
    "pipeline_parallel_size,speculative_method,error",
    [
        (1, None, None),
        (1, "mtp", None),
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


def test_alice_config_hook_is_registered() -> None:
    assert MODELS_CONFIG_MAP["AliceAIForCausalLM"] is AliceAIForCausalLMConfig


def test_mtp_loads_correction_bias_and_shared_weights() -> None:
    model = AliceAIMTP.__new__(AliceAIMTP)
    torch.nn.Module.__init__(model)
    model.model = AliceAIMultiTokenPredictor.__new__(AliceAIMultiTokenPredictor)
    torch.nn.Module.__init__(model.model)
    model.model.config = SimpleNamespace(num_experts=4)
    model.model.is_fused_shared_expert_enabled = False
    gate = torch.nn.Module()
    gate.register_parameter(
        "e_score_correction_bias", torch.nn.Parameter(torch.zeros(4))
    )
    model.model.layers = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"mlp": torch.nn.ModuleDict({"gate": gate})})]
    )
    model.model.embed_tokens = torch.nn.Embedding(2, 4)
    model.lm_head = torch.nn.Linear(4, 2, bias=False)
    bias = torch.arange(4, dtype=torch.float32)
    embedding = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    head = embedding + 1

    loaded = model.load_weights(
        iter(
            [
                ("model.layers.0.mlp.gate.e_score_correction_bias", torch.zeros(9)),
                (
                    "mtp.layers.0.mlp.gate.e_score_correction_bias",
                    bias,
                ),
                ("model.embed_tokens.weight", embedding),
                ("lm_head.weight", head),
            ]
        )
    )

    assert loaded == {
        "model.layers.0.mlp.gate.e_score_correction_bias",
        "model.embed_tokens.weight",
        "lm_head.weight",
    }
    torch.testing.assert_close(gate.e_score_correction_bias, bias)
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding)
    torch.testing.assert_close(model.lm_head.weight, head)
