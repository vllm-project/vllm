# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PretrainedConfig

from vllm.config.speculative import MTPModelTypes, SpeculativeConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    BailingHybridMTPModelArchConfigConvertor,
)


def _bailing_config() -> PretrainedConfig:
    config = PretrainedConfig(
        architectures=["BailingMoeV2_5ForCausalLM"],
        hidden_size=4096,
        kv_lora_rank=512,
        num_attention_heads=32,
        num_experts=256,
        num_hidden_layers=32,
        num_key_value_heads=32,
        num_nextn_predict_layers=1,
        qk_rope_head_dim=64,
        vocab_size=157184,
    )
    config.model_type = "bailing_hybrid"
    return config


def test_bailing_hybrid_mtp_hf_config_override():
    config = _bailing_config()

    overridden = SpeculativeConfig.hf_config_override(config)

    assert overridden.model_type == "bailing_hybrid_mtp"
    assert overridden.architectures == ["BailingMoeV25MTPModel"]
    assert overridden.n_predict == 1
    assert "bailing_hybrid_mtp" in MTPModelTypes.__args__


def test_bailing_hybrid_mtp_model_arch_config():
    config = _bailing_config()
    config.model_type = "bailing_hybrid_mtp"
    config.architectures = ["BailingMoeV25MTPModel"]

    model_arch_config = BailingHybridMTPModelArchConfigConvertor(
        config, config
    ).convert()

    assert model_arch_config.model_type == "bailing_hybrid_mtp"
    assert model_arch_config.architectures == ["BailingMoeV25MTPModel"]
    assert model_arch_config.total_num_hidden_layers == 1
    assert model_arch_config.is_deepseek_mla
