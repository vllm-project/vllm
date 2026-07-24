# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import GlmMoeDsaConfig

from vllm.model_executor.layers.fused_moe.expert_replacement import (
    get_mone_expert_ids,
)
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.configs import (
    DeepseekV2CompressedConfig,
    MiniMaxM2CompressedConfig,
    OlmoeCompressedConfig,
)
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)


def test_legacy_mone_config_types_are_registered():
    assert _CONFIG_REGISTRY["deepseek_v2_compressed"] is DeepseekV2CompressedConfig
    assert _CONFIG_REGISTRY["minimax_m2_compressed"] is MiniMaxM2CompressedConfig
    assert _CONFIG_REGISTRY["olmoe_compressed"] is OlmoeCompressedConfig


def test_legacy_mone_metadata_is_preserved():
    deepseek = DeepseekV2CompressedConfig.from_dict(
        {"approximate_experts": {"1": [2, 5]}}
    )
    minimax = MiniMaxM2CompressedConfig(
        use_routing_bias=True,
        scoring_func="sigmoid",
        approximate_experts={"0": [1, 3]},
    )
    olmoe = OlmoeCompressedConfig(approximate_experts={"1": [2]})

    assert minimax.approximate_experts == {0: [1, 3]}
    assert olmoe.approximate_experts == {"1": [2]}
    assert deepseek.approximate_experts == {"1": [2, 5]}
    assert ModelArchConfigConvertorBase(deepseek, deepseek).is_deepseek_mla()


def test_native_glm_mone_metadata_uses_shared_deepseek_path():
    glm = GlmMoeDsaConfig.from_dict(
        {
            "approximate_experts": {"3": [0, 7, 255]},
            "kv_lora_rank": 512,
            "model_type": "glm_moe_dsa",
        }
    )

    assert get_mone_expert_ids(glm, 3) == (0, 7, 255)
    assert ModelArchConfigConvertorBase(glm, glm).is_deepseek_mla()
