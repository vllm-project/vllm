# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-agnostic DeepSeek V2/V3 model implementation."""

from .deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
    SharedHead,
)
from .deepseek_v2 import (
    DeepseekAttention,
    DeepseekForCausalLM,
    DeepseekV2Attention,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV2MLAAttention,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV3ForCausalLM,
    GlmMoeDsaForCausalLM,
    Indexer,
    get_spec_layer_idx_from_weight_name,
    yarn_get_mscale,
)

__all__ = [
    "DeepSeekMTP",
    "DeepSeekMultiTokenPredictor",
    "DeepSeekMultiTokenPredictorLayer",
    "DeepSeekV2FusedQkvAProjLinear",
    "DeepseekAttention",
    "DeepseekForCausalLM",
    "DeepseekV2Attention",
    "DeepseekV2DecoderLayer",
    "DeepseekV2ForCausalLM",
    "DeepseekV2MLAAttention",
    "DeepseekV2MLP",
    "DeepseekV2Model",
    "DeepseekV3ForCausalLM",
    "GlmMoeDsaForCausalLM",
    "Indexer",
    "SharedHead",
    "get_spec_layer_idx_from_weight_name",
    "yarn_get_mscale",
]
