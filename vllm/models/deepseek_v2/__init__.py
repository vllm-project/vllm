# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V2/V3 model package.

The implementation currently lives under ``hw_agnostic/``. This package is
the registry-facing entry point and re-exports the public classes used by
model loading, matching the import style used by ``vllm.models.deepseek_v4``.
"""

from .hw_agnostic import (
    DeepseekAttention,
    DeepseekForCausalLM,
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
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
    SharedHead,
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
