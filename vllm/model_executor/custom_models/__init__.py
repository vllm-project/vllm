# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for integrating custom model implementations with vLLM."""

from vllm.model_executor.custom_models.attention_replacement import (
    replace_with_trainable_attention,
)
from vllm.model_executor.custom_models.custom_model_wrapper import (
    VLLMModelForCausalLM,
)
from vllm.model_executor.custom_models.trainable_attention import (
    TrainableFlashAttention,
)
from vllm.model_executor.custom_models.trainable_mla_attention import (
    MLAConfig,
    TrainableMLA,
)
from vllm.model_executor.custom_models.utils import (
    convert_freqs_cis_to_real,
    create_mla_kv_cache_spec,
    load_external_weights,
    store_positions_in_context,
)

__all__ = [
    # Attention modules
    "TrainableFlashAttention",
    "TrainableMLA",
    "MLAConfig",
    "replace_with_trainable_attention",
    # Base wrapper
    "VLLMModelForCausalLM",
    # Utilities
    "convert_freqs_cis_to_real",
    "create_mla_kv_cache_spec",
    "load_external_weights",
    "store_positions_in_context",
]
