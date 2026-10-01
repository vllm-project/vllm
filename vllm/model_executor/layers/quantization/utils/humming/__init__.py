# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Humming quantization integration."""

from vllm.model_executor.layers.quantization.utils.humming.activation import (
    get_humming_activation,
)
from vllm.model_executor.layers.quantization.utils.humming.linear import (
    apply_humming_linear,
    convert_linear_layer_to_humming_standard,
    get_humming_linear_compute_config,
    prepare_humming_linear_layer_config,
)
from vllm.model_executor.layers.quantization.utils.humming.moe import (
    HummingMoEQuantConfig,
    make_humming_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.humming.schema import (
    check_and_fallback_input_schema,
    humming_is_layer_skipped,
    humming_update_schema_hadamard_block_size,
    input_schema_to_quant_key,
    quant_key_to_input_schema,
    resolve_humming_layer_config,
    weight_schema_to_quant_key,
)

__all__ = [
    "check_and_fallback_input_schema",
    "get_humming_activation",
    "HummingMoEQuantConfig",
    "humming_is_layer_skipped",
    "humming_update_schema_hadamard_block_size",
    "make_humming_moe_quant_config",
    "apply_humming_linear",
    "convert_linear_layer_to_humming_standard",
    "get_humming_linear_compute_config",
    "prepare_humming_linear_layer_config",
    "input_schema_to_quant_key",
    "quant_key_to_input_schema",
    "resolve_humming_layer_config",
    "weight_schema_to_quant_key",
]
