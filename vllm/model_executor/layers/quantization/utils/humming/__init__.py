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
    convert_to_humming_moe_kernel_format,
    get_humming_moe_quant_config,
    make_humming_moe_kernel,
    make_humming_moe_quant_config,
    select_humming_moe_experts,
)
from vllm.model_executor.layers.quantization.utils.humming.schema import (
    check_and_fallback_input_schema,
    humming_is_layer_skipped,
    input_schema_to_quant_key,
    quant_key_to_input_schema,
    resolve_humming_layer_config,
    weight_schema_to_quant_key,
)

__all__ = [
    "check_and_fallback_input_schema",
    "get_humming_activation",
    "HummingMoEQuantConfig",
    "get_humming_moe_quant_config",
    "humming_is_layer_skipped",
    "make_humming_moe_quant_config",
    "apply_humming_linear",
    "convert_linear_layer_to_humming_standard",
    "get_humming_linear_compute_config",
    "prepare_humming_linear_layer_config",
    "make_humming_moe_kernel",
    "select_humming_moe_experts",
    "input_schema_to_quant_key",
    "quant_key_to_input_schema",
    "resolve_humming_layer_config",
    "weight_schema_to_quant_key",
    "convert_to_humming_moe_kernel_format",
]
