# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Helpers for resolving online quantization activation overrides."""

from typing import TYPE_CHECKING

from vllm.config import get_current_vllm_config
from vllm.config.quantization import QuantizationConfigArgs

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


def get_linear_activation_quant_key(
    default: "QuantKey | None",
) -> "QuantKey | None":
    """Resolve the linear activation quantization key."""
    quantization_config_args = (
        get_current_vllm_config().model_config.quantization_config
    )

    assert quantization_config_args is None or isinstance(
        quantization_config_args, QuantizationConfigArgs
    )

    if (
        quantization_config_args is None
        or quantization_config_args.linear is None
        or "activation" not in quantization_config_args.linear.fields_set
    ):
        return default

    return quantization_config_args.linear.activation


def get_moe_activation_quant_key(
    default: "QuantKey | None",
) -> "QuantKey | None":
    """Resolve the MoE activation quantization key."""
    quantization_config_args = (
        get_current_vllm_config().model_config.quantization_config
    )

    assert quantization_config_args is None or isinstance(
        quantization_config_args, QuantizationConfigArgs
    )
    if (
        quantization_config_args is None
        or quantization_config_args.moe is None
        or "activation" not in quantization_config_args.moe.fields_set
    ):
        return default

    return quantization_config_args.moe.activation
