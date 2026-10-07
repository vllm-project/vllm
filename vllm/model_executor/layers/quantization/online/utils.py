# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import get_current_vllm_config
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe.oracle.base import MoEKernelOracle
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


def get_linear_activation_quant_key(
    default_activation_quant_key: QuantKey | None,
) -> QuantKey | None:
    """Resolve an online linear activation key from config or its method default."""
    args = get_current_vllm_config().model_config.quantization_config
    if isinstance(args, QuantizationConfigArgs) and args.linear is not None:
        if "activation" in args.linear.fields_set:
            return args.linear.activation

    return default_activation_quant_key


def has_explicit_linear_activation_quant_key() -> bool:
    """Whether the online linear activation key was explicitly configured."""
    args = get_current_vllm_config().model_config.quantization_config
    return (
        isinstance(args, QuantizationConfigArgs)
        and args.linear is not None
        and "activation" in args.linear.fields_set
    )


def get_moe_activation_quant_key(
    default_activation_quant_key: QuantKey | None,
) -> QuantKey | None:
    """Resolve an online MoE activation key from config or its method default."""
    activation_key, has_activation_key = (
        MoEKernelOracle.get_user_moe_activation_override()
    )
    if has_activation_key:
        return activation_key

    return default_activation_quant_key
