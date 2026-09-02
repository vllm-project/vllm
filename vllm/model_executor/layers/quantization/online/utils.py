# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Helpers for resolving online quantization activation overrides."""

from typing import TYPE_CHECKING, Literal

from vllm.config import get_current_vllm_config
from vllm.config.quantization import QuantizationConfigArgs

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


def get_activation_quant_key(
    default: "QuantKey | None",
    layer_kind: Literal["linear", "moe"],
) -> "QuantKey | None":
    """Resolve an online layer kind's activation quantization key."""
    quantization_config_args = (
        get_current_vllm_config().model_config.quantization_config
    )

    assert quantization_config_args is None or isinstance(
        quantization_config_args, QuantizationConfigArgs
    )

    spec = (
        getattr(quantization_config_args, layer_kind, None)
        if quantization_config_args is not None
        else None
    )
    if spec is None or "activation" not in spec.fields_set:
        return default

    return spec.activation
