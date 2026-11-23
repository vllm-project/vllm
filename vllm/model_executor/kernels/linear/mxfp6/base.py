# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp6E2M3Static,
    kMxfp6E3M2Static,
)

from ..base import LinearKernel


@dataclass(slots=True)
class MxFp6LinearLayerConfig:
    """Configuration for an MXFP6 linear layer.

    All MXFP6 layers share the same structure: packed uint8 weights (4 FP6 values
    per 3 bytes) and per-block weight scales (group size 32).

    Attributes:
        weight_quant_key: Identifies the weight quantization format. Can be
            kMxfp6E2M3Static or kMxfp6E3M2Static.
        activation_quant_key: Identifies the activation quantization format,
            or `None` when activations must not be quantized.

    """

    weight_quant_key: QuantKey
    activation_quant_key: QuantKey | None = None


class MxFp6LinearKernel(LinearKernel[MxFp6LinearLayerConfig]):
    """Base class for MXFP6 quantized linear kernels.

    Each subclass implements a specific GEMM backend (CUTLASS, Marlin, etc).
    The kernel selection mechanism iterates over registered subclasses in
    priority order, calling `is_supported`, `_supports_quant_scheme` and
    `can_implement` to find the best match for the current hardware.
    """

    @classmethod
    def _supports_quant_scheme(
        cls, weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        """By default, let registered kernels validate activations in can_implement."""
        return weight_key in (kMxfp6E2M3Static, kMxfp6E3M2Static)
