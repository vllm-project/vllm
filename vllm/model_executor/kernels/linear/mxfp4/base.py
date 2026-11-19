# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Dynamic,
    kMxfp4Static,
)

from ..base import LinearKernel


@dataclass(slots=True)
class MxFp4LinearLayerConfig:
    """Configuration for an MXFP4 linear layer.

    All MXFP4 layers share the same structure: packed uint8 weights (2 FP4 values per
    byte) and per-block weight scales (group size 32).

    Attributes:
        activation_quant_key: Identifies the activation quantization format,
            or `None` when activations must not be quantized.

    """

    activation_quant_key: QuantKey | None = None


class MxFp4LinearKernel(LinearKernel[MxFp4LinearLayerConfig]):
    """Base class for MXFP4 quantized linear kernels.

    Each subclass implements a specific GEMM backend (CUTLASS, Marlin, etc).
    The kernel selection mechanism iterates over registered subclasses in
    priority order, calling `is_supported`, `_supports_quant_scheme` and
    `can_implement` to find the best match for the current hardware.
    """

    @classmethod
    def _supports_quant_scheme(
        cls,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Return whether this kernel implements *weight_key* x *activation_key*.

        Selection uses this to dispatch on the requested activation format
        (from the checkpoint or `--quantization-config.linear.activation`).
        The default covers MXFP4 weights with MXFP4 or unquantized activations;
        kernels for other activation formats override it.
        """
        return weight_key == kMxfp4Static and activation_key in (None, kMxfp4Dynamic)

    def fuse_input_hadamard(self, size: int) -> bool:
        """Rotate the input by the size-`size` block Hadamard matrix inside
        `apply_weights` from now on, if supported (see
        `InputHadamardFusable`). Call after
        `process_weights_after_loading`."""
        return False
