# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
    kMxfp8Static,
)

from ..base import LinearKernel


@dataclass(slots=True)
class Mxfp8LinearLayerConfig:
    """Configuration for an MXFP8 linear layer.

    All MXFP8 layers share the same structure: FP8-E4M3 weights with
    uint8 (E8M0) per-block scales at block size 32.

    Attributes:
        weight_shape: The layer's `(out_features, in_features)`, i.e. `(N, K)`.
        activation_quant_key: Requested activation format; None means weight-only.

    """

    weight_shape: tuple[int, int]
    bmm_batch_size: int | None = None
    activation_quant_key: QuantKey | None = kMxfp8Dynamic


class Mxfp8LinearKernel(LinearKernel[Mxfp8LinearLayerConfig]):
    """Base class for MXFP8 quantized linear kernels.

    Each subclass implements a specific GEMM backend (FlashInfer CUTLASS,
    Marlin, emulation).
    """

    supports_pre_processed_weights: bool = False
    """True if `process_weights_after_loading` only rewrites parameters, so
    weights exported by the weight cache daemon can be used as-is."""

    @classmethod
    def _supports_quant_scheme(
        cls, weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        """Self-quantizing kernels require explicit MXFP8 activations.

        This is separate from input_quant_key(), which describes support for
        externally quantized inputs, not the precision used inside the GEMM.
        """
        return weight_key == kMxfp8Static and activation_key == kMxfp8Dynamic
