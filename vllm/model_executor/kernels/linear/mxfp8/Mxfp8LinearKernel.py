# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from ..base import LinearKernel


@dataclass(slots=True)
class Mxfp8LinearLayerConfig:
    """Configuration for an MXFP8 linear layer.

    All MXFP8 layers share the same structure: FP8-E4M3 weights with
    uint8 (E8M0) per-block scales at block size 32.

    Attributes:
        weight_shape: The layer's `(out_features, in_features)`, i.e. `(N, K)`.

    """

    weight_shape: tuple[int, int]
    bmm_batch_size: int | None = None


class Mxfp8LinearKernel(LinearKernel[Mxfp8LinearLayerConfig]):
    """Base class for MXFP8 quantized linear kernels.

    Each subclass implements a specific GEMM backend (FlashInfer CUTLASS,
    Marlin, emulation).
    """

    supports_pre_processed_weights: bool = False
    """True if `process_weights_after_loading` only rewrites parameters, so
    weights exported by the weight cache daemon can be used as-is."""

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        """Whether a row's output is independent of the other rows in the batch."""
        return False
