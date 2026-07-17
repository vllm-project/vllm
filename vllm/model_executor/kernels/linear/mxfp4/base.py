# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


@dataclass
class MxFp4LinearLayerConfig:
    """Configuration for an MXFP4 linear layer.

    All MXFP4 layers share the same structure: packed uint8 weights (2 FP4 values per
    byte) and per-block weight scales (group size 32).

    weight_quant_key: identifies the weight quantization format (e.g. OCP MXFP4,
        MXFP6). Kernels compare this against the exact format(s) they implement.
    activation_quant_key: identifies the activation quantization format, or None
        when the caller has no explicit expectation (e.g. a weight-only a16
        kernel may be selected, or a kernel may quantize activations itself).
    """

    weight_quant_key: QuantKey
    activation_quant_key: QuantKey | None = None


class MxFp4LinearKernel(ABC):
    """Base class for MXFP4 quantized linear kernels.

    Each subclass implements a specific GEMM backend (CUTLASS, Marlin, etc).
    The kernel selection mechanism iterates over registered subclasses in
    priority order,calling ``is_supported`` and ``can_implement`` to find the best
    match for the current hardware.
    """

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        assert self.can_implement(config)[0]
        assert self.is_supported()[0]
        self.config = config

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        """Return whether this kernel can run on the current platform."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        """Return whether this kernel can handle *config*."""
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Transform weights into the format required by this kernel.

        Called once after checkpoint weights have been loaded onto the
        device.  Implementations should repack / swizzle / pad weights
        and scales in-place on *layer*.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the quantized GEMM."""
        raise NotImplementedError
