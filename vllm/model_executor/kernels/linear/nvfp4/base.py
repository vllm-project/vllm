# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


def store_partial_sum_alpha(layer: torch.nn.Module) -> None:
    """With FP16 partial sums (VLLM_TP_FP16_PARTIAL_SUMS), store alpha times the
    2^-k prescale once; ``layer.alpha`` keeps its unscaled value."""
    scale = getattr(layer, "partial_sum_scale", 1.0)
    if getattr(layer, "partial_sum_dtype", None) is not None and scale != 1.0:
        layer.partial_sum_alpha = layer.alpha * scale


def partial_sum_alpha(layer: torch.nn.Module) -> torch.Tensor:
    """``layer.alpha`` with the 2^-k prescale of FP16 partial sums."""
    alpha = getattr(layer, "partial_sum_alpha", None)
    return layer.alpha * layer.partial_sum_scale if alpha is None else alpha


@dataclass
class NvFp4LinearLayerConfig:
    """Configuration for an NVFP4 linear layer.

    All NVFP4 layers share the same structure: packed uint8 weights (2 FP4 values per
    byte), FP8-E4M3 per-block weight scales (group size 16), and scalar global
    scales for both weights and activations.
    """

    pass


class NvFp4LinearKernel(ABC):
    """Base class for NVFP4 quantized linear kernels.

    Each subclass implements a specific GEMM backend (CUTLASS, Marlin, etc).
    The kernel selection mechanism iterates over registered subclasses in
    priority order,calling ``is_supported`` and ``can_implement`` to find the best
    match for the current hardware.
    """

    # True if apply_weights honors layer.partial_sum_dtype/partial_sum_scale
    # (FP16 partial sums of RowParallelLinear, VLLM_TP_FP16_PARTIAL_SUMS).
    supports_partial_sum_dtype: ClassVar[bool] = False

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        assert self.can_implement(config)[0]
        assert self.is_supported()[0]
        self.config = config

    def input_quant_key(self) -> QuantKey | None:
        """Return the input quantization key supported by this kernel. If the kernel
        does not support input quantization outside of the kernel, return None.
        """
        return None

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        """Return whether this kernel can run on the current platform."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
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
