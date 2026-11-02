# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
)


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

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        assert self.can_implement(config)[0]
        assert self.is_supported()[0]
        self.config = config

    def input_quant_key(self) -> QuantKey | None:
        """Return the input quantization key supported by this kernel. If the kernel
        does not support input quantization outside of the kernel, return None.
        """
        return None

    def prepare_input(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        *,
        padded_n: int | None = None,
    ) -> QuantizedActivation:
        """Prepare 2D packed NVFP4 data with swizzled 128x4 block scales.

        Tensor inputs are quantized with optional K padding. Prequantized inputs
        are validated and padded without modifying the caller's activation.
        """
        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is None:
            assert isinstance(x, torch.Tensor)
            data, scale = scaled_fp4_quant(
                x,
                layer.input_global_scale_inv,
                is_sf_swizzled_layout=True,
                padded_n=padded_n,
            )
            return QuantizedActivation(data, scale, x.dtype, x.shape, kNvfp4Dynamic)

        data = qa.data.reshape(-1, qa.data.shape[-1])
        if padded_n is not None:
            padding_bytes = padded_n // 2 - data.shape[-1]
            assert padding_bytes >= 0
            data = pad_nvfp4_activation_for_cutlass(data, padding_bytes)
        return replace(qa, data=data)

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
