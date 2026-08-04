# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.utils.flashinfer import has_flashinfer_cutedsl_nvfp4_quant

if TYPE_CHECKING:
    from vllm.config.kernel import NvFp4InputQuantBackend


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

    # Subclasses that route their NVFP4 activation quantization through
    # FlashInfer (see ``KernelConfig.nvfp4_input_quant_backend``) set this to True.
    uses_flashinfer_input_quant: bool = False

    # Resolved per instance in __init__; the class-level default keeps
    # input_quant_key() safe on instances probed without running __init__.
    input_quant_backend: "NvFp4InputQuantBackend" = "auto"

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        assert self.can_implement(config)[0]
        assert self.is_supported()[0]
        self.config = config
        self.input_quant_backend = self._resolve_input_quant_backend()

    def _resolve_input_quant_backend(self) -> "NvFp4InputQuantBackend":
        """Resolve once at setup which backend performs this kernel's activation
        quant, so apply_weights only reads a plain attribute."""
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is None:
            return "auto"
        backend = config.kernel_config.nvfp4_input_quant_backend
        if backend != "flashinfer_cutedsl":
            return backend
        # An explicit backend request that cannot be honored is an error, matching
        # how linear_backend rejects unsatisfiable selections.
        if not self.uses_flashinfer_input_quant:
            raise ValueError(
                f"nvfp4_input_quant_backend=flashinfer_cutedsl was requested but "
                f"{type(self).__name__} does not route activation quant through "
                f"FlashInfer. Select a FlashInfer NVFP4 linear backend (e.g. "
                f"flashinfer_cutlass) or set nvfp4_input_quant_backend=auto."
            )
        if not has_flashinfer_cutedsl_nvfp4_quant():
            raise ValueError(
                "nvfp4_input_quant_backend=flashinfer_cutedsl requires SM100+ and a "
                "FlashInfer build with CuTe-DSL available "
                "(flashinfer.cute_dsl.is_cute_dsl_available())."
            )
        return backend

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
