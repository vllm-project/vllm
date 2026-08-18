# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    apply_humming_linear,
    get_humming_linear_compute_config,
    prepare_humming_linear_layer_config,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


class HummingNvFp4LinearKernel(NvFp4LinearKernel):
    """Humming GEMM Kernel for NVFP4."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Humming only supported on CUDA"

        if not has_humming():
            return False, "Humming is not installed"

        if not current_platform.has_device_capability(75):
            return False, "Humming only supported on SM75+"

        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Route through humming's compressed-tensors nvfp4 loader (same path as
        # the MoE oracle); the native group_tensor schema mishandles a scalar
        # global scale.
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "type": "float",
            "num_bits": 4,
            "strategy": "group",
            "group_size": 16,
        }
        # CT pack-quantized reads `weight_packed`; the scheme renamed it to `weight`.
        if not hasattr(layer, "weight_packed"):
            layer.weight_packed = layer.weight
            del layer.weight
        # The CT linear scheme inverts the global scale (1/scale) for
        # marlin/cutlass; humming wants the original.
        layer.weight_global_scale = torch.nn.Parameter(
            1.0 / layer.weight_global_scale, requires_grad=False
        )
        self.layer_config = prepare_humming_linear_layer_config(layer, quant_config)
        self.compute_config = get_humming_linear_compute_config()
        self.locks = torch.zeros(1024, dtype=torch.int32, device=layer.weight.device)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_humming_linear(
            layer,
            x,
            layer_config=self.layer_config,
            compute_config=self.compute_config,
            locks=self.locks,
        )
