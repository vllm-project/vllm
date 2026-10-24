# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.humming import (
    apply_humming_linear,
    get_humming_linear_compute_config,
    prepare_humming_linear_layer_config,
    quant_key_to_input_schema,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.humming import create_humming_locks
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
        # Reuse the CT loader for packed FP4 weights and tensor scales.
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
        replace_parameter(layer, "weight_global_scale", 1.0 / layer.weight_global_scale)
        input_quant_key = None
        if hasattr(layer, "input_global_scale"):
            replace_parameter(
                layer, "input_scale_2", layer.input_global_scale.reshape(-1)
            )
            input_quant_key = kNvfp4Dynamic
        input_schema = quant_key_to_input_schema(input_quant_key)
        self.layer_config = prepare_humming_linear_layer_config(
            layer, quant_config, input_schema=input_schema
        )
        self.compute_config = get_humming_linear_compute_config()
        self.locks = create_humming_locks(layer.weight.device)
        layer.register_buffer("humming_locks", self.locks, persistent=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_humming_linear(
            layer,
            x,
            skip_bias_add=bias is None,
            layer_config=self.layer_config,
            compute_config=self.compute_config,
            locks=self.locks,
        )
