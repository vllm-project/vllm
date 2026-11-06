# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.humming import (
    apply_humming_linear,
    get_humming_linear_compute_config,
    prepare_humming_linear_layer_config,
    quant_key_to_input_schema,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
from vllm.model_executor.utils import (
    move_parameter,
    replace_parameter,
    set_kernel_parameter,
)
from vllm.platforms import current_platform
from vllm.utils.humming import create_humming_locks
from vllm.utils.import_utils import has_humming

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


class HummingNvFp4LinearKernel(NvFp4LinearKernel):
    """Humming GEMM Kernel for NVFP4."""

    allow_input_schema_fallback = True

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
        # CT pack-quantized reads `weight_packed`: make it the weight's runtime name
        # (a CT checkpoint already has it as an alias of `weight`).
        move_parameter(layer, "weight", "weight_packed")
        # The CT linear scheme inverts the global scale (1/scale) for
        # marlin/cutlass; humming wants the original.
        replace_parameter(layer, "weight_global_scale", 1.0 / layer.weight_global_scale)
        input_quant_key = None
        if hasattr(layer, "input_global_scale"):
            # Quark keeps its loaded (retired) `input_scale_2`: revive that one.
            if "input_scale_2" not in layer._parameters:
                move_parameter(layer, "input_global_scale", "input_scale_2")
            set_kernel_parameter(
                layer, "input_scale_2", layer.input_global_scale.reshape(-1)
            )
            input_quant_key = kNvfp4Dynamic
        input_schema = quant_key_to_input_schema(input_quant_key)
        self.layer_config = prepare_humming_linear_layer_config(
            layer,
            quant_config,
            input_schema=input_schema,
            allow_input_schema_fallback=self.allow_input_schema_fallback,
        )
        self.compute_config = get_humming_linear_compute_config()
        self.locks = create_humming_locks(layer.weight.device)

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


class HummingHadamardNvFp4LinearKernel(HummingNvFp4LinearKernel):
    """Humming W4A4 NVFP4 with a Sylvester Hadamard fused into input quantization.

    Runs the activation dtype the checkpoint declares, without A16 fallback.
    """

    allow_input_schema_fallback = False

    def __init__(
        self, config: NvFp4LinearLayerConfig, hadamard_block_size: int
    ) -> None:
        super().__init__(config)
        self.hadamard_block_size = hadamard_block_size

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.has_device_capability(100):
            return False, "Humming FP4 activations require SM100+"
        return super().is_supported(compute_capability)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.weight_schema = dataclasses.replace(
            layer.weight_schema, hadamard_block_size=self.hadamard_block_size
        )
