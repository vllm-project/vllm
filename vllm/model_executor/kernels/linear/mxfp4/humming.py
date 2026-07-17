# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    convert_linear_layer_to_humming_standard,
    prepare_humming_layer,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

logger = init_logger(__name__)


class HummingMxFp4LinearKernel(MxFp4LinearKernel):
    """Humming GEMM Kernel for MXFP4."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Humming only supported on CUDA"

        if not current_platform.has_device_capability(75):
            return False, "Humming only supported on SM75+"

        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.weight_quant_key != kMxfp4Static:
            return False, "only supports MXFP4 weights"
        if config.activation_quant_key not in (None, kMxfp4Dynamic, kMxfp4Static):
            return False, "only supports MXFP4 or unquantized activations"
        if config.activation_quant_key is not None:
            logger.warning_once(
                "HummingMxFp4LinearKernel is a weight-only (A16) kernel; "
                "the requested activation quantization (%s) is ignored.",
                config.activation_quant_key,
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale.data = layer.weight_scale.data.view(torch.float8_e8m0fnu)
        name_map = {"weight": "weight", "weight_scale": "weight_scale"}

        quant_config = {
            "quant_method": "humming",
            "dtype": "float4e2m1",
            "scale_dtype": "float8e8m0",
            "group_size": 32,
            "weight_scale_type": "group",
        }

        convert_linear_layer_to_humming_standard(layer=layer, name_map=name_map)
        prepare_humming_layer(layer, quant_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.humming import HummingMethod

        flatten_inputs = x.view(-1, x.size(-1))
        output = HummingMethod.forward_layer(
            layer=layer,
            inputs=flatten_inputs,
            compute_config=layer.compute_config,
        )
        return output.view(*x.shape[:-1], output.size(-1))
