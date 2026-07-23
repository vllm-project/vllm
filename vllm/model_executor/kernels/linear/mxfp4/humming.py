# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.humming_utils import (
    apply_humming_linear,
    convert_linear_layer_to_humming_standard,
    get_humming_linear_compute_config,
    prepare_humming_linear_layer_config,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


class HummingMxFp4LinearKernel(MxFp4LinearKernel):
    """Humming GEMM Kernel for MXFP4."""

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
    def can_implement(cls, c: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
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
