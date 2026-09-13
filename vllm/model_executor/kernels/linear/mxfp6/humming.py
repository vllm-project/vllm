# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.humming import (
    apply_humming_linear,
    convert_linear_layer_to_humming_standard,
    get_humming_linear_compute_config,
    prepare_humming_linear_layer_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kMxfp4Dynamic,
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

from .base import MxFp6LinearKernel, MxFp6LinearLayerConfig


class HummingMxFp6LinearKernel(MxFp6LinearKernel):
    """Humming GEMM for packed MXFP6 E2M3 and E3M2 weights."""

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
    def can_implement(cls, config: MxFp6LinearLayerConfig) -> tuple[bool, str | None]:
        if config.weight_quant_key not in (kMxfp6E2M3Static, kMxfp6E3M2Static):
            return False, "only supports MXFP6 E2M3 or E3M2 weights"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_dtype = "float6e3m2"
        if self.config.weight_quant_key == kMxfp6E2M3Static:
            weight_dtype = "float6e2m3"

        quant_config = {
            "quant_method": "humming",
            "dtype": weight_dtype,
            "scale_dtype": "float8e8m0",
            "group_size": 32,
            "weight_scale_type": "group",
        }
        layer.weight_scale.data = layer.weight_scale.data.view(torch.float8_e8m0fnu)

        convert_linear_layer_to_humming_standard(
            layer=layer,
            name_map={"weight": "weight", "weight_scale": "weight_scale"},
        )
        input_quant_config = None
        key = self.config.activation_quant_key
        if key is not None:
            input_dtype = {
                kMxfp4Dynamic: "float4e2m1",
                kMxfp6E2M3Dynamic: "float6e2m3",
                kMxfp6E3M2Dynamic: "float6e3m2",
                kMxfp8Dynamic: "float8e4m3",
                kFp8DynamicTokenSym: "float8e4m3",
            }[key]
            scale_dtype = "float32" if key == kFp8DynamicTokenSym else "float8e8m0"
            input_quant_config = {
                "quant_method": "humming",
                "dtype": input_dtype,
                "group_size": 0 if key == kFp8DynamicTokenSym else 32,
                "scale_dtype": scale_dtype,
            }
        self.layer_config = prepare_humming_linear_layer_config(
            layer, quant_config, input_quant_config
        )
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
