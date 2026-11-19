# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.humming import (
    apply_humming_linear,
    convert_linear_layer_to_humming_standard,
    get_humming_linear_compute_config,
    prepare_humming_linear_layer_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.utils.humming import create_humming_locks
from vllm.utils.import_utils import has_humming

from .base import (
    Int8LinearKernel,
    Int8LinearLayerConfig,
)


class HummingInt8LinearKernel(Int8LinearKernel):
    """Humming GEMM Kernel for INT8."""

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
    def executed_activation_key(
        cls, activation_quant_key: QuantKey | None
    ) -> QuantKey | None:
        return None

    @classmethod
    def can_implement(cls, config: Int8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_name, weight_scale_name, *_ = self.layer_param_names
        name_map = {"weight": weight_name, "weight_scale": weight_scale_name}
        quant_config = {"quant_method": "humming", "dtype": "int8"}
        weight = getattr(layer, weight_name)
        weight.data = weight.data + 128

        convert_linear_layer_to_humming_standard(layer=layer, name_map=name_map)
        self.layer_config = prepare_humming_linear_layer_config(layer, quant_config)
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
