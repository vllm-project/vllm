# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    convert_linear_layer_to_humming_standard,
    prepare_humming_layer,
)
from vllm.platforms import current_platform
from vllm.utils.humming import dtypes

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)


class HummingFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """Humming GEMM Kernel for FP8."""

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
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        name_map = {"weight": "weight", "weight_scale": "weight_scale"}
        scale_torch_dtype = self.config.weight_quant_key.scale.dtype
        scale_dtype = dtypes.DataType.from_torch_dtype(scale_torch_dtype)

        quant_config = {
            "quant_method": "humming",
            "dtype": "float8e4m3",
            "scale_dtype": scale_dtype,
        }

        assert self.config.weight_quant_key.scale2 is None
        scale_group_shape = self.config.weight_quant_key.scale.group_shape
        if scale_group_shape.is_per_tensor():
            quant_config["weight_scale_type"] = "tensor"
        elif scale_group_shape.is_per_channel():
            quant_config["weight_scale_type"] = "channel"
        elif scale_group_shape.is_per_group():
            quant_config["weight_scale_type"] = "group"
            quant_config["group_size"] = scale_group_shape.col
        else:
            assert scale_group_shape.row > 0 and scale_group_shape.col > 0
            quant_config["weight_scale_type"] = "block"
            quant_config["weight_scale_group_size_n"] = scale_group_shape.row
            quant_config["weight_scale_group_size"] = scale_group_shape.col

            if hasattr(layer, "weight_scale_inv"):
                name_map["weight_scale"] = "weight_scale_inv"

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

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass


class HummingInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    """Humming GEMM Kernel for INT8."""

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
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_name, weight_scale_name, *_ = self.layer_param_names
        name_map = {"weight": weight_name, "weight_scale": weight_scale_name}
        quant_config = {"quant_method": "humming", "dtype": "int8"}
        weight = getattr(layer, weight_name)
        weight.data = weight.data + 128

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
