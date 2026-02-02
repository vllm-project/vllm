# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn import Parameter

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.kernels.block_scaled_mm import (
    init_fp8_block_scaled_linear_kernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    process_fp8_weight_channel_strategy,
    process_fp8_weight_tensor_strategy,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    PerTensorScaleParameter,
)

__all__ = ["CompressedTensorsW8A8Fp8"]

strategy_to_parameter_type = {
    QuantizationStrategy.BLOCK: BlockQuantScaleParameter,
    QuantizationStrategy.CHANNEL: ChannelQuantScaleParameter,
    QuantizationStrategy.TENSOR: PerTensorScaleParameter,
}

STATIC_QUANT = True
DYNAMIC_QUANT = False
activation_quant_key_mapping = {
    STATIC_QUANT: kFp8StaticTensorSym,
    DYNAMIC_QUANT: kFp8DynamicTokenSym,
}
weight_quant_key_mapping = {
    QuantizationStrategy.CHANNEL: kFp8StaticTokenSym,
    QuantizationStrategy.TENSOR: kFp8StaticTensorSym,
}
logger = init_logger(__name__)


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme):
    def __init__(self, weight_quant: QuantizationArgs, is_static_input_scheme: bool):
        self.weight_quant = weight_quant
        self.strategy = weight_quant.strategy
        self.out_dtype = torch.get_default_dtype()
        self.is_static_input_scheme = is_static_input_scheme
        self.weight_block_size = self.weight_quant.block_structure

        if self.weight_block_size is not None:
            self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
            self.use_aiter_and_is_supported = rocm_aiter_ops.is_linear_fp8_enabled()
            assert not self.is_static_input_scheme
            self.act_q_group_shape = GroupShape(1, self.weight_block_size[0])

            weight_quant_key = create_fp8_quant_key(
                static=True, group_shape=GroupShape(*self.weight_block_size)
            )
            activation_quant_key = create_fp8_quant_key(
                static=False, group_shape=self.act_q_group_shape
            )
            self.w8a8_block_fp8_linear = init_fp8_block_scaled_linear_kernel(
                weight_quant_key=weight_quant_key,
                activation_quant_key=activation_quant_key,
                out_dtype=self.out_dtype,
                module_name=self.__class__.__name__,
            )
        else:
            activation_quant_key = activation_quant_key_mapping[is_static_input_scheme]
            weight_quant_key = weight_quant_key_mapping[self.strategy]
            self.fp8_linear = init_fp8_linear_kernel(
                activation_quant_key=activation_quant_key,
                weight_quant_key=weight_quant_key,
                out_dtype=self.out_dtype,
                module_name=self.__class__.__name__,
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = None
        layer.orig_dtype = params_dtype

        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            # Validate block quantization shapes
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        # WEIGHT
        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = create_fp8_scale_parameter(
            strategy_to_parameter_type[self.strategy],
            output_partition_sizes,
            input_size_per_partition,
            layer.weight_block_size,
            weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
            layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer) -> None:
        if self.strategy == QuantizationStrategy.TENSOR:
            weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
                layer.weight,
                layer.weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            weight = weight.t()
        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight, weight_scale, input_scale = process_fp8_weight_channel_strategy(
                layer.weight, layer.weight_scale, getattr(layer, "input_scale", None)
            )
            weight = weight.t()

        elif self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            self.w8a8_block_fp8_linear.process_weights_after_loading(layer)
            input_scale = None

        else:
            raise ValueError(
                f"Unknown quantization strategy {self.strategy}: "
                f"should be one of {list(QuantizationStrategy)}"
            )

        # required by torch.compile to be torch.nn.Parameter
        layer.weight = Parameter(weight.data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        if input_scale is not None:
            layer.input_scale = Parameter(input_scale.data, requires_grad=False)

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, "input_scale"):
            layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
        else:
            layer.input_scale = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.weight_block_size is not None:
            return self.w8a8_block_fp8_linear.apply_weights(
                layer,
                x,
                bias,
            )

        return self.fp8_linear.apply_weights(layer, x, bias)
