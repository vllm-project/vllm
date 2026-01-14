# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn import Parameter

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    maybe_post_process_fp8_weight_block,
    process_fp8_weight_block_strategy,
    process_fp8_weight_channel_strategy,
    process_fp8_weight_tensor_strategy,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp,
    cutlass_block_fp8_supported,
    maybe_create_device_identity,
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


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme):
    def __init__(self, weight_quant: QuantizationArgs, is_static_input_scheme: bool):
        self.weight_quant = weight_quant
        self.strategy = weight_quant.strategy
        self.out_dtype = torch.get_default_dtype()
        self.is_static_input_scheme = is_static_input_scheme

        self.weight_block_size = self.weight_quant.block_structure
        if self.weight_block_size is not None:
            self.act_q_group_shape = GroupShape(1, self.weight_block_size[0])
        else:
            self.act_q_group_shape = (
                GroupShape.PER_TENSOR
                if is_static_input_scheme
                else GroupShape.PER_TOKEN
            )

        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.use_aiter_and_is_supported = rocm_aiter_ops.is_linear_fp8_enabled()

        if self.weight_block_size is not None:
            assert not self.is_static_input_scheme
            self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(
                weight_group_shape=GroupShape(*self.weight_block_size),
                act_quant_group_shape=self.act_q_group_shape,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
                use_aiter_and_is_supported=self.use_aiter_and_is_supported,
            )
        else:
            self.fp8_linear = Fp8LinearOp(
                act_quant_static=self.is_static_input_scheme,
                act_quant_group_shape=self.act_q_group_shape,
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
        maybe_create_device_identity()

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
            weight, weight_scale = process_fp8_weight_block_strategy(
                layer.weight, layer.weight_scale
            )
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

        if self.strategy == QuantizationStrategy.BLOCK:
            maybe_post_process_fp8_weight_block(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.weight_block_size is not None:
            return self.w8a8_block_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
            )

        return self.fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=layer.input_scale,
            bias=bias,
        )
