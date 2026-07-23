# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
import math

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearKernel,
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.auto_awq import (
    _convert_awq_to_standard_format,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_scheme import (
    QuarkScheme,
)
from vllm.model_executor.layers.quantization.quark.utils import (
    canonicalize_quark_packed_int4,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


class QuarkW4A16Int4(QuarkScheme):
    """Quark packed INT4 weight-only linear scheme via MPLinearKernel."""

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, group_size: int, pack_method: str, is_symmetric: bool):
        self.group_size = group_size
        self.pack_factor = 8
        self.pack_reorder = pack_method == "reorder"
        self.is_symmetric = is_symmetric
        self.quant_type = (
            scalar_types.uint4b8 if is_symmetric else scalar_types.uint4
        )
        self.kernel: MPLinearKernel | None = None

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        input_size = kwargs["input_size"]
        output_size = kwargs["output_size"]
        group_size = (
            self.group_size if self.group_size != -1 else input_size_per_partition
        )
        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        packed_output_size_per_partition = math.ceil(
            output_size_per_partition / self.pack_factor
        )
        layer.output_size_per_partition = output_size_per_partition
        layer.packed_output_size_per_partition = packed_output_size_per_partition

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=group_size,
            zero_points=not self.is_symmetric,
            has_g_idx=False,
        )
        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)
        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for QuarkW4A16Int4", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        def weight_scale_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            *args,
            **loader_kwargs,
        ) -> None:
            if loaded_weight.shape[1] < param.data.shape[1]:
                padded_weight = loaded_weight.new_zeros(param.data.shape)
                padded_weight[:, : loaded_weight.shape[1]] = loaded_weight
                loaded_weight = padded_weight
            weight_loader(param, loaded_weight, *args, **loader_kwargs)

        weight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                packed_output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        num_groups = input_size_per_partition // group_size
        weight_zero_point = PackedvLLMParameter(
            data=torch.zeros(
                num_groups,
                packed_output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                packed_output_size_per_partition * self.pack_factor,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_scale_loader,
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_zero_point", weight_zero_point)
        layer.register_parameter("weight_scale", weight_scale)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="weight",
            w_s_param_name="weight_scale",
            w_zp_param_name="weight_zero_point",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = canonicalize_quark_packed_int4(
            layer.weight.data,
            pack_reorder=self.pack_reorder,
            is_symmetric=self.is_symmetric,
            pack_factor=self.pack_factor,
        )
        layer.weight_zero_point.data = canonicalize_quark_packed_int4(
            layer.weight_zero_point.data,
            pack_reorder=self.pack_reorder,
            is_symmetric=self.is_symmetric,
            pack_factor=self.pack_factor,
        )
        output_size = layer.output_size_per_partition
        packed_output_size = layer.packed_output_size_per_partition * self.pack_factor
        if output_size < packed_output_size:
            layer.weight_scale.data[:, output_size:].zero_()

        _convert_awq_to_standard_format(
            layer, "weight", "weight_zero_point", self.quant_type.size_bits
        )
        assert self.kernel is not None
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        assert self.kernel is not None
        return self.kernel.apply_weights(layer, x, bias)
