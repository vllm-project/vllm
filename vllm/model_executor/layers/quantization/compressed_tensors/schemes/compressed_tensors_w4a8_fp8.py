# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import ActivationOrdering, QuantizationArgs

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repeat_scales_on_all_ranks,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A8Fp8"]
W4A8_SUPPORTED_TYPES_MAP = {
    4: scalar_types.int4,
}
W4A8_SUPPORTED_BITS = list(W4A8_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsW4A8Fp8(CompressedTensorsScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        symmetric: bool | None = True,
        actorder: ActivationOrdering | None = None,
    ):
        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP

        if self.group_size != 128 or self.strategy != "group":
            raise ValueError(
                "W4A8 kernels require group quantization with group size 128"
            )

        if num_bits not in W4A8_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {W4A8_SUPPORTED_TYPES_MAP.keys()}"
            )

        self.quant_type = W4A8_SUPPORTED_TYPES_MAP[num_bits]

    @classmethod
    def get_min_capability(cls, weight_quant: QuantizationArgs) -> int:
        # hopper
        return 90

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=torch.float8_e4m3fn,  # always use fp8(e4m3)
            group_size=self.group_size,
            zero_points=not self.symmetric,
            has_g_idx=self.has_g_idx,
            out_type=params_dtype,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for CompressedTensorsW4A8Fp8", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            self.has_g_idx, self.group_size, row_parallel
        )

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )

        # After loading, we will transform bf16 -> fp8 ->
        # expand by 8x via `cutlass_pack_scale_fp8`
        # and construct per-channel fp32 scales.
        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name="weight_zero_point",
            w_gidx_param_name="weight_g_idx",
        )

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
