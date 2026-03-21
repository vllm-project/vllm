# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    apply_awq_marlin_linear,
    awq_to_marlin_zero_points,
    check_marlin_supports_shape,
    marlin_act_int8_process_scales,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    query_marlin_supported_quant_types,
)
from vllm.platforms import current_platform

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class AWQMarlinLinearKernel(MPLinearKernel):
    """Marlin kernel for AWQ-format quantized weights.

    AWQ uses a different on-disk weight packing format than GPTQ:
      - qweight shape: [K, N // pack_factor]  (packed_dim=1, column-packed)
      - vs GPTQ's:     [K // pack_factor, N]  (packed_dim=0, row-packed)

    This kernel handles the AWQ-specific repacking to the marlin layout and
    delegates the actual GEMM to apply_awq_marlin_linear.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "AWQMarlin only supported on CUDA"

        # AWQ always uses zero_points; skip otherwise so the plain
        # MarlinLinearKernel can handle symmetric cases.
        if not c.zero_points:
            return False, "AWQMarlinLinearKernel requires zero_points=True"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by AWQMarlin, "
                f"supported types are: {quant_types}",
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by AWQMarlin, "
                f"supported group sizes are: {MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        # AWQ does not support activation ordering.
        if c.has_g_idx:
            return False, "AWQMarlinLinearKernel does not support g_idx"

        return check_marlin_supports_shape(
            c.partition_weight_shape[1],  # out_features (N)
            c.partition_weight_shape[0],  # in_features  (K)
            c.full_weight_shape[0],  # full in_features
            c.group_size,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        device = getattr(layer, self.w_q_name).device
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        size_k = c.partition_weight_shape[0]
        size_n = c.partition_weight_shape[1]
        num_bits = c.weight_type.size_bits
        num_groups = size_k // c.group_size if c.group_size != -1 else 1

        # fp8 activation preprocessing must happen before repacking.
        if c.act_type == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(
                getattr(layer, self.w_q_name),
                getattr(layer, self.w_zp_name),
                inplace=True,
            )
            getattr(layer, self.w_s_name).data = (
                getattr(layer, self.w_s_name).data * 512
            )

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace_new(device)

        # Repack qweight from AWQ format [K, N//pack] to marlin format.
        marlin_qweight = ops.awq_marlin_repack(
            getattr(layer, self.w_q_name),
            size_k=size_k,
            size_n=size_n,
            num_bits=num_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, self.w_q_name, marlin_qweight)

        # Permute scales from AWQ format to marlin format.
        marlin_scales = marlin_permute_scales(
            getattr(layer, self.w_s_name),
            size_k=size_k,
            size_n=size_n,
            group_size=c.group_size,
            is_a_8bit=is_a_8bit,
        )
        if c.act_type == torch.int8 and num_groups > 1:
            marlin_scales, input_global_scale = marlin_act_int8_process_scales(
                marlin_scales
            )
            layer.register_parameter(
                "input_global_scale",
                torch.nn.Parameter(input_global_scale, requires_grad=False),
            )
        else:
            layer.input_global_scale = None
        replace_parameter(layer, self.w_s_name, marlin_scales)

        # Convert zero-points from AWQ format to marlin format.
        marlin_zp = awq_to_marlin_zero_points(
            getattr(layer, self.w_zp_name),
            size_k=num_groups,
            size_n=size_n,
            num_bits=num_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, self.w_zp_name, marlin_zp)

        # AWQ does not use activation ordering; provide empty tensors.
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data = marlin_permute_bias(layer.bias)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        return apply_awq_marlin_linear(
            input=x,
            weight=getattr(layer, self.w_q_name),
            weight_scale=getattr(layer, self.w_s_name),
            weight_zp=getattr(layer, self.w_zp_name),
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            quant_type=c.weight_type,
            output_size_per_partition=c.partition_weight_shape[1],
            input_size_per_partition=c.partition_weight_shape[0],
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )
