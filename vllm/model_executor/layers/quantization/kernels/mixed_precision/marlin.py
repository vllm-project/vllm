# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES, apply_gptq_marlin_linear,
    check_marlin_supports_shape, marlin_is_k_full, marlin_make_empty_g_idx,
    marlin_make_workspace_new, marlin_permute_scales, marlin_sort_g_idx,
    marlin_zero_points, query_marlin_supported_quant_types, unpack_cols)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           permute_param_layout_)
from vllm.platforms import current_platform

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class MarlinLinearKernel(MPLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> tuple[bool, Optional[str]]:
        # Marlin uses inline PTX, so it can only be compatible with Nvidia
        if not current_platform.is_cuda():
            return False, "Marlin only supported on CUDA"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return False, f"Quant type ({c.weight_type}) not supported by"\
                          f"  Marlin, supported types are: {quant_types}"

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return False, f"Group size ({c.group_size}) not supported by "\
                            "Marlin, supported group sizes are: "\
                            f"{MARLIN_SUPPORTED_GROUP_SIZES}"

        return check_marlin_supports_shape(
            c.partition_weight_shape[1],  # out_features
            c.partition_weight_shape[0],  # in_features
            c.full_weight_shape[0],  # in_features
            c.group_size)

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config

        row_parallel = (c.partition_weight_shape[0] != c.full_weight_shape[0])
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace_new(device)

        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        if self.w_gidx_name is None:
            self.w_gidx_name = "g_idx"
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = ops.gptq_marlin_repack(x.data.contiguous(),
                                            perm=layer.g_idx_sort_indices,
                                            size_k=c.partition_weight_shape[0],
                                            size_n=c.partition_weight_shape[1],
                                            num_bits=c.weight_type.size_bits)
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(x.data.contiguous(),
                                           size_k=c.partition_weight_shape[0],
                                           size_n=c.partition_weight_shape[1],
                                           group_size=c.group_size)
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name))
            self._transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = (c.partition_weight_shape[0] //
                         c.group_size if c.group_size != -1 else 1)
            self._transform_param(layer, self.w_zp_name, lambda x: \
                marlin_zero_points(
                    unpack_cols(x.t(), c.weight_type.size_bits,
                                grouped_k,
                                c.partition_weight_shape[1]),
                    size_k=grouped_k,
                    size_n=c.partition_weight_shape[1],
                    num_bits=c.weight_type.size_bits))
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for marlin
        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            is_k_full=self.is_k_full,
            bias=bias)
