# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.machete_utils import (
    MACHETE_SUPPORTED_GROUP_SIZES, check_machete_supports_shape,
    query_machete_supported_quant_types)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           permute_param_layout_)

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class MacheteLinearKernel(MPLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> Tuple[bool, Optional[str]]:

        if c.has_g_idx and\
            c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return False, "Act reordering currently not supported by Machete, "\
                          "when the input features are partitioned across "\
                          "devices"
        if c.zero_points:
            return False, "Zero points currently not supported by Machete"

        if c.weight_type not in query_machete_supported_quant_types(
                c.zero_points):
            return False, f"Quant type ({c.weight_type}) not supported by "\
                           "Machete, supported types are: "\
                           f"{query_machete_supported_quant_types(c.zero_points)}"

        if c.group_size not in MACHETE_SUPPORTED_GROUP_SIZES:
            return False, f"Group size ({c.group_size}) not supported by "\
                            "Machete, supported group sizes are: "\
                            f"{MACHETE_SUPPORTED_GROUP_SIZES}"

        return check_machete_supports_shape(c.partition_weight_shape[0],
                                            c.partition_weight_shape[1])

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale`  is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config

        if c.has_g_idx:
            assert self.w_gidx_name is not None
            perm = torch.argsort(getattr(layer, self.w_gidx_name))\
                .to(torch.int)

            self.act_perm = lambda x: x[:, perm]
            # use `ops.permute_cols` if possible
            if c.act_type in [torch.float16, torch.bfloat16] \
                and c.partition_weight_shape[0] % 8 == 0:
                self.act_perm = partial(ops.permute_cols, perm=perm)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            if c.has_g_idx:
                x_unpacked = unpack_quantized_values_into_int32(x.data,
                                                                c.weight_type,
                                                                packed_dim=0)
                x_perm = x_unpacked[perm, :]
                x.data = pack_quantized_values_into_int32(x_perm,
                                                          c.weight_type,
                                                          packed_dim=0)
            x.data = ops.machete_prepack_B(x.data.t().contiguous().t(),
                                           a_type=c.act_type,
                                           b_type=c.weight_type,
                                           group_scales_type=c.act_type)
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x

        # Repack weights and scales for Machete
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.config
        w_q, w_s, _, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1], )

        if c.has_g_idx:
            x_2d = self.act_perm(x_2d)

        output = ops.machete_mm(a=x_2d,
                                b_q=w_q,
                                b_type=c.weight_type,
                                b_group_zeros=None,
                                b_group_scales=w_s,
                                b_group_size=c.group_size)

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
