# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           permute_param_layout_)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class CutlassW4A8LinearKernel(MPLinearKernel):
    # hack the fp8 quant op here
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> tuple[bool, Optional[str]]:
        if not current_platform.is_cuda():
            return False, "CUTLASS only supported on CUDA"

        if not current_platform.is_device_capability(90):
            return False, "CUTLASS W4A8 requires compute capability of 90 (Hopper)"

        # TODO: figure out how to register the fp8 activation part
        # if c.act_type != torch.float8_e4m3fn:
        #     return False, "CUTLASS W4A8 only supports FP8 (e4m3) activations"

        if c.has_g_idx:
            return False, "Act reordering currently not supported by CUTLASS W4A8"

        if c.zero_points:
            return False, "Zero points not currently supported by CUTLASS W4A8"
        
        # TODO: enforce signed int4? The testing is with the existing w4a16 config
        # and int4b8 weights converted to int4, but the config is the same so we
        # expect int4b8 here.
        # if c.weight_type not in query_machete_supported_quant_types(
        #         c.zero_points):
        #     return False, f"Quant type ({c.weight_type}) not supported by "\
        #                    "Machete, supported types are: "\
        #                    f"{query_machete_supported_quant_types(c.zero_points)}"

        # TODO: column-wise should work
        if c.group_size != 128:
            return False, "Only group_size 128 is supported"

        # TODO: verify shapes (c.partition_weight_shape[0], c.partition_weight_shape[1]))
        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale`  is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config

        # TODO: seems a bit slow/mem intensive
        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = ops.cutlass_encode_and_reorder_int4b(x.data.t().contiguous().t())
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous().to(torch.float8_e4m3fn)
            x.data = ops.cutlass_pack_scale_fp8(x.data)
            return x

        # Encode/reorder weights and pack scales
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        # dummy channel scales
        self.w_ch_s = torch.ones((c.partition_weight_shape[1],), dtype=torch.float32, device='cuda')
        
    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert bias is None, "bias not supported by CUTLASS W4A8"
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1], )

        if c.has_g_idx:
            x_2d = self.act_perm(x_2d)

        if c.zero_points:
            assert w_zp is not None
        else:
            w_zp = None

        # per-tok quant
        x_2d, act_scales = self.quant_fp8(
            x_2d,
        )
        output = ops.cutlass_w4a8_mm(a=x_2d,
                                     b_q=w_q,
                                     b_group_scales=w_s,
                                     b_group_size=c.group_size,
                                     a_token_scales=act_scales,
                                     b_channel_scales=self.w_ch_s)

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
