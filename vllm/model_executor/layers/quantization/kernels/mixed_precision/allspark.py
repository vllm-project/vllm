# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.allspark_utils import (
    ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD, check_allspark_supported_dtype_shape)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           permute_param_layout_)

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class AllSparkLinearKernel(MPLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> tuple[bool, Optional[str]]:
        if c.has_g_idx:
            return False, "Act reordering currently not supported by AllSpark"

        if c.zero_points:
            return False, "Zero points currently not supported by AllSpark"

        return check_allspark_supported_dtype_shape(
            c.partition_weight_shape[0],  # in_features
            c.partition_weight_shape[1],  # out_features
            c.group_size,
            c.weight_type,
            c.act_type)

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config

        # prepare the parameters required for the kernel
        properties = torch.cuda.get_device_properties(device.index)
        sm_count = properties.multi_processor_count
        sm_version = properties.major * 10 + properties.minor
        gemm_args = {}
        gemm_args['sm_count'] = sm_count
        gemm_args['sm_version'] = sm_version

        self.gemm_args = gemm_args

        # transform param weight, scale
        old_weight_param = getattr(layer, self.w_q_name)
        old_scale_param = getattr(layer, self.w_s_name)

        assert isinstance(old_weight_param, BasevLLMParameter)
        permute_param_layout_(old_weight_param,
                              input_dim=0,
                              output_dim=1,
                              packed_dim=0)

        assert isinstance(old_scale_param, BasevLLMParameter)
        permute_param_layout_(old_scale_param, input_dim=0, output_dim=1)

        # unpack weight from K / 4 x N int32 to K x N uint8
        new_weight_param = torch.nn.Parameter(old_weight_param.data,
                                              requires_grad=False)
        new_weight_param.data = new_weight_param.data.t().contiguous().view(
            dtype=torch.uint8)
        new_weight_param.data = new_weight_param.data.t().contiguous()

        new_scale_param = torch.nn.Parameter(old_scale_param.data,
                                             requires_grad=False)

        # reorder K x N weight as N32K16 format for Ampere W8A16
        new_weight_param.data, new_scale_param.data, _ = \
            ops.allspark_repack_weight(
                new_weight_param.data, new_scale_param.data, None,
                c.zero_points)

        replace_parameter(layer, self.w_q_name, new_weight_param.data)
        replace_parameter(layer, self.w_s_name, new_scale_param.data)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.config
        gemm_args = self.gemm_args
        w_q, w_s, _, _ = self._get_weight_params(layer)

        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1], )

        output = ops.allspark_w8a16_gemm(
            a=reshaped_x,
            b_qweight=w_q,
            b_scales=w_s,
            b_qzeros=None,
            n=c.partition_weight_shape[1],
            group_size=c.group_size,
            sm_count=gemm_args['sm_count'],
            sm_version=gemm_args['sm_version'],
            CUBLAS_M_THRESHOLD=ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
            has_zp=c.zero_points,
            n32k16_reorder=True)

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
