# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from typing import Optional

import torch
import os
from safetensors.torch import save_file

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.machete_utils import (
    check_machete_supports_shape, query_machete_supported_group_sizes,
    query_machete_supported_quant_types)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           permute_param_layout_)
from vllm.platforms import current_platform

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# czhu: on the fly dynamic fp8 quant
def fp8_per_token(
    x_bf16: torch.Tensor,
    safety: float = 0.95,         # shrink range a bit to avoid infs
    eps: float = 1e-12,
    fake: bool = True
):
    """
    Per-token 'fake' FP8 quantization of activations (row = token).
    x_bf16: [M, K] BF16 tensor
    Returns: (x_bf16_fake, scales_bf16) where x_bf16_fake is BF16 after FP8 Q/DQ
    """

    assert x_bf16.dim() == 2, "expected [M, K]"
    assert x_bf16.dtype == torch.bfloat16, "input must be BF16"
    fp8_dtype = torch.float8_e4m3fn

    # Compute per-token amax in FP32 for stability
    x_f32 = x_bf16.to(torch.float32)
    amax = torch.amax(x_f32.abs(), dim=1)  # [M]

    # Scale so that per-row max maps near FP8 max (with a safety margin)
    alpha = torch.finfo(fp8_dtype).max  # FP8 max finite value
    # avoid divide-by-zero; if a row is all zeros, set scale=1
    scales = torch.where(amax > 0, amax / (alpha * safety), torch.ones_like(amax))

    # Apply scaling, Q->DQ through FP8
    inv_scales = (1.0 / (scales + eps)).to(torch.float32)      # [M]
    x_scaled = x_f32 * inv_scales.unsqueeze(1)                  # [M, K]
    x_fp8 = x_scaled.to(fp8_dtype)                              # quantize
                        # dequantize to BF16
    if fake:
        x_dq = x_fp8.to(torch.bfloat16)         
        # x_out = (x_dq * scales.to(torch.bfloat16).unsqueeze(1))     # re-apply scale
        return x_dq, scales.to(torch.bfloat16)
    else:
        return x_fp8, scales.to(torch.bfloat16)

class MacheteLinearKernel(MPLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> tuple[bool, Optional[str]]:
        # Machete uses CUTLASS, so it can only be compatible with Nvidia
        if not current_platform.is_cuda():
            return False, "Machete only supported on CUDA"

        if not current_platform.is_device_capability(90):
            return False, "Machete requires compute capability of 90 (Hopper)"

        if c.has_g_idx and\
            c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return False, "Act reordering currently not supported by Machete, "\
                          "when the input features are partitioned across "\
                          "devices"

        if c.weight_type not in query_machete_supported_quant_types(
                c.zero_points):
            return False, f"Quant type ({c.weight_type}) not supported by "\
                           "Machete, supported types are: "\
                           f"{query_machete_supported_quant_types(c.zero_points)}"

        if c.group_size not in query_machete_supported_group_sizes(c.act_type):
            return False, f"Group size ({c.group_size}) not supported by "\
                            "Machete, supported group sizes are: "\
                            f"{query_machete_supported_group_sizes(c.act_type)}"

        return check_machete_supports_shape(c.partition_weight_shape[0],
                                            c.partition_weight_shape[1])

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale`  is: {input_dim = 0, output_dim = 1}
    #  `weight_zp`     is: {input_dim = 0, output_dim = 1, packed_dim = 1}
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

        # hack a transform for w4a8 weight
        def transform_w_q_w4a8(x):
            assert isinstance(x, BasevLLMParameter)
            assert not c.has_g_idx, "w4a8 does not support act reordering"
            # save orig
            self.w_orig = x.detach().clone().cpu()
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = ops.cutlass_encode_and_reorder_int4b(x.data.t().contiguous().t())
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x

        # hack a transform for w4a8 fp8 scale
        def transform_w_s_w4a8(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            # this needs to be fp8
            x.data = x.data.contiguous().to(torch.float8_e4m3fn)
            x.data = ops.cutlass_pack_scale_fp8(x.data)
            return x

        def transform_w_zp(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=1)
            x_unpacked = unpack_quantized_values_into_int32(x.data,
                                                            c.weight_type,
                                                            packed_dim=1)
            w_s = getattr(layer, self.w_s_name).data
            # pre-apply scales to zero-points
            x.data = (-1.0 * w_s * (x_unpacked.to(w_s.dtype))).contiguous()
            return x

        # Repack weights and scales for Machete
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)
        # hack w4a8
        # self._transform_param(layer, self.w_q_name, transform_w_q_w4a8)
        # self._transform_param(layer, self.w_s_name, transform_w_s_w4a8)
        if c.zero_points:
            self._transform_param(layer, self.w_zp_name, transform_w_zp)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert bias is None, "bias not supported by Machete"
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

        # czhu: dynamic fp8 quant
        # x_2d, act_scales = fp8_per_token(x_2d, fake=False)
        # # # # call cutlass w4a8
        # output = ops.cutlass_w4a8_mm(a=x_2d,
        #                              b_q=w_q,
        #                              b_type=c.weight_type, # not actually used?
        #                              b_group_scales=w_s)
        output = ops.machete_mm(
                                a=x_2d,
                                b_q=w_q,
                                b_type=c.weight_type,
                                b_group_zeros=w_zp,
                                b_group_scales=w_s,
                                b_group_size=c.group_size)

        # simulate per-row fp8 quant, apply in fp32
        # output = ((output.to(torch.float32)) * (act_scales.to(torch.float32).unsqueeze(1))).to(torch.bfloat16)

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
