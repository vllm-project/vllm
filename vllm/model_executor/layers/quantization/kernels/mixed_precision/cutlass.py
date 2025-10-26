# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class CutlassW4A8LinearKernel(MPLinearKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic per-tok fp8 activation quantization
        self.quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "CUTLASS only supported on CUDA"

        if not current_platform.is_device_capability(90):
            return False, "CUTLASS W4A8 requires compute capability of 90 (Hopper)"

        if c.act_type != torch.float8_e4m3fn:
            return False, "CUTLASS W4A8 only supports FP8 (e4m3) activations"

        if c.has_g_idx:
            return False, "Act reordering not supported by CUTLASS W4A8"

        if c.zero_points:
            return False, "Zero points not supported by CUTLASS W4A8"

        if c.weight_type != scalar_types.int4:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "CUTLASS W4A8, only supported int4",
            )

        # TODO(czhu): support -1 (column-wise)
        if c.group_size != 128:
            return False, "Only group_size 128 is supported"

        in_features, out_features = c.partition_weight_shape
        if in_features % 128 or out_features % 128:
            return (
                False,
                f"K and N must be divisible by 128, got {c.partition_weight_shape}",
            )

        if c.out_type != torch.bfloat16:
            return (
                False,
                f"Only bfloat16 output type currently supportedgot {c.out_type=}",
            )

        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale`  is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module):
        # TODO(czhu): optimize speed/mem usage
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
        self._transform_param(layer, "weight_chan_scale", lambda x: x)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, _, _ = self._get_weight_params(layer)
        w_ch_s = layer.weight_chan_scale

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        x_2d, act_scales = self.quant_fp8(x_2d)
        output = ops.cutlass_w4a8_mm(
            a=x_2d,
            b_q=w_q,
            b_group_scales=w_s,
            b_group_size=c.group_size,
            a_token_scales=act_scales,
            b_channel_scales=w_ch_s,
        )

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
