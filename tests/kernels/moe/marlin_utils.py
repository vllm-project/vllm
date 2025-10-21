# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from tests.kernels.utils import stack_and_dev
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_permute_bias,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_mxfp4_like,
    rand_marlin_weight_nvfp4_like,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    awq_marlin_quantize,
    marlin_quantize,
)
from vllm.scalar_type import ScalarType, scalar_types


@dataclass
class MarlinMoEWeightData:
    w_ref: torch.Tensor
    qweight: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor | None
    g_idx: torch.Tensor | None
    zeros: torch.Tensor | None
    sort_indices: torch.Tensor | None
    marlin_bias: torch.Tensor | None

    @staticmethod
    def make(
        w: torch.Tensor,
        quant_type: ScalarType,
        group_size: int,
        act_order: bool | None = None,
        bias: torch.Tensor | None = None,
    ) -> "MarlinMoEWeightData":
        assert w.ndim == 3
        has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]
        k = w.shape[-1]

        w_ref_l: list[torch.Tensor] = []
        qweight_l: list[torch.Tensor] = []
        scales_l: list[torch.Tensor] = []
        global_scale_l: list[torch.Tensor] = []
        zeros_l: list[torch.Tensor] = []
        g_idx_l: list[torch.Tensor] = []
        sort_indices_l: list[torch.Tensor] = []
        bias_l: list[torch.Tensor] = []

        for i in range(w.shape[0]):
            if quant_type == scalar_types.float4_e2m1f:
                if group_size == 16:
                    w_ref, qweight, scales, global_scale = (
                        rand_marlin_weight_nvfp4_like(w[i], group_size)
                    )
                else:
                    w_ref, qweight, scales = rand_marlin_weight_mxfp4_like(
                        w[i], group_size
                    )
                    global_scale = None

                w_ref_l.append(w_ref.T)
                qweight_l.append(qweight)
                scales_l.append(scales)
                if global_scale is not None:
                    global_scale_l.append(global_scale)
            elif quant_type == scalar_types.float8_e4m3fn:
                w_ref, qweight, scales = marlin_quant_fp8_torch(w[i], group_size)
                w_ref_l.append(w_ref.T)
                qweight_l.append(qweight)
                scales_l.append(scales)
            elif has_zp:
                w_ref, qweight, scales, zeros = awq_marlin_quantize(
                    w[i].transpose(1, 0), quant_type, group_size
                )

                w_ref_l.append(w_ref.T)
                qweight_l.append(qweight)
                scales_l.append(scales)
                zeros_l.append(zeros)
            else:
                test_perm = torch.randperm(k)
                assert act_order is not None
                w_ref, qweight, scales, g_idx, sort_indices, _ = marlin_quantize(
                    w[i].transpose(1, 0), quant_type, group_size, act_order, test_perm
                )

                w_ref_l.append(w_ref.T)
                qweight_l.append(qweight)
                scales_l.append(scales)
                g_idx_l.append(g_idx)
                sort_indices_l.append(sort_indices)

            if bias is not None:
                bias_l.append(marlin_permute_bias(bias[i]))

        w_ref = stack_and_dev(w_ref_l)
        qweight = stack_and_dev(qweight_l).contiguous()
        scales = stack_and_dev(scales_l)
        global_scale = stack_and_dev(global_scale_l) if global_scale_l else None
        g_idx = stack_and_dev(g_idx_l) if g_idx_l else None
        zeros = stack_and_dev(zeros_l) if zeros_l else None
        sort_indices = stack_and_dev(sort_indices_l) if sort_indices_l else None
        marlin_bias = stack_and_dev(bias_l) if bias_l else None

        return MarlinMoEWeightData(
            w_ref=w_ref,
            qweight=qweight,
            scales=scales,
            global_scale=global_scale,
            g_idx=g_idx,
            zeros=zeros,
            sort_indices=sort_indices,
            marlin_bias=marlin_bias,
        )


def make_marlin_moe_weights(
    e: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    quant_type: ScalarType | str,
    group_size: int,
    act_order: bool | None = None,
) -> tuple[MarlinMoEWeightData, MarlinMoEWeightData]:
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    if isinstance(quant_type, str):
        assert quant_type == "mxfp4"
        quant_type = scalar_types.float4_e2m1f

    w1_data = MarlinMoEWeightData.make(
        w=w1,
        quant_type=quant_type,
        group_size=group_size,
        act_order=act_order,
        bias=None,
    )
    w2_data = MarlinMoEWeightData.make(
        w=w2,
        quant_type=quant_type,
        group_size=group_size,
        act_order=act_order,
        bias=None,
    )
    return (w1_data, w2_data)
