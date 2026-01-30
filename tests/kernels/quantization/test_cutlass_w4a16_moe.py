# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test CUTLASS W4A16 MoE."""

import pytest
import torch
from test_cutlass_w4a8_moe import cutlass_quantize

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import (
    cutlass_moe_w4a16_bf16,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    grouped_topk,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.v1.worker.workspace import init_workspace_manager

torch.random.manual_seed(42)
device = torch.device("cuda:0")
init_workspace_manager(device=device)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUTLASS is only supported on CUDA.",
)
@pytest.mark.parametrize("bs", [1, 128, 1024])
@pytest.mark.parametrize(
    "expert_num,intermediate_size,hidden_size",
    [
        (48, 2048, 7168),  # Kimi-K2 EP8
        (384, 256, 7168),  # Kimi-K2 TP8
        (96, 1024, 7168),  # Kimi-K2 TP4 EP4
    ],
)
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("group_size", [64])
def test_cutlass_w4a16_moe(
    bs, intermediate_size, hidden_size, expert_num, topk, group_size
):
    x = torch.randn((bs, hidden_size), device=device, dtype=torch.bfloat16)
    w13_bf16 = torch.randn(
        (expert_num, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    w2_bf16 = torch.randn(
        (expert_num, hidden_size, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )

    # quantize and dequantize weights
    w13_ref, w13_cutlass, w13_scales_cutlass = [], [], []
    w2_ref, w2_cutlass, w2_scales_cutlass = [], [], []
    for i in range(expert_num):
        w13_ref_, w13_cutlass_, w13_scales_cutlass_, _ = cutlass_quantize(
            torch.bfloat16,
            w13_bf16[i].T,
            scalar_types.int4,
            torch.bfloat16,
            group_size,
            zero_points=False,
        )
        w13_ref.append(w13_ref_)
        w13_cutlass.append(w13_cutlass_.view(torch.int32))
        w13_scales_cutlass.append(w13_scales_cutlass_)
        w2_ref_, w2_cutlass_, w2_scales_cutlass_, _ = cutlass_quantize(
            torch.bfloat16,
            w2_bf16[i].T,
            scalar_types.int4,
            torch.bfloat16,
            group_size,
            zero_points=False,
        )
        w2_ref.append(w2_ref_)
        w2_cutlass.append(w2_cutlass_.view(torch.int32))
        w2_scales_cutlass.append(w2_scales_cutlass_)
    w13_ref = torch.stack(w13_ref)
    w13_cutlass = torch.stack(w13_cutlass)
    w13_scales_cutlass = torch.stack(w13_scales_cutlass)
    w2_ref = torch.stack(w2_ref)
    w2_cutlass = torch.stack(w2_cutlass)
    w2_scales_cutlass = torch.stack(w2_scales_cutlass)

    # routing
    routing_logits = (
        torch.randn((bs, expert_num), device=device, dtype=torch.float32) * 1.5
    )
    routing_bias = torch.randn(expert_num, device=device, dtype=torch.float32) * 0.8
    topk_weights, topk_ids = grouped_topk(
        hidden_states=x,
        gating_output=routing_logits,
        topk=topk,
        renormalize=False,  # DeepSeekV3 doesn't renormalize
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",  # DeepSeekV3 uses sigmoid
        routed_scaling_factor=2.827,
        e_score_correction_bias=routing_bias,
    )

    # cutlass output
    a_strides1_c_strides2 = torch.full(
        (expert_num,),
        hidden_size,
        device=device,
        dtype=torch.int64,
    )
    a_strides2 = torch.full(
        (expert_num,),
        intermediate_size,
        device=device,
        dtype=torch.int64,
    )
    c_strides1 = torch.full(
        (expert_num,),
        2 * intermediate_size,
        device=device,
        dtype=torch.int64,
    )

    # sizeof(StrideS) = 16 bytes, so we need to use 2xint64 to encode it
    s_strides1 = torch.zeros((expert_num, 2), device=device, dtype=torch.int64)
    s_strides1[:, 0] = 2 * intermediate_size

    s_strides2 = torch.zeros((expert_num, 2), device=device, dtype=torch.int64)
    s_strides2[:, 0] = hidden_size
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w13_scales_cutlass,
        w2_scale=w2_scales_cutlass,
        w1_zp=None,
        w2_zp=None,
        block_shape=[0, group_size],
    )
    w13_cutlass, b_strides1 = ops.cutlass_reorder_int4b_grouped(w13_cutlass)
    w2_cutlass, b_strides2 = ops.cutlass_reorder_int4b_grouped(w2_cutlass)
    cutlass_output = cutlass_moe_w4a16_bf16(
        x,
        w13_cutlass,
        w2_cutlass,
        topk_weights,
        topk_ids,
        quant_config=quant_config,
        activation="silu",
        global_num_experts=expert_num,
        expert_map=None,
        a_strides1=a_strides1_c_strides2,
        a_strides2=a_strides2,
        b_strides1=b_strides1,
        b_strides2=b_strides2,
        c_strides1=c_strides1,
        c_strides2=a_strides1_c_strides2,
        s_strides1=s_strides1,
        s_strides2=s_strides2,
        group_size=group_size,
    )

    # reference output
    ref_output = fused_experts(
        x,
        w13_ref.transpose(1, 2).contiguous(),
        w2_ref.transpose(1, 2).contiguous(),
        topk_weights,
        topk_ids,
        activation="silu",
        global_num_experts=expert_num,
    )

    torch.testing.assert_close(cutlass_output, ref_output, atol=1e5, rtol=0.05)
