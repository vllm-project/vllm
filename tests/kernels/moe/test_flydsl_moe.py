# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 FlyDSL Project Contributors


import importlib.util

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    rocm_aiter_fused_experts,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

if not (current_platform.is_rocm() and on_gfx950()):
    pytest.skip("This test can only run on ROCm and gfx950.", allow_module_level=True)

aiter_available = importlib.util.find_spec("aiter") is not None

if not aiter_available:
    pytest.skip("These tests require AITER to run.", allow_module_level=True)

from tests.kernels.moe.utils import make_dummy_moe_config  # noqa: E402
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E402, E501
    compressed_tensors_moe_w4a16_flydsl,
)


def _torch_w4a16_reference(
    x: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    unpack = compressed_tensors_moe_w4a16_flydsl._unpack_gptq_int32_to_signed_int4
    w13 = unpack(w13_weight)
    w2 = unpack(w2_weight)
    result = torch.empty_like(x)

    for token_idx in range(x.shape[0]):
        expert_ids = topk_ids[token_idx]
        w13_selected = w13[expert_ids].float()
        w13_scale_selected = (
            w13_scale[expert_ids]
            .permute(0, 2, 1)
            .repeat_interleave(group_size, dim=2)
            .float()
        )
        gate_up = torch.einsum(
            "h,enh->en",
            x[token_idx].float(),
            w13_selected * w13_scale_selected,
        )
        gate, up = gate_up.chunk(2, dim=1)
        intermediate = F.silu(gate) * up

        w2_selected = w2[expert_ids].float()
        w2_scale_selected = (
            w2_scale[expert_ids]
            .permute(0, 2, 1)
            .repeat_interleave(group_size, dim=2)
            .float()
        )
        expert_outputs = torch.einsum(
            "ei,ehi->eh",
            intermediate,
            w2_selected * w2_scale_selected,
        )
        result[token_idx] = torch.sum(
            expert_outputs * topk_weights[token_idx, :, None], dim=0
        ).to(result.dtype)

    return result


@pytest.mark.parametrize(
    "num_tokens", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
)
@pytest.mark.parametrize("inter_dim", [256, 512])
def test_flydsl_moe(num_tokens: int, inter_dim: int):
    device = "cuda"
    topk = 8
    num_experts = 384
    hidden_size = 7168
    packed_factor = 8
    w13_num_shards = 2
    params_dtype = torch.bfloat16
    group_size = 32
    w2_scales_size = inter_dim
    scale_factor = 0.01

    num_groups_w2 = w2_scales_size // group_size
    num_groups_w13 = hidden_size // group_size

    w13_weight = torch.randint(
        0,
        255,
        (num_experts, hidden_size // packed_factor, w13_num_shards * inter_dim),
        dtype=torch.int32,
        device=device,
    )

    w2_weight = torch.randint(
        0,
        255,
        (num_experts, inter_dim // packed_factor, hidden_size),
        dtype=torch.int32,
        device=device,
    )
    w13_scale = scale_factor * torch.randn(
        num_experts,
        num_groups_w13,
        w13_num_shards * inter_dim,
        dtype=params_dtype,
        device=device,
    )
    w2_scale = scale_factor * torch.randn(
        num_experts, num_groups_w2, hidden_size, dtype=params_dtype, device=device
    )

    w13_weight_packed = w13_weight.transpose(1, 2).contiguous().view(torch.uint8)
    w2_weight_packed = w2_weight.transpose(1, 2).contiguous().view(torch.uint8)
    w13_weight_scale = w13_scale.transpose(1, 2).contiguous()
    w2_weight_scale = w2_scale.transpose(1, 2).contiguous()

    moe_quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w13_weight_scale,
        w2_scale=w2_weight_scale,
        w1_zp=None,
        w2_zp=None,
        block_shape=[0, group_size],
    )
    score = torch.rand((num_tokens, num_experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)

    w13 = w13_weight
    w13 = compressed_tensors_moe_w4a16_flydsl._gptq_int32_to_flydsl_packed(w13)

    w2 = w2_weight
    w2 = compressed_tensors_moe_w4a16_flydsl._gptq_int32_to_flydsl_packed(w2)

    w13_scale_flydsl = w13_scale
    w2_scale_flydsl = w2_scale

    if group_size > 0 and w13_scale.dim() == 3 and w13_scale.shape[1] > 1:
        E, G, N = w13_scale.shape
        w13_scale_flydsl = (
            w13_scale_flydsl.view(E, G // 2, 2, N)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(-1)
            .contiguous()
        )
    elif w13_scale.dim() == 3 and w13_scale.shape[1] == 1:
        w13_scale_flydsl = w13_scale_flydsl.squeeze(1)

    if group_size > 0 and w2_scale.dim() == 3 and w2_scale.shape[1] > 1:
        E, G, N = w2_scale.shape
        w2_scale_flydsl = (
            w2_scale_flydsl.view(E, G // 2, 2, N)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(-1)
            .contiguous()
        )
    elif w2_scale.dim() == 3 and w2_scale.shape[1] == 1:
        w2_scale_flydsl = w2_scale_flydsl.squeeze(1)

    w13_scale_flydsl = w13_scale_flydsl.contiguous()
    w2_scale_flydsl = w2_scale_flydsl.contiguous()

    w13.is_shuffled = True
    w2.is_shuffled = True

    aiter_quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w13_scale_flydsl,
        w2_scale=w2_scale_flydsl,
        block_shape=[0, group_size],
    )
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size=inter_dim,
        in_dtype=torch.bfloat16,
    )
    out = rocm_aiter_fused_experts(
        hidden_states=x,
        w1=w13,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        moe_config=moe_config,
        activation=MoEActivation.SILU,
        quant_config=aiter_quant_config,
        output_dtype=torch.bfloat16,
    )
    if hasattr(torch.ops._moe_C, "moe_align_block_size"):
        out_ref = fused_experts(
            x,
            w13_weight_packed,
            w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            global_num_experts=num_experts,
            expert_map=None,
            quant_config=moe_quant_config,
        )
    else:
        out_ref = _torch_w4a16_reference(
            x,
            w13_weight,
            w2_weight,
            w13_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            group_size,
        )

    assert torch.allclose(out, out_ref, atol=0.5, rtol=0.1)


if __name__ == "__main__":
    test_flydsl_moe(512, 256)
