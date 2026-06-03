# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 FlyDSL Project Contributors


import pytest
import torch
from aiter.ops.shuffle import shuffle_weight

from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_flydsl_moe import fused_flydsl_moe
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

RoutingBuffers = tuple[
    torch.Tensor,  # sorted_token_ids
    torch.Tensor,  # sorted_weights
    torch.Tensor,  # sorted_expert_ids
    torch.Tensor,  # num_valid_ids (shape [1], i32)
    int,  # sorted_size
    int,  # blocks
]


def _pack_shuffled_int8_to_packed_int4_no_perm(x_shuf_i8: torch.Tensor) -> torch.Tensor:
    """Pack a preshuffled int8 tensor (values in [-8, 7]) into packed int4 bytes.
    Each contiguous 8-value block [v0..v7] -> 4 bytes:
      b0=(v4<<4)|v0, b1=(v5<<4)|v1, b2=(v6<<4)|v2, b3=(v7<<4)|v3.
    This matches the 7-op in-kernel unpack sequence and avoids any v_perm.
    """
    flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
    assert flat.numel() % 8 == 0
    u = (flat & 0xF).to(torch.uint8).view(-1, 8)
    out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
    out[:, 0] = u[:, 0] | (u[:, 4] << 4)
    out[:, 1] = u[:, 1] | (u[:, 5] << 4)
    out[:, 2] = u[:, 2] | (u[:, 6] << 4)
    out[:, 3] = u[:, 3] | (u[:, 7] << 4)
    return out.view(-1).to(torch.int8)


def _unpack_gptq_int32_to_signed_int4(w_int32):
    """Unpack GPTQ int32 [E, K//8, N] to signed int4 values [E, N, K] (as int8).
    Shared by both the packed-int4 and bf16-dequant paths.
    """
    E = w_int32.shape[0]
    # [E, K//8, N] -> transpose -> [E, N, K//8]
    w = w_int32.transpose(1, 2).contiguous()
    N = w.shape[1]
    K_div8 = w.shape[2]
    K = K_div8 * 8

    # Unpack int32 -> 8 x uint4 values along K
    w_expanded = w.unsqueeze(-1).expand(E, N, K_div8, 8)  # [E, N, K//8, 8]
    shifts = torch.arange(8, device=w.device) * 4  # [0, 4, 8, ..., 28]
    nibbles = ((w_expanded >> shifts) & 0xF).to(torch.int8)  # [E, N, K//8, 8]
    nibbles = nibbles.reshape(E, N, K)  # [E, N, K] unsigned int4 as int8

    # Convert unsigned [0,15] to signed [-8,7]
    signed = nibbles.to(torch.int16) - 8
    signed = signed.to(torch.int8)  # [E, N, K] signed int4 as int8
    return signed


def _gptq_int32_to_flydsl_packed(w_int32):
    """Convert GPTQ int32 [E, K//8, N] to FlyDSL shuffled packed int4 [E, N, K//2].
    Steps:
    1. Unpack int32 to individual signed int4 values (as int8)
    2. Apply FlyDSL preshuffle (on individual int8 values)
    3. Pack with FlyDSL's interleaved int4 packing
    """
    signed = _unpack_gptq_int32_to_signed_int4(w_int32)
    E, N, K = signed.shape

    # FlyDSL preshuffle (operates on individual values)
    shuffled = shuffle_weight(signed, layout=(16, 16))

    # FlyDSL interleaved int4 packing
    packed = _pack_shuffled_int8_to_packed_int4_no_perm(shuffled).contiguous()
    return packed.view(E, N, K // 2)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx950()),
    reason="FlyDSL MoE requires HIP device and gfx950 arch",
)
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
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

    w13 = w13_weight
    w13 = _gptq_int32_to_flydsl_packed(w13)
    w13 = w13.view(-1).contiguous()

    w2 = w2_weight
    w2 = _gptq_int32_to_flydsl_packed(w2)
    w2 = w2.view(-1).contiguous()

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

    out = fused_flydsl_moe(
        x,
        w13,
        w2,
        num_experts,
        inter_dim,
        topk_weights,
        topk_ids,
        w1_scale=w13_scale_flydsl,
        w2_scale=w2_scale_flydsl,
        topk=topk_weights.shape[-1],
        group_size=group_size,
        doweight_stage1=False,
        scale_is_bf16=True,
    )

    assert torch.allclose(out, out_ref, atol=0.5, rtol=0.1)


if __name__ == "__main__":
    test_flydsl_moe(512, 256)
