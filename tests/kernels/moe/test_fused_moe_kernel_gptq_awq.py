# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

DEVICE = "xpu" if current_platform.is_xpu() else "cuda"

vllm_config = VllmConfig()

WN16_MNK = [
    (1, 128, 128),
    (32, 2048, 128),
    (222, 2048, 1024),
]
NUM_EXPERTS = [8]
TOP_KS = [2]
GROUP_SIZES = [128]
WEIGHT_BITS = [4, 8]
HAS_ZP = [True, False]


def fused_moe(
    hidden_states,
    w1,
    w2,
    score,
    topk,
    renormalize=False,
    quant_config=None,
    global_num_experts=-1,
    expert_map=None,
):
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score.float(), topk, renormalize
    )
    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        quant_config=quant_config,
    )


def torch_moe(a, w1, w2, score, topk):
    """Pure-PyTorch MoE reference for correctness validation.
    
    Implements fused MoE with SiLU+Mul activation and expert routing.
    Used as reference to validate Triton kernel outputs.
    """
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    m, k = a.shape
    a_rep = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)
    out = torch.zeros(m * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_flat = topk_ids.view(-1)
    act = SiluAndMul()
    for i in range(w1.shape[0]):
        mask = topk_flat == i
        if mask.sum():
            tmp = a_rep[mask] @ w1[i].transpose(0, 1)
            tmp = act(tmp)
            out[mask] = tmp @ w2[i].transpose(0, 1)

    return (
        (out.view(m, -1, w2.shape[1]).to(torch.float32) * topk_weight.view(m, -1, 1))
        .sum(dim=1)
        .to(out.dtype)
    )


def _prepare_quantized_weights(e, n, k, group_size, weight_bits, has_zp, device, dtype):
    """Prepare quantized MoE weights with scales and zero-points.
    
    CRITICAL: qzeros must be torch.int32 (packed int4 format) to match
    upstream awq.py and test_awq_triton.py conventions. NOT float16 or uint8.
    
    Args:
        e: number of experts
        n, k: weight dimensions
        group_size: quantization group size
        weight_bits: 4 or 8
        has_zp: include zero-points
        device, dtype: tensor placement and type
    
    Returns:
        (w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp)
    """
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10

    if weight_bits == 4:
        pack_factor = 2
        quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    else:
        pack_factor = 1
        quant_type = scalar_types.uint8 if has_zp else scalar_types.uint8b128

    w1_ref = w1.clone()
    w2_ref = w2.clone()
    w1_qw = torch.empty((e, 2 * n, k // pack_factor), device=device, dtype=torch.uint8)
    w2_qw = torch.empty((e, k, n // pack_factor), device=device, dtype=torch.uint8)
    w1_sc = torch.empty((e, 2 * n, k // group_size), device=device, dtype=dtype)
    w2_sc = torch.empty((e, k, n // group_size), device=device, dtype=dtype)

    w1_zp = torch.empty(
        (e, 2 * n // pack_factor, k // group_size), device=device, dtype=torch.int32
    )
    w2_zp = torch.empty(
        (e, k // pack_factor, n // group_size), device=device, dtype=torch.int32
    )

    for i in range(e * 2):
        expert_id = i % e
        if i // e == 0:
            w, w_ref_arr, w_qw_arr, w_sc_arr, w_zp_arr = w1, w1_ref, w1_qw, w1_sc, w1_zp
        else:
            w, w_ref_arr, w_qw_arr, w_sc_arr, w_zp_arr = w2, w2_ref, w2_qw, w2_sc, w2_zp

        weight, qweight, scales, qzeros = quantize_weights(
            w[expert_id].T, quant_type, group_size, has_zp, False
        )
        weight = weight.T
        qweight = qweight.T.contiguous().to(torch.uint8)
        scales = scales.T

        if has_zp:
            qzeros = qzeros.T.contiguous().to(torch.int32)

        if weight_bits == 4:
            qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
            if has_zp:
                qzeros = (qzeros[1::2, :] << 4) + qzeros[::2, :]

        w_ref_arr[expert_id] = weight
        w_qw_arr[expert_id] = qweight
        w_sc_arr[expert_id] = scales
        if has_zp:
            w_zp_arr[expert_id] = qzeros

    return w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp


@pytest.mark.parametrize("m,n,k", WN16_MNK)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("has_zp", HAS_ZP)
@pytest.mark.parametrize("weight_bits", WEIGHT_BITS)
def test_fused_moe_wn16(m, n, k, e, topk, group_size, has_zp, weight_bits):
    """Test fused_moe_kernel_gptq_awq correctness vs PyTorch reference.

    Adapted from upstream test_moe.py::test_fused_moe_wn16 with reduced params.
    """
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)

    w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = _prepare_quantized_weights(
        e, n, k, group_size, weight_bits, has_zp, DEVICE, dtype
    )

    if weight_bits == 4:
        quant_config = int4_w4a16_moe_quant_config(
            w1_scale=w1_sc,
            w2_scale=w2_sc,
            w1_zp=w1_zp if has_zp else None,
            w2_zp=w2_zp if has_zp else None,
            block_shape=[0, group_size],
        )
    else:
        quant_config = int8_w8a16_moe_quant_config(
            w1_scale=w1_sc,
            w2_scale=w2_sc,
            w1_zp=w1_zp if has_zp else None,
            w2_zp=w2_zp if has_zp else None,
            block_shape=[0, group_size],
        )

    with set_current_vllm_config(vllm_config):
        triton_output = fused_moe(
            a,
            w1_qw,
            w2_qw,
            score,
            topk,
            renormalize=False,
            global_num_experts=e,
            quant_config=quant_config,
        )
        torch_output = torch_moe(a, w1_ref, w2_ref, score, topk)

    torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
