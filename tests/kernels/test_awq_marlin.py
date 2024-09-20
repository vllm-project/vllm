"""Test AWQ with fused MoE Marlin kernels.

Run `pytest tests/kernels/test_awq_marlin.py`.
"""
from typing import List

import pytest
import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    fused_marlin_moe, single_marlin_moe)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    awq_marlin_quantize
)

def stack_and_dev(tensors: List[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)

def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))

def torch_moe(a, w1, w2, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


def torch_moe_single(a, w, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    _, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.view(-1)
    for i in range(w.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = a[mask] @ w[i].transpose(0, 1)
    return (out.view(B, -1, w.shape[1])).sum(dim=1)

@pytest.mark.skip("TODO")
@pytest.mark.parametrize("m", [64, 512, 222, 33, 1])
@pytest.mark.parametrize("n", [128, 2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 1024, 512])
@pytest.mark.parametrize("e", [4, 8, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("num_bits", [4, 8])
def test_fused_marlin_moe_awq(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
    num_bits: int,
):
    torch.manual_seed(7)

    if topk > e:
        return

    quant_type = (scalar_types.uint4b8
                  if num_bits == 4 else scalar_types.uint8b128)
    dtype = torch.float16
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    w_ref1_l = []
    qweights1_l = []
    scales1_l = []
    zp1_l = []

    for i in range(w1.shape[0]):
        w_ref1, qweight1, scales1, zp1 = awq_marlin_quantize(
            w1[i].transpose(1, 0), quant_type, group_size)
        w_ref1_l.append(w_ref1)
        qweights1_l.append(qweight1)
        scales1_l.append(scales1)
        zp1_l.append(zp1)

    w_ref1 = stack_and_dev(w_ref1_l)
    qweight1 = stack_and_dev(qweights1_l).contiguous()
    scales1 = stack_and_dev(scales1_l)
    zp1 = stack_and_dev(zp1_l)

    w_ref2_l = []
    qweights2_l = []
    scales2_l = []
    zp2_l = []

    for i in range(w2.shape[0]):
        w_ref2, qweight2, scales2, zp2 = awq_marlin_quantize(
            w2[i].transpose(1, 0), quant_type, group_size)
        w_ref2_l.append(w_ref2)
        qweights2_l.append(qweight2)
        scales2_l.append(scales2)
        zp2_l.append(zp2)

    w_ref2 = stack_and_dev(w_ref2_l)
    qweight2 = stack_and_dev(qweights2_l).contiguous()
    scales2 = stack_and_dev(scales2_l)
    zp2 = stack_and_dev(zp2_l)

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(a, score, topk, False)

    triton_output = fused_moe(
        a,
        w_ref1.transpose(1, 2).contiguous(),
        w_ref2.transpose(1, 2).contiguous(),
        score,
        topk,
        renormalize=False,
    )
    marlin_output = fused_marlin_moe(
        a,
        qweight1,
        qweight2,
        scales1,
        scales2,
        score,
        topk_weights,
        topk_ids,
        w1_zeros=zp1,
        w2_zeros=zp2,
        num_bits=num_bits,
    )

    assert compute_max_diff(marlin_output, triton_output) < 4e-2


# @pytest.mark.skip("This test is here for the sake of debugging, "
#                   "don't run it in automated tests.")
# @pytest.mark.parametrize("m", [64, 512, 222, 33, 1])
# @pytest.mark.parametrize("n", [128, 2048, 256, 1024])
# @pytest.mark.parametrize("k", [128, 1024, 512])
# @pytest.mark.parametrize("e", [4, 8, 64])
# @pytest.mark.parametrize("topk", [2, 6])
# @pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
# @pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("m", [1])
@pytest.mark.parametrize("n", [128])
@pytest.mark.parametrize("k", [128])
@pytest.mark.parametrize("e", [4])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("group_size", [-1])
@pytest.mark.parametrize("num_bits", [4])
def test_single_marlin_moe_multiply_awq(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
    num_bits: int,
):
    if topk > e:
        return

    quant_type = (scalar_types.uint4b8
                  if num_bits == 4 else scalar_types.uint8b128)
    dtype = torch.float16
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w = torch.randn((e, n, k), device="cuda", dtype=dtype) / 10

    w_ref_l = []
    qweights_l = []
    scales_l = []
    zp_l = []

    for i in range(w.shape[0]):
        w_ref, qweight, scales, zp = awq_marlin_quantize(
            w[i].transpose(1, 0), quant_type, group_size)
        w_ref_l.append(w_ref)
        qweights_l.append(qweight)
        scales_l.append(scales)
        zp_l.append(zp)

    w_ref = stack_and_dev(w_ref_l)
    qweight = stack_and_dev(qweights_l).contiguous()
    scales = stack_and_dev(scales_l)
    zp = stack_and_dev(zp_l)

    print(scales.dtype)
    print(zp.dtype)

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    marlin_output = single_marlin_moe(a,
                                      qweight,
                                      scales,
                                      score,
                                      topk,
                                      renormalize=False,
                                      w_zeros=zp,
                                      num_bits=num_bits)

    torch_output = torch_moe_single(a, w_ref.transpose(1, 2), score, topk)

    assert compute_max_diff(marlin_output, torch_output) < 1e-2
