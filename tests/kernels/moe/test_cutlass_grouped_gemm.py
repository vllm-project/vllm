# SPDX-License-Identifier: Apache-2.0

import random
import pytest
import torch
from typing import Tuple

from vllm import _custom_ops as ops

def cdiv(a, b):
    return (a + b - 1) // b

def scale_shape(shape, group_shape):
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))

def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)
    

def baseline_scaled_mm(a, b, a_scales, b_scales, out_dtype):

    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s% t.shape[i] == 0
                t = (
                    t.unsqueeze(i+1)
                    .expand(*t.shape[:i+1], s // t.shape[i], *t.shape[i+1:])
                    .flatten(i, i+1)
                )
        return t

    scale_a = group_broadcast(a_scales, a.shape)
    scale_b = group_broadcast(b_scales, b.shape)

    return torch.mm(
        (scale_a * a.to(dtype=torch.float32)),
        (scale_b * b.to(dtype=torch.float32))
    ).to(dtype=out_dtype)

@pytest.mark.parametrize("num_groups, expected_m_per_group, k, n", [
    (4, 8192, 7168, 4096),
    # (16, 128, 128, 128),
    # (64, 128, 128, 128),
])
@pytest.mark.parametrize("out_dtype", [torch.half])
def test_cutlass_grouped_gemm(
    num_groups: int,
    expected_m_per_group: int,
    k: int,
    n: int,
    out_dtype: torch.dtype,
):
    device = "cuda"
    alignment = 8
    group_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    m = sum([cdiv(m, alignment) * alignment for m in group_ms])

    x = torch.randn((m, k), device=device, dtype=out_dtype)
    y = torch.randn((num_groups, n, k), device=device, dtype=out_dtype)
    m_indicies = torch.empty(m, device=device, dtype=torch.int32)
    out = torch.empty((m, n), device=device, dtype=out_dtype)
    ref_out = torch.randn((m, n), device=device, dtype=out_dtype)

    start = 0
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        aligned_end = start + cdiv(group_m, alignment) * alignment
        m_indicies[start:aligned_end] = i
        m_indicies[aligned_end:actual_end] = -1
        ref_out[start:aligned_end] = x[start:aligned_end] @ y[i].t()
        start = aligned_end
    ref_out = torch.where((m_indicies == -1))
    

    print(num_groups, expected_m_per_group, k, n)


    
    

@pytest.mark.parametrize("num_experts", [8, 16, 64])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_cutlass_grouped_moe(
    num_experts: int,
    out_dtype: torch.dtype,
):
    device = "cuda"
    alignment = 8
    n_g = alignment * random.randint(1, 5) * 128
    k_g = alignment * random.randint(1, 5) * 128

    scale_a_group_shape = (1, 128)
    scale_b_group_shape = (128, 128)

    expert_offsets = torch.zeros((num_experts + 1), device=device, dtype=torch.int32)
    problem_sizes = torch.zeros((num_experts, 3), device=device, dtype=torch.int32)

    a_tensors = []
    b_tensors = []
    a_scales_tensors = []
    b_scales_tensors = []
    baseline_tensors = []

    for g in range(num_experts):
        m_g = alignment * random.randint(1, 64)
        expert_offsets[g+1] = expert_offsets[g] + m_g
        problem_sizes[g][:] = torch.tensor([m_g, n_g, k_g], device=device)

        a_g = to_fp8(torch.randn((m_g, k_g), device=device))
        b_g = to_fp8(torch.randn((n_g, k_g), device=device).t())
        a_tensors.append(a_g)
        b_tensors.append(b_g)

        scale_a_shape = scale_shape(a_g.shape, scale_a_group_shape)
        scale_b_shape = scale_shape(b_g.shape, scale_b_group_shape)

        a_scales_tensors.append(torch.randn(scale_a_shape, device=device) * 0.001)
        b_scales_tensors.append(torch.randn(scale_b_shape, device=device) * 0.001)

        baseline = baseline_scaled_mm(
            a_g, b_g, a_scales_tensors[-1], b_scales_tensors[-1], out_dtype
        )
        baseline_tensors.append(baseline)

    a_stack = torch.empty((expert_offsets[-1], k_g), device=device, dtype=torch.float8_e4m3fn)
    b_stack = torch.empty((num_experts, n_g, k_g), device=device, dtype=torch.float8_e4m3fn)

    for g in range(num_experts):
        a_stack[expert_offsets[g]:expert_offsets[g+1]] = a_tensors[g]
        b_stack[g] = b_tensors[g].t()
    b_stack = b_stack.transpose(1, 2)

    a_scale_stack = torch.empty((expert_offsets[-1], k_g // 128), device=device, dtype=torch.float32)
    b_scale_stack = torch.empty((num_experts, n_g // 128, k_g // 128), device=device, dtype=torch.float32)

    for g in range(num_experts):
        a_scale_stack[expert_offsets[g]:expert_offsets[g+1]] = a_scales_tensors[g]
        b_scale_stack[g] = b_scales_tensors[g].t()
    b_scale_stack = b_scale_stack.transpose(1, 2)

    c_out = torch.empty((expert_offsets[-1], n_g), device=device, dtype=out_dtype)

    ops.cutlass_blockwise_scaled_grouped_mm(
        c_out,
        a_stack,
        b_stack,
        a_scale_stack,
        b_scale_stack,
        problem_sizes,
        expert_offsets[:-1],
    )

    for g in range(num_experts):
        baseline = baseline_tensors[g]
        actual = c_out[expert_offsets[g]:expert_offsets[g+1]]

        torch.testing.assert_close(baseline, actual, atol=1e-2, rtol=5e-4)
    