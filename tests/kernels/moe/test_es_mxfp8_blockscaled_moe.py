# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from SGLang:
# https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/tests/test_es_fp8_blockwise_moe.py

"""Tests for SM100 MXFP8 blockscaled grouped MoE kernels."""

import random

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

random.seed(42)
set_random_seed(42)


def align(val: int, alignment: int = 128) -> int:
    return int((val + alignment - 1) // alignment * alignment)


# Copy from: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def is_sm100_supported() -> bool:
    return current_platform.is_cuda() and current_platform.has_device_capability(100)


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="es_sm100_mxfp8_blockscaled kernels are only supported on CUDA SM100+",
)
@pytest.mark.parametrize("num_experts", [8, 16, 32, 64])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_es_sm100_mxfp8_blockscaled_grouped_mm(num_experts, out_dtype):
    device = "cuda"
    alignment = 128
    n_g = random.randint(1, 64) * alignment
    k_g = random.randint(1, 64) * alignment

    expert_offset = 0
    expert_offsets = []
    aux_expert_offset = 0
    aux_expert_offsets = []
    a_blockscale_offset = 0
    a_blockscale_offsets = []
    b_blockscale_offset = 0
    b_blockscale_offsets = []
    problem_sizes = []
    aux_problem_sizes = []
    a_list = []
    b_list = []
    ref_d_list = []

    for g in range(num_experts):
        m_g = random.randint(1, 512)
        expert_offsets.append(expert_offset)
        expert_offset += m_g
        aux_expert_offsets.append(aux_expert_offset)
        aux_expert_offset += n_g
        a_blockscale_offsets.append(a_blockscale_offset)
        a_blockscale_offset += align(m_g, 128)
        b_blockscale_offsets.append(b_blockscale_offset)
        b_blockscale_offset += n_g  # n_g already align to 128
        problem_sizes.append([m_g, n_g, k_g])
        aux_problem_sizes.append([n_g, m_g, k_g])

        a = torch.normal(
            0.0, std=1.0, size=(m_g, k_g), device=device, dtype=out_dtype
        )  # (M, K):(K, 1)
        b = torch.normal(
            0.0, std=1.0, size=(n_g, k_g), device=device, dtype=out_dtype
        )  # (N, K):(K, 1)

        a_list.append(a)
        b_list.append(b)
        ref_d = a @ b.T
        ref_d_list.append(ref_d)
    a = torch.concat(a_list, dim=0)
    b = torch.concat(b_list, dim=0)

    _problem_sizes = torch.tensor(problem_sizes).to(device=device, dtype=torch.int32)
    _aux_problem_sizes = torch.tensor(aux_problem_sizes).to(
        device=device, dtype=torch.int32
    )
    _expert_offsets = torch.tensor(expert_offsets).to(device=device, dtype=torch.int32)
    _aux_expert_offsets = torch.tensor(aux_expert_offsets).to(
        device=device, dtype=torch.int32
    )
    _a_blockscale_offsets = torch.tensor(a_blockscale_offsets).to(
        device=device, dtype=torch.int32
    )
    _b_blockscale_offsets = torch.tensor(b_blockscale_offsets).to(
        device=device, dtype=torch.int32
    )

    a_quant = torch.zeros_like(a, dtype=torch.float8_e4m3fn, device=device)
    a_scale_factor = torch.zeros(
        (a_blockscale_offset, k_g // 32), dtype=torch.uint8, device=device
    )

    b_quant = torch.zeros_like(b, dtype=torch.float8_e4m3fn, device=device)
    b_scale_factor = torch.zeros(
        (num_experts, n_g, k_g // 32), dtype=torch.uint8, device=device
    )

    ops.es_sm100_mxfp8_blockscaled_grouped_quant(
        a,
        _problem_sizes,
        _expert_offsets,
        _a_blockscale_offsets,
        a_quant,
        a_scale_factor,
    )

    ops.es_sm100_mxfp8_blockscaled_grouped_quant(
        b,
        _aux_problem_sizes,
        _aux_expert_offsets,
        _b_blockscale_offsets,
        b_quant,
        b_scale_factor,
    )
    b_quant = b_quant.view(num_experts, n_g, k_g).transpose(1, 2)
    b_scale_factor = b_scale_factor.view(num_experts, n_g, k_g // 32).transpose(1, 2)

    d = torch.empty((expert_offset, n_g), device=device, dtype=out_dtype)
    ops.es_sm100_mxfp8_blockscaled_grouped_mm(
        a_quant,
        b_quant,
        a_scale_factor,
        b_scale_factor,
        d,
        _problem_sizes,
        _expert_offsets,
        _a_blockscale_offsets,
    )

    for g in range(num_experts):
        baseline = ref_d_list[g]
        actual = d[expert_offsets[g] : (expert_offsets[g] + problem_sizes[g][0])]
        diff = calc_diff(actual, baseline)
        assert diff < 0.001
        print(
            f"m_g={baseline.shape[0]} n_g={n_g} k_g={k_g} num_experts={num_experts}, "
            f"out_dtype={out_dtype}, diff={diff:.5f}: OK"
        )


if __name__ == "__main__":
    pytest.main([__file__])
