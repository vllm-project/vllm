# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from SGLang:
# https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/tests/test_es_fp8_blockwise_moe.py

"""Tests for SM100 CUTLASS MXFP8 grouped MoE kernels."""

import random

import pytest
import torch

from tests.kernels.utils import torch_moe_single
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
    return current_platform.is_cuda() and current_platform.is_device_capability_family(
        100
    )


def compute_ref_output(
    input_tensor: torch.Tensor,
    weight_list: list[torch.Tensor],
    expert_offsets: list[int],
    expert_offset: int,
    num_experts: int,
) -> torch.Tensor:
    # Build a top-1 routing score so each token maps to its owning expert.
    score = torch.full(
        (expert_offset, num_experts),
        -1e9,
        device=input_tensor.device,
        dtype=torch.float32,
    )
    for g in range(num_experts):
        start = expert_offsets[g]
        end = expert_offsets[g + 1] if g + 1 < num_experts else expert_offset
        score[start:end, g] = 0.0

    return torch_moe_single(
        input_tensor, torch.stack(weight_list, dim=0), score, topk=1
    )


def compute_kernel_output(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    problem_sizes: list[list[int]],
    aux_problem_sizes: list[list[int]],
    expert_offsets: list[int],
    aux_expert_offsets: list[int],
    input_blockscale_offsets: list[int],
    weight_blockscale_offsets: list[int],
    input_blockscale_offset: int,
    n_g: int,
    k_g: int,
    num_experts: int,
    expert_offset: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    device = input_tensor.device
    _problem_sizes = torch.tensor(problem_sizes).to(device=device, dtype=torch.int32)
    _aux_problem_sizes = torch.tensor(aux_problem_sizes).to(
        device=device, dtype=torch.int32
    )
    _expert_offsets = torch.tensor(expert_offsets).to(device=device, dtype=torch.int32)
    _aux_expert_offsets = torch.tensor(aux_expert_offsets).to(
        device=device, dtype=torch.int32
    )
    _input_blockscale_offsets = torch.tensor(input_blockscale_offsets).to(
        device=device, dtype=torch.int32
    )
    _weight_blockscale_offsets = torch.tensor(weight_blockscale_offsets).to(
        device=device, dtype=torch.int32
    )

    input_quant = torch.zeros_like(
        input_tensor, dtype=torch.float8_e4m3fn, device=device
    )
    input_scale_factor = torch.zeros(
        (input_blockscale_offset, k_g // 32), dtype=torch.uint8, device=device
    )

    weight_quant = torch.zeros_like(
        weight_tensor, dtype=torch.float8_e4m3fn, device=device
    )
    weight_scale_factor = torch.zeros(
        (num_experts, n_g, k_g // 32), dtype=torch.uint8, device=device
    )

    ops.mxfp8_experts_quant(
        input_tensor,
        _problem_sizes,
        _expert_offsets,
        _input_blockscale_offsets,
        input_quant,
        input_scale_factor,
    )

    ops.mxfp8_experts_quant(
        weight_tensor,
        _aux_problem_sizes,
        _aux_expert_offsets,
        _weight_blockscale_offsets,
        weight_quant,
        weight_scale_factor,
    )
    weight_quant = weight_quant.view(num_experts, n_g, k_g).transpose(1, 2)
    weight_scale_factor = weight_scale_factor.view(
        num_experts, n_g, k_g // 32
    ).transpose(1, 2)

    output = torch.empty((expert_offset, n_g), device=device, dtype=out_dtype)
    ops.cutlass_mxfp8_grouped_mm(
        input_quant,
        weight_quant,
        input_scale_factor,
        weight_scale_factor,
        output,
        _problem_sizes,
        _expert_offsets,
        _input_blockscale_offsets,
    )
    return output


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason=(
        "cutlass_mxfp8_grouped_mm and mxfp8_experts_quant "
        "are only supported on CUDA SM100"
    ),
)
@pytest.mark.parametrize("num_experts", [8, 16, 32, 64])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
def test_cutlass_mxfp8_grouped_mm(num_experts, out_dtype):
    device = "cuda"
    alignment = 128
    n_g = random.randint(1, 64) * alignment
    k_g = random.randint(1, 64) * alignment

    expert_offset = 0
    expert_offsets = []
    aux_expert_offset = 0
    aux_expert_offsets = []
    input_blockscale_offset = 0
    input_blockscale_offsets = []
    weight_blockscale_offset = 0
    weight_blockscale_offsets = []
    problem_sizes = []
    aux_problem_sizes = []
    input_list = []
    weight_list = []

    for g in range(num_experts):
        m_g = random.randint(1, 512)
        expert_offsets.append(expert_offset)
        expert_offset += m_g
        aux_expert_offsets.append(aux_expert_offset)
        aux_expert_offset += n_g
        input_blockscale_offsets.append(input_blockscale_offset)
        input_blockscale_offset += align(m_g, 128)
        weight_blockscale_offsets.append(weight_blockscale_offset)
        weight_blockscale_offset += n_g  # n_g already align to 128
        problem_sizes.append([m_g, n_g, k_g])
        aux_problem_sizes.append([n_g, m_g, k_g])

        input_tensor = torch.normal(
            0.0, std=1.0, size=(m_g, k_g), device=device, dtype=out_dtype
        )  # (M, K):(K, 1)
        weight_tensor = torch.normal(
            0.0, std=1.0, size=(n_g, k_g), device=device, dtype=out_dtype
        )  # (N, K):(K, 1)

        input_list.append(input_tensor)
        weight_list.append(weight_tensor)
    input_tensor = torch.concat(input_list, dim=0)
    weight_tensor = torch.concat(weight_list, dim=0)

    ref_output = compute_ref_output(
        input_tensor=input_tensor,
        weight_list=weight_list,
        expert_offsets=expert_offsets,
        expert_offset=expert_offset,
        num_experts=num_experts,
    )
    output = compute_kernel_output(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        problem_sizes=problem_sizes,
        aux_problem_sizes=aux_problem_sizes,
        expert_offsets=expert_offsets,
        aux_expert_offsets=aux_expert_offsets,
        input_blockscale_offsets=input_blockscale_offsets,
        weight_blockscale_offsets=weight_blockscale_offsets,
        input_blockscale_offset=input_blockscale_offset,
        n_g=n_g,
        k_g=k_g,
        num_experts=num_experts,
        expert_offset=expert_offset,
        out_dtype=out_dtype,
    )

    for g in range(num_experts):
        baseline = ref_output[
            expert_offsets[g] : (expert_offsets[g] + problem_sizes[g][0])
        ]
        actual = output[expert_offsets[g] : (expert_offsets[g] + problem_sizes[g][0])]
        diff = calc_diff(actual, baseline)
        assert diff < 0.001
        print(
            f"m_g={baseline.shape[0]} n_g={n_g} k_g={k_g} num_experts={num_experts}, "
            f"out_dtype={out_dtype}, diff={diff:.5f}: OK"
        )


if __name__ == "__main__":
    pytest.main([__file__])
