# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

os.environ["VLLM_USE_DEEP_GEMM"] = "1"

import math
import random

import torch

from vllm.utils import deep_gemm as vllm_deep_gemm

BLOCK_SIZE = (128, 128)
BLOCK_N, BLOCK_K = BLOCK_SIZE


def generate_bf16_and_downcast_to_fp8(shape, device="cuda"):
    bf16_weight = torch.randn(shape, dtype=torch.bfloat16, device=device)
    fp8_weight = bf16_weight.to(dtype=torch.float8_e4m3fn)
    return fp8_weight


def run_batched_deepgemm_fp8(
    expected_group_batch_size: int,
    num_groups: int,
    output_size: int,
    input_size: int,
):
    weight = generate_bf16_and_downcast_to_fp8((num_groups, output_size, input_size))
    output_tiles = math.ceil(output_size / BLOCK_N)
    input_tiles = math.ceil(input_size / BLOCK_K)
    weight_scale = torch.randn(
        num_groups,
        output_tiles,
        input_tiles,
        dtype=torch.float32,
        device="cuda",
    )
    group_batch_size = [
        int(expected_group_batch_size * random.uniform(0.7, 1.3))
        for _ in range(num_groups)
    ]
    batch_size = sum(group_batch_size)
    group_batch_size = torch.tensor(
        group_batch_size,
        dtype=torch.int32,
        device="cuda",
    )
    x = generate_bf16_and_downcast_to_fp8((num_groups, batch_size, input_size))
    x_scale = torch.randn(
        num_groups,
        batch_size,
        input_tiles,
        dtype=torch.float32,
        device="cuda",
    )
    output = torch.zeros(
        num_groups,
        batch_size,
        output_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    vllm_deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (x, x_scale),
        (weight, weight_scale),
        output,
        group_batch_size,
        expected_group_batch_size,
    )
    print(output)


def run_batched_deepgemm_bf16(
    expected_group_batch_size: int,
    num_groups: int,
    output_size: int,
    input_size: int,
):
    weight = torch.randn(
        num_groups,
        output_size,
        input_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    group_batch_size = [
        int(expected_group_batch_size * random.uniform(0.7, 1.3))
        for _ in range(num_groups)
    ]
    batch_size = sum(group_batch_size)
    group_batch_size = torch.tensor(
        group_batch_size,
        dtype=torch.int32,
        device="cuda",
    )
    x = torch.randn(
        num_groups,
        batch_size,
        input_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    ground_truth_output = torch.einsum("bnk, bmk -> bnm", x, weight)
    output = torch.zeros(
        num_groups,
        batch_size,
        output_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    vllm_deep_gemm.bf16_m_grouped_gemm_nt_masked(
        x,
        weight,
        output,
        group_batch_size,
        expected_group_batch_size,
    )
    for i in range(num_groups):
        torch.testing.assert_close(
            output[i, : group_batch_size[i]],
            ground_truth_output[i, : group_batch_size[i]],
        )
        print(
            (
                output[i, : group_batch_size[i]]
                - ground_truth_output[i, : group_batch_size[i]]
            )
            .abs()
            .max()
        )


run_batched_deepgemm_fp8(512, 8, 1024, 512)
run_batched_deepgemm_bf16(512, 8, 1024, 512)
