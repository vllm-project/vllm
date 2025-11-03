# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

os.environ["VLLM_USE_DEEP_GEMM"] = "1"

import math
import random

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.moe.fused_batched_moe import (
    invoke_moe_batched_triton_kernel,
)
from vllm.model_executor.layers.moe.fused_moe import invoke_fused_moe_kernel
from vllm.triton_utils import tl
from vllm.utils import deep_gemm as vllm_deep_gemm


def generate_bf16_and_downcast_to_fp8(shape, device="cuda"):
    bf16_weight = torch.randn(shape, dtype=torch.bfloat16, device=device)
    fp8_weight = bf16_weight.to(dtype=torch.float8_e4m3fn)
    return fp8_weight


def run_batched_deepgemm_masked_fp8(
    expected_group_batch_size: int,
    num_groups: int,
    output_size: int,
    input_size: int,
):
    BLOCK_SIZE = (128, 128)
    BLOCK_N, BLOCK_K = BLOCK_SIZE
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


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def run_batched_deepgemm_contiguous_bf16(
    expected_group_batch_size: int,
    num_groups: int,
    output_size: int,
    input_size: int,
):
    actual_ms = [
        int(expected_group_batch_size * random.uniform(0.7, 1.3))
        for _ in range(num_groups)
    ]
    # Magic number in deepseek pacakge
    # TODO(zhuohan): change it to real deepseek number
    MK_ALIGNMENT_FOR_CONTIGUOUS_LAYOUT = 128
    aligned_ms = [
        align(actual_m, MK_ALIGNMENT_FOR_CONTIGUOUS_LAYOUT) for actual_m in actual_ms
    ]
    batch_size = sum(aligned_ms)

    weight = torch.randn(
        num_groups,
        output_size,
        input_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = torch.randn(
        batch_size,
        input_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expert_ids = torch.zeros(
        batch_size,
        dtype=torch.int32,
        device="cuda",
    )
    reference_output = torch.zeros(
        batch_size,
        output_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    start = 0
    for i in range(num_groups):
        actual_end = start + actual_ms[i]
        aligned_end = start + aligned_ms[i]
        expert_ids[start:actual_end] = i
        expert_ids[actual_end:aligned_end] = -1
        reference_output[start:actual_end] = x[start:actual_end] @ weight[i].t()
        start = aligned_end

    output = torch.zeros(
        batch_size,
        output_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    vllm_deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        x,
        weight,
        output,
        expert_ids,
    )
    output = output * (expert_ids != -1).unsqueeze(1)
    torch.testing.assert_close(output, reference_output)


def run_batched_deepgemm_masked_bf16(
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
    ground_truth_output = torch.einsum("gmk, gnk -> gmn", x, weight)
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
        # print(
        #     (
        #         output[i, : group_batch_size[i]]
        #         - ground_truth_output[i, : group_batch_size[i]]
        #     )
        #     .abs()
        #     .max()
        # )


def run_triton_group_gemm_contiguous_bf16(
    batch_size: int,
    num_groups: int,
    output_size: int,
    input_size: int,
    topk: int = 4,
):
    weight = torch.randn(
        num_groups,
        output_size,
        input_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = torch.randn(
        batch_size,
        input_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_ids = torch.randint(
        num_groups,
        (batch_size, topk),
        dtype=torch.int32,
        device="cuda",
    )
    reference_output = torch.einsum("mk, mtnk -> mtn", x, weight[topk_ids])
    output = torch.zeros(
        batch_size,
        topk,
        output_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_groups, expert_map=None
    )
    invoke_fused_moe_kernel(
        A=x,
        B=weight,
        C=output,
        A_scale=None,
        B_scale=None,
        B_zp=None,
        topk_weights=None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=topk,
        config=config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )
    torch.testing.assert_close(output, reference_output)


def run_triton_group_gemm_masked_bf16(
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
    ground_truth_output = torch.einsum("gmk, gnk -> gmn", x, weight)
    output = torch.zeros(
        num_groups,
        batch_size,
        output_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    invoke_moe_batched_triton_kernel(
        A=x,
        B=weight,
        C=output,
        expert_num_tokens=group_batch_size,
        compute_type=tl.bfloat16,
        A_scale=None,
        B_scale=None,
        B_zp=None,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        config=config,
        per_act_token_quant=False,
    )
    for i in range(num_groups):
        torch.testing.assert_close(
            output[i, : group_batch_size[i]],
            ground_truth_output[i, : group_batch_size[i]],
        )


# run_batched_deepgemm_masked_fp8(512, 8, 1024, 512)
run_batched_deepgemm_contiguous_bf16(512, 8, 1024, 512)
# run_batched_deepgemm_masked_bf16(512, 8, 1024, 512)
# run_triton_group_gemm_contiguous_bf16(512, 8, 1024, 512, 4)
# run_triton_group_gemm_masked_bf16(512, 8, 1024, 512)
