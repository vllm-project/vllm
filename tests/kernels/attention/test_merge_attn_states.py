# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm._custom_ops import merge_attn_states as merge_attn_states_cuda
from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_merge_attn_states import (
    merge_attn_states as merge_attn_states_triton,
)


# Naive PyTorch Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def merge_attn_states_torch(
    output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse: torch.Tensor,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse: torch.Tensor,  # [NUM_HEADS, NUM_TOKENS]
    output_lse: torch.Tensor | None = None,  # [NUM_HEADS, NUM_TOKENS]
):
    p_lse = prefix_lse
    s_lse = suffix_lse
    # inf -> -inf
    p_lse[p_lse == torch.inf] = -torch.inf
    s_lse[s_lse == torch.inf] = -torch.inf
    # max_lse [NUM_HEADS, NUM_TOKENS]
    max_lse = torch.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_lse_exp = torch.exp(p_lse)
    s_lse_exp = torch.exp(s_lse)
    out_se = p_lse_exp + s_lse_exp
    if output_lse is not None:
        output_lse = torch.log(out_se) + max_lse
    p_scale = p_lse_exp / out_se  # [NUM_HEADS, NUM_TOKENS]
    s_scale = s_lse_exp / out_se  # [NUM_HEADS, NUM_TOKENS]
    p_scale = torch.transpose(p_scale, 0, 1).unsqueeze(2)  # [NUM_TOKENS, NUM_HEADS, 1]
    s_scale = torch.transpose(s_scale, 0, 1).unsqueeze(2)  # [NUM_TOKENS, NUM_HEADS, 1]
    output = prefix_output * p_scale + suffix_output * s_scale
    return output, output_lse


NUM_BATCH_TOKENS = [256, 512, 613, 1024, 1536, 4096]
NUM_QUERY_HEADS = [4, 8, 16, 32, 48, 64]
HEAD_SIZES = [32, 48, 64, 96, 128, 256]
DTYPES = [torch.float32, torch.half, torch.bfloat16]

all_case_info: list[tuple] = []


def generate_markdown_table():
    global all_case_info
    table_header = (
        "| tokens | heads | headsize | dtype "
        "| device | torch | triton | cuda | speedup |"
    )
    table_separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"

    def shortly_dtype(dtype: torch.dtype) -> str:
        return str(dtype).removeprefix("torch.")

    def shortly_device(device: str) -> str:
        return device.removeprefix("NVIDIA").strip()

    print(table_header)
    print(table_separator)
    for info in all_case_info:
        (
            num_tokens,
            num_heads,
            head_size,
            dtype,
            device,
            avg_time_torch_kernel,
            avg_time_triton_kernel,
            avg_time_cuda_kernel,
            performance_improved,
        ) = info
        dtype = shortly_dtype(dtype)
        device = shortly_device(device)
        print(
            f"| {num_tokens} | {num_heads} | {head_size} "
            f"| {dtype} | {device} | {avg_time_torch_kernel:.5f}ms "
            f"| {avg_time_triton_kernel:.5f}ms "
            f"| {avg_time_cuda_kernel:.5f}ms "
            f"| {performance_improved:.4f}x |"
        )


@pytest.mark.parametrize("num_tokens", NUM_BATCH_TOKENS)
@pytest.mark.parametrize("num_query_heads", NUM_QUERY_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("output_dtype", DTYPES)
@torch.inference_mode()
def test_merge_attn_states(
    num_tokens: int, num_query_heads: int, head_size: int, output_dtype: torch.dtype
):
    if not current_platform.is_cuda():
        pytest.skip(
            "Currently only support compare triton merge_attn_states "
            "with custom cuda merge_attn_states kernel"
        )

    NUM_TOKENS = num_tokens
    NUM_HEADS = num_query_heads
    HEAD_SIZE = head_size

    print(
        f"\nNUM_TOKENS:{NUM_TOKENS}, NUM_HEADS:{NUM_HEADS}, "
        f"HEAD_SIZE:{HEAD_SIZE}, DTYPE: {output_dtype}, "
        f"Device: {current_platform.get_device_name()}"
    )

    # prefix_lse and suffix_lse contain inf and normal values
    prefix_lse = torch.randn(NUM_HEADS, NUM_TOKENS, dtype=torch.float32, device="cuda")
    suffix_lse = torch.randn(NUM_HEADS, NUM_TOKENS, dtype=torch.float32, device="cuda")

    # Generate boolean masks
    mask_prefix = torch.rand(NUM_HEADS, NUM_TOKENS) < 0.1
    mask_suffix = torch.rand(NUM_HEADS, NUM_TOKENS) < 0.1
    # Ensure that the same position is not True at the same time
    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)

    prefix_lse[mask_prefix] = float("inf")
    suffix_lse[mask_suffix] = float("inf")

    # Other input tensors (need to be initialized but
    # no actual calculation needed)
    output = torch.zeros(
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE), dtype=output_dtype, device="cuda"
    )
    output_lse = torch.zeros(
        (NUM_HEADS, NUM_TOKENS), dtype=torch.float32, device="cuda"
    )
    prefix_output = torch.randn(
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE), dtype=output_dtype, device="cuda"
    )
    suffix_output = torch.randn(
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE), dtype=output_dtype, device="cuda"
    )

    warmup_times = 2
    repeat_times = 20

    output_torch = output.clone()
    output_lse_torch = output_lse.clone()
    total_time_torch_kernel = 0
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)

    # 0. Run the Torch kernel
    prefix_lse_torch = prefix_lse.clone()
    suffix_lse_torch = suffix_lse.clone()
    for _ in range(warmup_times):
        output_torch, output_lse_torch = merge_attn_states_torch(
            output_torch,
            prefix_output,
            prefix_lse_torch,
            suffix_output,
            suffix_lse_torch,
            output_lse_torch,
        )
    torch.cuda.synchronize()

    for _ in range(repeat_times):
        start.record()
        output_torch, output_lse_torch = merge_attn_states_torch(
            output_torch,
            prefix_output,
            prefix_lse_torch,
            suffix_output,
            suffix_lse_torch,
            output_lse_torch,
        )
        end.record()
        torch.cuda.synchronize()
        total_time_torch_kernel += start.elapsed_time(end)

    avg_time_torch_kernel = total_time_torch_kernel / repeat_times

    # 1. Run the Triton kernel
    output_ref_triton = output.clone()
    output_lse_ref_triton = output_lse.clone()

    total_time_triton_kernel = 0
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)

    for _ in range(warmup_times):
        merge_attn_states_triton(
            output_ref_triton,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_ref_triton,
        )
    torch.cuda.synchronize()

    for _ in range(repeat_times):
        start.record()
        merge_attn_states_triton(
            output_ref_triton,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_ref_triton,
        )
        end.record()
        torch.cuda.synchronize()
        total_time_triton_kernel += start.elapsed_time(end)

    avg_time_triton_kernel = total_time_triton_kernel / repeat_times

    # 2. Run the CUDA kernel
    total_time_cuda_kernel = 0
    output_cuda = output.clone()
    output_lse_cuda = output_lse.clone()

    for _ in range(warmup_times):
        merge_attn_states_cuda(
            output_cuda,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_cuda,
        )
    torch.cuda.synchronize()

    for _ in range(repeat_times):
        start.record()
        merge_attn_states_cuda(
            output_cuda,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_cuda,
        )
        end.record()
        torch.cuda.synchronize()
        total_time_cuda_kernel += start.elapsed_time(end)

    avg_time_cuda_kernel = total_time_cuda_kernel / repeat_times

    # 3. Performance compare
    performance_improved = avg_time_triton_kernel / avg_time_cuda_kernel
    print(f" Torch time: {avg_time_torch_kernel:.6f}ms")
    print(f"Triton time: {avg_time_triton_kernel:.6f}ms")
    print(
        f"  CUDA time: {avg_time_cuda_kernel:.6f}ms, "
        f"Performance: {performance_improved:.5f}x"
    )
    print("-" * 100)

    # 4. Correctness compare
    # Liger Kernel: Efficient Triton Kernels for LLM Training
    # https://arxiv.org/pdf/2410.10989, 3.3 Correctness
    # use rtol = 1e-2 for bfloat16.
    rtol = 1e-2 if output_dtype == torch.bfloat16 else 1e-3

    def diff(a: torch.Tensor, b: torch.Tensor):
        max_diff = torch.max(torch.abs(a.float() - b.float()))
        return max_diff

    # Use Triton output as reference because we want to replace
    # the Triton kernel with custom CUDA kernel for merge attn
    # states operation.
    output_ref = output_ref_triton
    output_lse_ref = output_lse_ref_triton
    torch.testing.assert_close(
        output_cuda.float(), output_ref.float(), atol=1e-3, rtol=rtol
    )
    print("Output all match, max abs diff:")
    print(f"(Triton vs Torch) : {diff(output_torch, output_ref)}")
    print(f"  (CUDA vs Torch) : {diff(output_torch, output_cuda)}")
    print(f"  (CUDA vs Triton): {diff(output_ref, output_cuda)}")
    print("-" * 100)

    torch.testing.assert_close(
        output_lse_cuda.float(), output_lse_ref.float(), atol=1e-3, rtol=rtol
    )
    print("Output LSE all match, max abs diff:")
    print(f"(Triton vs Torch) : {diff(output_lse_torch, output_lse_ref)}")
    print(f"  (CUDA vs Torch) : {diff(output_lse_torch, output_lse_cuda)}")
    print(f"  (CUDA vs Triton): {diff(output_lse_ref, output_lse_cuda)}")
    print("-" * 100)

    print(
        "All output values test passed! All inf values "
        "are correctly replaced with -inf."
    )
    print("-" * 100)

    device = current_platform.get_device_name()
    all_case_info.append(
        (
            NUM_TOKENS,
            NUM_HEADS,
            HEAD_SIZE,
            output_dtype,
            device,
            avg_time_torch_kernel,
            avg_time_triton_kernel,
            avg_time_cuda_kernel,
            performance_improved,
        )
    )
    if len(all_case_info) == (
        len(NUM_BATCH_TOKENS) * len(HEAD_SIZES) * len(NUM_QUERY_HEADS) * len(DTYPES)
    ):
        generate_markdown_table()
