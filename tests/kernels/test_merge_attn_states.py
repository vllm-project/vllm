# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm._custom_ops import merge_attn_states as merge_attn_states_cuda
from vllm.attention.ops.triton_merge_attn_states import (
    merge_attn_states as merge_attn_states_triton)
from vllm.platforms import current_platform

NUM_TOKENS = [256, 512, 613, 1024, 1536, 4096, 8192, 16384]
NUM_QUERY_HEADS = [4, 8, 16, 32, 48, 64]
HEAD_SIZES = [48, 64, 96, 128, 256]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_query_heads", NUM_QUERY_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@torch.inference_mode()
def test_merge_attn_states(num_tokens: int, num_query_heads: int,
                           head_size: int):
    if not current_platform.is_cuda():
        pytest.skip('Currently only support compare triton merge_attn_states '
                    'with custom cuda merge_attn_states kernel')

    # Set test parameters
    NUM_TOKENS = num_tokens
    # Num query heads
    NUM_HEADS = num_query_heads
    # Set HEAD_SIZE to a power of 2 in the test code
    HEAD_SIZE = head_size

    # Generate test inputs
    # prefix_lse and suffix_lse contain inf and normal values
    prefix_lse = torch.randn(NUM_HEADS,
                             NUM_TOKENS,
                             dtype=torch.float32,
                             device="cuda")
    suffix_lse = torch.randn(NUM_HEADS,
                             NUM_TOKENS,
                             dtype=torch.float32,
                             device="cuda")

    # Generate boolean masks
    mask_prefix = torch.rand(NUM_HEADS, NUM_TOKENS) < 0.1
    mask_suffix = torch.rand(NUM_HEADS, NUM_TOKENS) < 0.1
    # Ensure that the same position is not True at the same time
    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)

    prefix_lse[mask_prefix] = float('inf')
    suffix_lse[mask_suffix] = float('inf')

    # Other input tensors (need to be initialized but
    # no actual calculation needed)
    output = torch.zeros((NUM_TOKENS, NUM_HEADS, HEAD_SIZE),
                         dtype=torch.float32,
                         device="cuda")
    output_lse = torch.zeros((NUM_HEADS, NUM_TOKENS),
                             dtype=torch.float32,
                             device="cuda")
    prefix_output = torch.randn((NUM_TOKENS, NUM_HEADS, HEAD_SIZE),
                                dtype=torch.float32,
                                device="cuda")
    suffix_output = torch.randn((NUM_TOKENS, NUM_HEADS, HEAD_SIZE),
                                dtype=torch.float32,
                                device="cuda")

    output_ref = output.clone()
    output_lse_ref = output_lse.clone()

    # Run the Triton kernel
    # Warmup and measure performance of merge_attn_states_triton
    warmup_times = 2
    repeat_times = 20
    total_time_kernel = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup_times):
        merge_attn_states_triton(output_ref, prefix_output, prefix_lse,
                                 suffix_output, suffix_lse, output_lse_ref)
    torch.cuda.synchronize()

    # Repeat and measure
    for _ in range(repeat_times):
        start.record()
        merge_attn_states_triton(output_ref, prefix_output, prefix_lse,
                                 suffix_output, suffix_lse, output_lse_ref)
        end.record()
        torch.cuda.synchronize()
        total_time_kernel += start.elapsed_time(end)

    avg_time_kernel = total_time_kernel / repeat_times

    # Warmup and measure performance of merge_attn_states_cuda
    total_time_kernel_cuda = 0
    output_cuda = output.clone()
    output_lse_cuda = output_lse.clone()

    # Warmup
    for _ in range(warmup_times):
        merge_attn_states_cuda(output_cuda, prefix_output, prefix_lse,
                               suffix_output, suffix_lse, output_lse_cuda)
    torch.cuda.synchronize()

    # Repeat and measure
    for _ in range(repeat_times):
        start.record()
        merge_attn_states_cuda(output_cuda, prefix_output, prefix_lse,
                               suffix_output, suffix_lse, output_lse_cuda)
        end.record()
        torch.cuda.synchronize()
        total_time_kernel_cuda += start.elapsed_time(end)

    avg_time_kernel_cuda = total_time_kernel_cuda / repeat_times
    performance_improved = avg_time_kernel / avg_time_kernel_cuda
    print(f"\nNUM_TOKENS:{NUM_TOKENS}, NUM_HEADS:{NUM_HEADS}, "
          f"HEAD_SIZE:{HEAD_SIZE}, "
          f"Device: {current_platform.get_device_name()}")
    print(f"Average time taken by Triton merge_attn_states "
          f"kernel: {avg_time_kernel}ms")
    print(f"Average time taken by   CUDA merge_attn_states "
          f"kernel: {avg_time_kernel_cuda}ms, "
          f"Performance: {performance_improved:.5f}x")
    print("-" * 100)

    if not torch.allclose(output_ref, output_cuda, rtol=1e-3, atol=1e-3):
        diff = torch.abs(output_ref - output_cuda)
        max_diff = torch.max(diff)
        max_diff_index = torch.argmax(diff)
        print(f"Max difference in output: {max_diff} "
              f"at index {max_diff_index}")
        print(f"Triton output at max diff index: "
              f"{output_ref.flatten()[max_diff_index]}")
        print(f"CUDA output at max diff index: "
              f"{output_cuda.flatten()[max_diff_index]}")
    assert torch.allclose(output_ref, output_cuda, rtol=1e-3,
                          atol=1e-3), \
           "Output of Triton and CUDA do not match."
    print("Output of Triton and CUDA all match.")

    if not torch.allclose(
            output_lse_ref, output_lse_cuda, rtol=1e-3, atol=1e-3):
        diff = torch.abs(output_lse_ref - output_lse_cuda)
        max_diff = torch.max(diff)
        max_diff_index = torch.argmax(diff)
        print(f"Max difference in output_lse: "
              f"{max_diff} at index {max_diff_index}")
        print(f"Triton output_lse at max diff index: "
              f"{output_lse_ref.flatten()[max_diff_index]}")
        print(f"CUDA output_lse at max diff index: "
              f"{output_lse_cuda.flatten()[max_diff_index]}")
    assert torch.allclose(
        output_lse_ref, output_lse_cuda, rtol=1e-3,
        atol=1e-3), "Output_lse of Triton and CUDA do not match."
    print("Output LSE of Triton and CUDA all match.")

    print("All output values test passed! All inf values "
          "are correctly replaced with -inf.")
    print("-" * 100)
