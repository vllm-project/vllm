# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm._custom_ops import (
    merge_attn_states as merge_attn_states_cuda,
)
from vllm._custom_ops import (
    scaled_fp8_quant,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
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
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,  # scalar, per-tensor FP8 scale
):
    # Apply prefill_tokens_with_context mask if needed
    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = output.shape[0]
    p_lse = prefix_lse
    s_lse = suffix_lse
    # inf -> -inf
    p_lse[p_lse == torch.inf] = -torch.inf
    s_lse[s_lse == torch.inf] = -torch.inf
    # max_lse [NUM_HEADS, NUM_TOKENS]
    max_lse = torch.maximum(p_lse, s_lse)

    mask = torch.ones((prefix_lse.shape[1], 1, 1), device=p_lse.device)
    mask[prefill_tokens_with_context:].fill_(0)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_lse_exp = torch.exp(p_lse)
    s_lse_exp = torch.exp(s_lse)
    out_se = p_lse_exp + s_lse_exp
    if output_lse is not None:
        output_lse = torch.log(out_se) + max_lse
        output_lse[prefill_tokens_with_context:] = suffix_lse[
            prefill_tokens_with_context:
        ]
    p_scale = p_lse_exp / out_se  # [NUM_HEADS, NUM_TOKENS]
    s_scale = s_lse_exp / out_se  # [NUM_HEADS, NUM_TOKENS]
    p_scale = torch.transpose(p_scale, 0, 1).unsqueeze(2)  # [NUM_TOKENS, NUM_HEADS, 1]
    s_scale = torch.transpose(s_scale, 0, 1).unsqueeze(2)  # [NUM_TOKENS, NUM_HEADS, 1]
    output = prefix_output * p_scale * mask + suffix_output * (
        s_scale * mask + (1 - mask)
    )
    if output_scale is not None:
        shape = output.shape
        output, _ = scaled_fp8_quant(output.float().view(-1, shape[-1]), output_scale)
        output = output.view(shape)
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


@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("prefill_tokens_with_context", [None, 128])
@pytest.mark.parametrize("num_tokens", NUM_BATCH_TOKENS)
@pytest.mark.parametrize("num_query_heads", NUM_QUERY_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("input_dtype", DTYPES)
@torch.inference_mode()
def test_merge_attn_states(
    prefill_tokens_with_context: int | None,
    num_tokens: int,
    num_query_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    use_fp8: bool,
):
    if not current_platform.is_cuda():
        pytest.skip(
            "Currently only support compare triton merge_attn_states "
            "with custom cuda merge_attn_states kernel"
        )

    NUM_TOKENS = num_tokens
    NUM_HEADS = num_query_heads
    HEAD_SIZE = head_size

    # When use_fp8 is set, inputs stay as input_dtype (bf16/fp16/fp32)
    # and output becomes FP8.
    output_dtype = input_dtype
    output_scale = None
    if use_fp8:
        output_dtype = current_platform.fp8_dtype()
        output_scale = torch.tensor([0.05], dtype=torch.float32, device="cuda")

    print(
        f"\nNUM_TOKENS:{NUM_TOKENS}, NUM_HEADS:{NUM_HEADS}, "
        f"HEAD_SIZE:{HEAD_SIZE}, input_dtype: {input_dtype}, "
        f"output_dtype: {output_dtype}, use_fp8: {use_fp8}, "
        f"prefill_tokens_with_context: {prefill_tokens_with_context}, "
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
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE), dtype=input_dtype, device="cuda"
    )
    suffix_output = torch.randn(
        (NUM_TOKENS, NUM_HEADS, HEAD_SIZE), dtype=input_dtype, device="cuda"
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
            prefill_tokens_with_context,
            output_scale,
        )
    torch.accelerator.synchronize()

    for _ in range(repeat_times):
        start.record()
        output_torch, output_lse_torch = merge_attn_states_torch(
            output_torch,
            prefix_output,
            prefix_lse_torch,
            suffix_output,
            suffix_lse_torch,
            output_lse_torch,
            prefill_tokens_with_context,
            output_scale,
        )
        end.record()
        torch.accelerator.synchronize()
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
            prefill_tokens_with_context,
            output_scale,
        )
    torch.accelerator.synchronize()

    for _ in range(repeat_times):
        start.record()
        merge_attn_states_triton(
            output_ref_triton,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_ref_triton,
            prefill_tokens_with_context,
            output_scale,
        )
        end.record()
        torch.accelerator.synchronize()
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
            prefill_tokens_with_context,
            output_scale,
        )
    torch.accelerator.synchronize()

    for _ in range(repeat_times):
        start.record()
        merge_attn_states_cuda(
            output_cuda,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_cuda,
            prefill_tokens_with_context,
            output_scale,
        )
        end.record()
        torch.accelerator.synchronize()
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
    if use_fp8:
        # Compare in dequantized space (multiply back by scale) so that
        # absolute differences reflect real precision, not amplified FP8
        # quantization steps.
        atol, rtol = 1e-1, 1e-1
        assert output_scale is not None
        scale = output_scale.item()
    elif output_dtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
        scale = 1.0
    else:
        atol, rtol = 1e-3, 1e-3
        scale = 1.0

    def diff(a: torch.Tensor, b: torch.Tensor):
        max_diff = torch.max(torch.abs(a.float() - b.float()))
        return max_diff

    # Use Triton output as reference because we want to replace
    # the Triton kernel with custom CUDA kernel for merge attn
    # states operation.
    output_ref = output_ref_triton
    output_lse_ref = output_lse_ref_triton
    torch.testing.assert_close(
        output_cuda.float() * scale,
        output_ref.float() * scale,
        atol=atol,
        rtol=rtol,
    )
    print(
        "Output all match, max abs diff (dequantized):"
        if use_fp8
        else "Output all match, max abs diff:"
    )
    _diff = diff(output_ref.float() * scale, output_torch.float() * scale)
    print(f"(Triton vs Torch) : {_diff}")
    _diff = diff(output_torch.float() * scale, output_cuda.float() * scale)
    print(f"  (CUDA vs Torch) : {_diff}")
    _diff = diff(output_ref.float() * scale, output_cuda.float() * scale)
    print(f"  (CUDA vs Triton): {_diff}")
    print("-" * 100)

    torch.testing.assert_close(
        output_lse_cuda.float(), output_lse_ref.float(), atol=atol, rtol=rtol
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


# Per-token-per-group FP8 merge tests. Reference: bf16 merge →
# per_token_group_quant_fp8. Compares CUDA + Triton fused merge kernels in
# dequantized space.
GROUP_FP8_TOKENS = [16, 128, 257, 512]
GROUP_FP8_HEADS = [16, 64, 128]
GROUP_FP8_HEAD_SIZES = [128]
GROUP_FP8_GROUP_SIZES = [64, 128]
GROUP_FP8_INPUT_DTYPES = [torch.bfloat16, torch.float16]


def _scale_layout_kwargs(layout: str) -> dict:
    if layout == "row_major":
        return {"column_major_scales": False, "tma_aligned_scales": False}
    if layout == "col_major":
        return {"column_major_scales": True, "tma_aligned_scales": False}
    if layout == "col_major_tma":
        return {"column_major_scales": True, "tma_aligned_scales": True}
    raise ValueError(layout)


def _bf16_merge_reference(
    prefix_output, prefix_lse, suffix_output, suffix_lse, prefill_with_context
):
    output = torch.empty_like(prefix_output)
    lse = torch.empty(
        (prefix_output.shape[1], prefix_output.shape[0]),
        device=prefix_output.device,
        dtype=torch.float32,
    )
    output, lse = merge_attn_states_torch(
        output,
        prefix_output.clone(),
        prefix_lse.clone(),
        suffix_output.clone(),
        suffix_lse.clone(),
        lse,
        prefill_with_context,
        None,
    )
    return output, lse


def _broadcast_scales(
    sf: torch.Tensor, num_heads: int, head_size: int, group_size: int
) -> torch.Tensor:
    num_tokens = sf.shape[0]
    num_groups = num_heads * head_size // group_size
    sf_rm = sf.contiguous().float().reshape(num_tokens, num_groups)
    sf_full = sf_rm.repeat_interleave(group_size, dim=-1)
    return sf_full.view(num_tokens, num_heads, head_size)


@pytest.mark.parametrize("kernel", ["cuda", "triton"], ids=["cuda", "triton"])
@pytest.mark.parametrize("scale_layout", ["row_major", "col_major", "col_major_tma"])
@pytest.mark.parametrize("use_ue8m0", [False, True])
@pytest.mark.parametrize("group_size", GROUP_FP8_GROUP_SIZES)
@pytest.mark.parametrize("head_size", GROUP_FP8_HEAD_SIZES)
@pytest.mark.parametrize("num_heads", GROUP_FP8_HEADS)
@pytest.mark.parametrize("num_tokens", GROUP_FP8_TOKENS)
@pytest.mark.parametrize("input_dtype", GROUP_FP8_INPUT_DTYPES)
@torch.inference_mode()
def test_merge_attn_states_group_fp8(
    kernel: str,
    scale_layout: str,
    use_ue8m0: bool,
    group_size: int,
    head_size: int,
    num_heads: int,
    num_tokens: int,
    input_dtype: torch.dtype,
):
    if not current_platform.is_cuda():
        pytest.skip("group-FP8 merge fusion is CUDA-only")

    fp8_dtype = current_platform.fp8_dtype()
    prefill_with_context = num_tokens // 2

    prefix_output = torch.randn(
        num_tokens, num_heads, head_size, dtype=input_dtype, device="cuda"
    )
    suffix_output = torch.randn(
        num_tokens, num_heads, head_size, dtype=input_dtype, device="cuda"
    )
    prefix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device="cuda")
    suffix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device="cuda")

    bf16_ref, _ = _bf16_merge_reference(
        prefix_output, prefix_lse, suffix_output, suffix_lse, prefill_with_context
    )
    flat_ref = bf16_ref.reshape(num_tokens, num_heads * head_size)
    ref_q, ref_s = per_token_group_quant_fp8(
        flat_ref,
        group_size,
        dtype=fp8_dtype,
        use_ue8m0=use_ue8m0,
        **_scale_layout_kwargs(scale_layout),
    )
    ref_q = ref_q.view(num_tokens, num_heads, head_size)

    test_q = torch.empty(
        (num_tokens, num_heads, head_size), dtype=fp8_dtype, device="cuda"
    )
    test_s = torch.empty_strided(
        ref_s.shape, ref_s.stride(), dtype=ref_s.dtype, device=ref_s.device
    )
    test_s.zero_()

    output_lse = torch.empty(
        (num_heads, num_tokens), dtype=torch.float32, device="cuda"
    )

    if kernel == "cuda":
        merge_attn_states_cuda(
            test_q,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_with_context,
            None,  # output_scale
            test_s,
            group_size,
            use_ue8m0,
        )
    else:
        merge_attn_states_triton(
            test_q,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_with_context,
            None,
            test_s,
            group_size,
            use_ue8m0,
        )

    test_deq = test_q.float() * _broadcast_scales(
        test_s, num_heads, head_size, group_size
    )
    ref_deq = ref_q.float() * _broadcast_scales(ref_s, num_heads, head_size, group_size)

    # Both sides FP8-quantize the same merged values; the only source of
    # divergence is ref's extra bf16 round-trip subtly shifting the absmax
    # (and thus the scale) on a small fraction of groups, which can flip a
    # value into an adjacent FP8 bucket. Empirically ~99.98% of FP8 bytes
    # are bit-identical; tolerances cover the boundary cases.
    torch.testing.assert_close(test_deq, ref_deq, atol=2e-1, rtol=2e-1)
