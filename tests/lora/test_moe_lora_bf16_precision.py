# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that the fused MoE LoRA intermediate buffer uses float32
to prevent precision loss that causes hallucinated output.

This test verifies the fix for the bf16 precision bug in
vllm/lora/ops/triton_ops/fused_moe_lora_op.py where the intermediate
buffer between the shrink (lora_a) and expand (lora_b) kernels was
incorrectly allocated with output.dtype (bf16) instead of float32.
"""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Requires CUDA GPU",
)
def test_fused_moe_lora_intermediate_buffer_is_float32():
    """Verify the intermediate buffer in fused MoE LoRA uses float32.

    The non-MoE LoRA path (punica_gpu.py) explicitly uses float32 for the
    intermediate buffer between shrink and expand operations. The fused MoE
    path should do the same to prevent precision loss that compounds across
    experts and layers, leading to hallucinated outputs in MoE models.
    """
    # Import the function that creates the intermediate buffer
    from vllm.lora.ops.triton_ops.fused_moe_lora_op import _fused_moe_lora

    import inspect
    source = inspect.getsource(_fused_moe_lora)

    # Verify the intermediate buffer uses float32
    assert "dtype=torch.float32" in source, (
        "fused_moe_lora intermediate buffer must use torch.float32 "
        "to match the non-MoE LoRA path and prevent precision loss. "
        "Found dtype=output.dtype which causes bf16 truncation."
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Requires CUDA GPU",
)
def test_fused_moe_lora_kernel_no_hardcoded_bfloat16():
    """Verify the fused MoE LoRA kernel does not hardcode tl.bfloat16.

    The kernel should use the output element type for dot product casting
    rather than hardcoding bfloat16, which would:
    1. Fail to handle fp16 models correctly
    2. Discard precision from the float32 intermediate buffer
    """
    from vllm.lora.ops.triton_ops.fused_moe_lora_op import (
        _fused_moe_lora_kernel,
    )

    import inspect
    source = inspect.getsource(_fused_moe_lora_kernel.fn)

    # The kernel should NOT contain hardcoded tl.bfloat16 casts
    assert "a.to(tl.bfloat16)" not in source, (
        "fused_moe_lora_kernel should not hardcode tl.bfloat16. "
        "Use c_ptr.dtype.element_ty to handle all dtypes correctly."
    )
    assert "b.to(tl.bfloat16)" not in source, (
        "fused_moe_lora_kernel should not hardcode tl.bfloat16. "
        "Use c_ptr.dtype.element_ty to handle all dtypes correctly."
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Requires CUDA GPU",
)
def test_bf16_precision_loss_in_matmul_chain():
    """Demonstrate that bf16 intermediate truncation causes precision loss.

    This test shows the numeric impact of the bug: when a float32 matmul
    result is truncated to bf16 before a second matmul, the final result
    diverges from the float32 reference, especially for small LoRA ranks
    typical in MoE models.
    """
    torch.manual_seed(42)
    device = "cuda"
    num_tokens = 32
    hidden_size = 2880  # gpt_oss hidden size
    rank = 8  # typical LoRA rank

    # Simulate LoRA computation: output = (hidden @ lora_a) @ lora_b
    hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16,
                         device=device)
    lora_a = torch.randn(hidden_size, rank, dtype=torch.bfloat16,
                         device=device) * 0.01
    lora_b = torch.randn(rank, hidden_size, dtype=torch.bfloat16,
                         device=device) * 0.01

    # Reference: full float32 intermediate (correct behavior)
    intermediate_f32 = torch.matmul(hidden.float(), lora_a.float())
    result_f32 = torch.matmul(intermediate_f32, lora_b.float()).bfloat16()

    # Bug path: bf16 intermediate (truncated)
    intermediate_bf16 = torch.matmul(
        hidden.float(), lora_a.float()
    ).bfloat16()
    result_bf16 = torch.matmul(
        intermediate_bf16.float(), lora_b.float()
    ).bfloat16()

    # Compute relative error
    abs_diff = (result_f32.float() - result_bf16.float()).abs()
    rel_error = abs_diff / (result_f32.float().abs() + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    print(f"Max relative error from bf16 intermediate: {max_rel_error:.6f}")
    print(f"Mean relative error from bf16 intermediate: {mean_rel_error:.6f}")

    # The bf16 intermediate should produce measurably different results
    # A typical threshold: relative error > 0.001 (0.1%) indicates
    # non-trivial precision loss
    assert max_rel_error > 0.0, (
        "Expected some precision difference between float32 and bf16 "
        "intermediate paths"
    )
    # Note: for a single layer this might be small, but it compounds
    # across 128 experts x 36 layers in gpt_oss
