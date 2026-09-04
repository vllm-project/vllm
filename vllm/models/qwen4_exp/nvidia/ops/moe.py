# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp MoE tail kernels."""

import torch

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _finalize_moe_with_shared_kernel(
    gemm2_ptr,
    expert_weights_ptr,
    expanded_idx_ptr,
    shared_ptr,
    shared_gate_ptr,
    output_ptr,
    stride_gemm2,
    stride_weights,
    stride_indices,
    stride_shared,
    stride_shared_gate,
    stride_output,
    HIDDEN_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = offsets < HIDDEN_SIZE

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    routed = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for slot in tl.static_range(TOP_K):
        permuted_idx = tl.load(expanded_idx_ptr + token * stride_indices + slot)
        route_mask = hidden_mask & (permuted_idx >= 0)
        values = tl.load(
            gemm2_ptr + permuted_idx * stride_gemm2 + offsets,
            mask=route_mask,
            other=0.0,
        )
        weight = tl.load(expert_weights_ptr + token * stride_weights + slot)
        routed += values.to(tl.float32) * weight.to(tl.float32)

    # Preserve the original FlashInfer finalize -> BF16 add boundary.
    routed = routed.to(output_ptr.dtype.element_ty)
    shared = tl.load(
        shared_ptr + token * stride_shared + offsets,
        mask=hidden_mask,
        other=0.0,
    )
    if shared_gate_ptr is not None:
        shared_gate = tl.load(shared_gate_ptr + token * stride_shared_gate)
        shared_gate = tl.sigmoid(shared_gate.to(tl.float32)).to(
            shared_ptr.dtype.element_ty
        )
        shared = (shared.to(tl.float32) * shared_gate.to(tl.float32)).to(
            shared_ptr.dtype.element_ty
        )
    output = routed.to(tl.float32) + shared.to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        output_ptr + token * stride_output + offsets,
        output,
        mask=hidden_mask,
    )


def _finalize_moe_with_shared(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx: torch.Tensor,
    shared_output: torch.Tensor,
    shared_gate_logits: torch.Tensor | None,
    top_k: int,
) -> torch.Tensor:
    num_tokens, hidden_size = shared_output.shape
    assert gemm2_permuted.shape[1] == hidden_size
    assert expert_weights.shape == (num_tokens, top_k)
    assert expanded_idx.shape == (num_tokens, top_k)
    assert gemm2_permuted.dtype == shared_output.dtype == torch.bfloat16
    assert expert_weights.dtype == torch.bfloat16
    assert expanded_idx.dtype == torch.int32
    if shared_gate_logits is not None:
        assert shared_gate_logits.shape == (num_tokens, 1)
        assert shared_gate_logits.dtype == shared_output.dtype
        assert shared_gate_logits.is_contiguous()
    assert all(
        tensor.is_contiguous()
        for tensor in (
            gemm2_permuted,
            expert_weights,
            expanded_idx,
            shared_output,
        )
    )

    output = torch.empty_like(shared_output)
    block_size = 512
    _finalize_moe_with_shared_kernel[
        (num_tokens, triton.cdiv(hidden_size, block_size))
    ](
        gemm2_permuted,
        expert_weights,
        expanded_idx,
        shared_output,
        shared_gate_logits,
        output,
        gemm2_permuted.stride(0),
        expert_weights.stride(0),
        expanded_idx.stride(0),
        shared_output.stride(0),
        shared_gate_logits.stride(0) if shared_gate_logits is not None else 0,
        output.stride(0),
        HIDDEN_SIZE=hidden_size,
        TOP_K=top_k,
        BLOCK_SIZE=block_size,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return output


def _finalize_moe_with_shared_fake(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx: torch.Tensor,
    shared_output: torch.Tensor,
    shared_gate_logits: torch.Tensor | None,
    top_k: int,
) -> torch.Tensor:
    del gemm2_permuted, expert_weights, expanded_idx, shared_gate_logits, top_k
    return torch.empty_like(shared_output)


direct_register_custom_op(
    op_name="qwen4_exp_finalize_moe_with_shared",
    op_func=_finalize_moe_with_shared,
    fake_impl=_finalize_moe_with_shared_fake,
)


def finalize_moe_with_shared(
    routed_output: UnfinalizedMoEOutput,
    shared_output: torch.Tensor,
    shared_gate_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_finalize_moe_with_shared(
        routed_output.gemm2_permuted,
        routed_output.expert_weights,
        routed_output.expanded_idx_to_permuted_idx,
        shared_output,
        shared_gate_logits,
        routed_output.expert_weights.shape[1],
    )


__all__ = ["finalize_moe_with_shared"]
