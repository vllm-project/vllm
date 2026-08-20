# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch._subclasses.fake_tensor import FakeTensor

from vllm.triton_utils import tl, triton


@triton.jit
def moe_fused_mul_sum_kernel(
    inputs_ptr,
    topk_weights_ptr,
    outputs_ptr,
    top_ids_ptr,
    expert_map_ptr,
    stride_m,
    has_topk_ids: tl.constexpr,
    has_expert_map: tl.constexpr,
    top_k: tl.constexpr,
    size: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One CTA owns one token's output row. `inputs` is (num_tokens, top_k, size)
    # but only the ~1 slot per token that a local expert wrote is live; looping
    # top_k with a per-slot uniform branch reads just those slots (the branch is
    # block-uniform, so padding slots are skipped, not masked-and-loaded).
    pid_m = tl.program_id(0)

    if has_topk_ids:
        # All-padding rows (every id < 0) are skipped, leaving their output
        # untouched -- the CUDA-graph decode contract for rows past num_recv.
        any_valid = 0
        for n in tl.static_range(top_k):
            idn = tl.load(top_ids_ptr + pid_m * top_k + n)
            any_valid += (idn >= 0).to(tl.int32)
        if any_valid == 0:
            return

    n_tiles: tl.constexpr = (size + BLOCK_K - 1) // BLOCK_K
    a_row = inputs_ptr + pid_m * stride_m
    w_row = topk_weights_ptr + pid_m * top_k
    out_row = outputs_ptr + pid_m * size

    # Static trip count (n_tiles is constexpr) so the loop pipelines.
    for t in tl.range(0, n_tiles):
        offs_k = t * BLOCK_K + tl.arange(0, BLOCK_K)
        kmask = offs_k < size
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for n in tl.static_range(top_k):
            take = True
            if has_topk_ids:
                id_val = tl.load(top_ids_ptr + pid_m * top_k + n)
                take = id_val >= 0
                if has_expert_map:
                    # -1 slots must not index the map; clamp to 0 first.
                    local_id = tl.load(expert_map_ptr + tl.where(take, id_val, 0))
                    take = take & (local_id >= 0)
            if take:
                w = tl.load(w_row + n).to(tl.float32)
                a = tl.load(a_row + n * size + offs_k, mask=kmask, other=0.0)
                acc += a.to(tl.float32) * w
        tl.store(out_row + offs_k, acc.to(outputs_ptr.dtype.element_ty), mask=kmask)


def _heuristic_config(
    size: int,
    element_size: int,
):
    is_fp32 = element_size > 2
    max_block_k = 256 if is_fp32 else 512
    BLOCK_K = max(128, min(triton.next_power_of_2(size), max_block_k))
    num_warps = 4 if is_fp32 else 2
    num_stages = 3
    return BLOCK_K, num_warps, num_stages


def moe_fused_mul_sum(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    outputs: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused kernel for MoE (Mixture of Experts) to perform weighted summation
    of expert outputs.

    Args:
        inputs: The output from experts.
            Shape: (num_tokens, top_k, hidden_size).
        topk_weights: The weights assigned to each expert for each token.
            Shape: (num_tokens, top_k).
        outputs: Optional pre-allocated output tensor.
            Shape: (num_tokens, hidden_size).
        topk_ids: Optional indices of the top-k experts. Shape:
            (num_tokens, top_k). A value of -1 marks a slot the expert GEMM
            skipped; those slots are excluded from the sum. When provided, rows
            with all top ids < 0 (worst-case padding) are skipped and their
            output rows left untouched. Required when `expert_map` is provided.
        expert_map: Optional mapping for Expert Parallelism. A value < 0
            indicates an invalid token/expert pair that will be skipped. Only
            needed when `topk_ids` may contain non-local expert ids; if every
            non-(-1) id is already a local expert, leave it None to skip the
            redundant per-slot lookup.

    Returns:
        The fused weighted sum of expert outputs.
        Shape: (num_tokens, hidden_size).
    """
    assert inputs.ndim == 3
    assert topk_weights.ndim == 2
    assert inputs.is_contiguous()
    assert topk_weights.is_contiguous()
    assert inputs.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert topk_weights.dtype in (torch.float32, torch.float16, torch.bfloat16)

    num_tokens, top_k, size = inputs.shape
    output_shape = (num_tokens, size)
    if outputs is None:
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)

    assert outputs.shape == output_shape
    assert topk_weights.shape == (num_tokens, top_k)
    assert expert_map is None or topk_ids is not None, (
        "topk_ids is required to interpret expert_map"
    )
    if topk_ids is not None:
        assert topk_ids.shape == (num_tokens, top_k)
        assert topk_ids.is_contiguous()
        assert topk_ids.dtype in (torch.int32, torch.int64)

    if not isinstance(inputs, FakeTensor):
        BLOCK_K, num_warps, num_stages = _heuristic_config(
            size,
            inputs.element_size(),
        )
        grid = (num_tokens,)
        moe_fused_mul_sum_kernel[grid](
            inputs,
            topk_weights,
            outputs,
            topk_ids,
            expert_map,
            top_k * size,
            topk_ids is not None,
            expert_map is not None,
            top_k,
            size,
            BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return outputs
