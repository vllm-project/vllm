# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused Router Kernel for GPT-OSS MoE.
Fuses the router linear layer (GEMM) and Top-K selection + Softmax.
"""

import torch

from vllm.triton_utils import tl, triton


# Define autotuning configurations
# We tune BLOCK_M and BLOCK_K. BLOCK_N is determined by the number of experts.
# num_stages and num_warps are also tuned for performance.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8),
    ],
    key=["M", "K", "N"],
)
@triton.jit
def fused_moe_router_kernel(
    # Pointers
    x_ptr,  # Input [M, K]
    w_ptr,  # Weight [N, K]
    bias_ptr,  # Bias [N] (Optional)
    out_w_ptr,  # Output Weights [M, TopK]
    out_i_ptr,  # Output Indices [M, TopK]
    # Dimensions
    M,
    K,
    N,
    TopK: tl.constexpr,
    # Strides
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_biasn,  # Bias stride
    stride_wm,
    stride_wk_out,  # output weights stride
    stride_im,
    stride_ik_out,  # output indices stride
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Must be >= N (number of experts)
    HAS_BIAS: tl.constexpr,
):
    # 1. Program ID
    pid = tl.program_id(axis=0)

    # 2. Create offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # 3. Initialize accumulator for GEMM (Logits)
    # Accumulator shape: [BLOCK_M, BLOCK_N]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 4. GEMM Loop over K dimension
    # We calculate X [M, K] @ W.T [K, N] + Bias [N]
    for k in range(0, K, BLOCK_K):
        # Load Input X [BLOCK_M, BLOCK_K]
        offs_k = k + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load Weight W tile.
        # The weight matrix in PyTorch is typically stored as [Out, In], i.e., [N, K].
        # To perform X @ W^T, we need to access it as [K, N] conceptually.
        # We load a tile of shape [BLOCK_N, BLOCK_K] directly from memory and
        # transpose it implicitly via the dot product structure or explicit
        # transpositions if needed.
        # Here we load [N, K] using strides to match dimensions.
        w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x, w)

    # 5. Add Bias (Optional)
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n * stride_biasn
        bias_mask = offs_n < N
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
        # Broadcast bias [BLOCK_N] to [BLOCK_M, BLOCK_N]
        acc += bias[None, :]

    # 6. Top-K Selection in SRAM
    # acc now contains the logits [BLOCK_M, BLOCK_N]
    logits = acc

    # Mask out invalid experts (if BLOCK_N > N) to -inf so they are not selected
    if BLOCK_N > N:
        logits = tl.where(tl.arange(0, BLOCK_N)[None, :] < N, logits, float("-inf"))

    # Storage for TopK results
    topk_val_storage = tl.zeros([BLOCK_M, TopK], dtype=tl.float32)
    topk_idx_storage = tl.zeros([BLOCK_M, TopK], dtype=tl.int32)

    for i in range(TopK):
        # Find max along the expert dimension
        val_max, idx_max = tl.max(logits, axis=1, return_indices=True)

        # Store current max
        topk_val_storage[:, i] = val_max
        topk_idx_storage[:, i] = idx_max

        # Mask out the selected expert to find the next max in next iteration
        mask = tl.arange(0, BLOCK_N)[None, :] == idx_max[:, None]
        logits = tl.where(mask, float("-inf"), logits)

    # 7. Softmax Renormalization
    # Subtract max for numerical stability
    val_max_for_softmax = tl.max(topk_val_storage, axis=1)
    numerator = tl.exp(topk_val_storage - val_max_for_softmax[:, None])
    denominator = tl.sum(numerator, axis=1)
    softmax_res = numerator / denominator[:, None]

    # 8. Write Output
    output_mask = offs_m[:, None] < M

    offs_topk = tl.arange(0, TopK)
    out_w_ptrs = out_w_ptr + (
        offs_m[:, None] * stride_wm + offs_topk[None, :] * stride_wk_out
    )
    out_i_ptrs = out_i_ptr + (
        offs_m[:, None] * stride_im + offs_topk[None, :] * stride_ik_out
    )

    tl.store(out_w_ptrs, softmax_res, mask=output_mask)
    tl.store(out_i_ptrs, topk_idx_storage, mask=output_mask)


def fused_router(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    router_bias: torch.Tensor | None,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        hidden_states: [num_tokens, hidden_size]
        router_weights: [num_experts, hidden_size]
        router_bias: [num_experts] or None
        top_k: int
    Returns:
        topk_weights: [num_tokens, top_k] (after softmax)
        topk_indices: [num_tokens, top_k]
    """
    assert hidden_states.ndim == 2
    assert router_weights.ndim == 2

    M, K = hidden_states.shape
    N, _ = router_weights.shape

    # Outputs
    topk_weights = torch.empty(
        (M, top_k), device=hidden_states.device, dtype=torch.float32
    )
    topk_indices = torch.empty(
        (M, top_k), device=hidden_states.device, dtype=torch.int32
    )

    # Determine BLOCK_N based on number of experts
    # Must be power of 2 and >= N for the kernel logic
    BLOCK_N = triton.next_power_of_2(N)

    # Handle Bias
    if router_bias is not None:
        assert router_bias.shape[0] == N
        bias_ptr = router_bias
        stride_biasn = router_bias.stride(0)
    else:
        bias_ptr = None  # Triton handles None ptrs fine if unused, but we need a value
        # We pass hidden_states as a dummy pointer if bias is None to avoid errors,
        # but the kernel guard HAS_BIAS will prevent access.
        bias_ptr = hidden_states
        stride_biasn = 0

    # Grid: Only need to tile along M dimension
    # BLOCK_M is handled by autotuner, so we pass a callable for grid size
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]), 1, 1)

    fused_moe_router_kernel[grid](
        hidden_states,
        router_weights,
        bias_ptr,
        topk_weights,
        topk_indices,
        M,
        K,
        N,
        TopK=top_k,
        stride_xm=hidden_states.stride(0),
        stride_xk=hidden_states.stride(1),
        stride_wn=router_weights.stride(0),
        stride_wk=router_weights.stride(1),
        stride_biasn=stride_biasn,
        stride_wm=topk_weights.stride(0),
        stride_wk_out=topk_weights.stride(1),
        stride_im=topk_indices.stride(0),
        stride_ik_out=topk_indices.stride(1),
        BLOCK_N=BLOCK_N,
        HAS_BIAS=router_bias is not None,
    )

    return topk_weights, topk_indices
