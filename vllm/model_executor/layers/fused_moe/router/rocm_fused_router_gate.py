# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused router gate for gfx950: gate GEMM plus sqrtsoftplus expert selection.

A split-K BF16 matrix multiplication reuses gate weights across token rows.
The second kernel fuses the FP32 partial-sum reduction, sqrtsoftplus scoring,
and deterministic expert selection.
"""

import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.triton_utils import tl, triton

# (hidden_size, num_experts): DeepSeek-V4.1-Flash.
ROCM_FUSED_ROUTER_GATE_SUPPORTED_SHAPES = frozenset({(7168, 384)})
_MAX_TOKENS = 1536


@triton.jit
def _router_gate_gemm(
    hidden_states_ptr,
    router_weight_ptr,
    partial_logits_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    experts = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(2)
    offsets_k = split * CHUNK_K + tl.arange(0, BLOCK_K)
    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(tl.cdiv(CHUNK_K, BLOCK_K)):
        hidden = tl.load(
            hidden_states_ptr + rows[:, None] * K + offsets_k[None, :],
            mask=(rows[:, None] < M) & (offsets_k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            router_weight_ptr + experts[None, :] * K + offsets_k[:, None],
            mask=(experts[None, :] < N) & (offsets_k[:, None] < K),
            other=0.0,
        )
        logits = tl.dot(hidden, weight, logits)
        offsets_k += BLOCK_K
    tl.store(
        partial_logits_ptr + split * M * N + rows[:, None] * N + experts[None, :],
        logits,
        mask=(rows[:, None] < M) & (experts[None, :] < N),
    )


@triton.jit
def _router_gate_reduce_topk(
    partial_logits_ptr,
    correction_bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    routed_scaling_factor,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    experts = tl.arange(0, BLOCK_N)
    splits = tl.arange(0, BLOCK_S)
    partial = tl.load(
        partial_logits_ptr + splits[:, None] * M * N + row * N + experts[None, :],
        mask=(splits[:, None] < SPLIT_K) & (experts[None, :] < N),
        other=0.0,
    )
    logits = tl.sum(partial, axis=0)
    scores = tl.sqrt(
        tl.maximum(logits, 0.0) + tl.extra.libdevice.log1p(tl.exp(-tl.abs(logits)))
    )
    ranked = scores
    if HAS_BIAS:
        ranked += tl.load(correction_bias_ptr + experts, mask=experts < N, other=0.0)
    ranked = tl.where(experts < N, ranked, -float("inf"))
    ranked = tl.where(ranked == 0.0, 0.0, ranked)

    # Sort the full FP32 score, then the inverse expert id for exact ties.
    bits = ranked.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    keys = (ordered.to(tl.uint64) << 32) | (BLOCK_N - experts).to(tl.uint64)
    selected_keys = tl.topk(keys, BLOCK_TOPK)
    selected_ids = BLOCK_N - (selected_keys & 0xFFFFFFFF).to(tl.int32)
    selected_weights = tl.gather(scores, selected_ids, axis=0)
    slots = tl.arange(0, BLOCK_TOPK)
    selected_weights = tl.where(slots < TOPK, selected_weights, 0.0)
    scale = routed_scaling_factor
    if RENORMALIZE:
        weight_sum = tl.sum(selected_weights, axis=0)
        scale = scale / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    tl.store(
        topk_weights_ptr + row * TOPK + slots,
        selected_weights * scale,
        mask=slots < TOPK,
    )
    tl.store(topk_ids_ptr + row * TOPK + slots, selected_ids, mask=slots < TOPK)


def can_use_rocm_fused_router_gate(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
) -> bool:
    """Return whether the tensors match the tuned gfx950 fused gate."""
    try:
        _validate_inputs(hidden_states, router_weight, correction_bias, topk)
    except (RuntimeError, ValueError):
        return False
    return True


def _validate_inputs(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
) -> None:
    if not current_platform.is_rocm() or not on_gfx950():
        raise RuntimeError("rocm_fused_router_gate requires ROCm gfx950")
    if hidden_states.dim() != 2 or router_weight.dim() != 2:
        raise ValueError("hidden_states and router_weight must be 2D tensors")
    if hidden_states.dtype != torch.bfloat16 or router_weight.dtype != torch.bfloat16:
        raise ValueError("hidden_states and router_weight must have dtype bfloat16")
    if not hidden_states.is_contiguous() or not router_weight.is_contiguous():
        raise ValueError("hidden_states and router_weight must be contiguous")
    if hidden_states.device.type != "cuda":
        raise ValueError("hidden_states and router_weight must be on a GPU")
    if hidden_states.device != router_weight.device:
        raise ValueError("hidden_states and router_weight must be on the same device")
    shape = (hidden_states.shape[1], router_weight.shape[0])
    if (
        shape not in ROCM_FUSED_ROUTER_GATE_SUPPORTED_SHAPES
        or router_weight.shape[1] != shape[0]
    ):
        raise ValueError("supported (hidden_size, num_experts) pairs are (7168, 384)")
    if not 0 <= hidden_states.shape[0] <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [0, {_MAX_TOKENS}]")
    if not 0 < topk <= router_weight.shape[0]:
        raise ValueError("topk must be in (0, num_experts]")
    if correction_bias is not None:
        if correction_bias.device != hidden_states.device:
            raise ValueError(
                "correction_bias must be on the same device as hidden_states"
            )
        if not correction_bias.is_contiguous():
            raise ValueError("correction_bias must be contiguous")
        if correction_bias.dtype != torch.float32:
            raise ValueError("correction_bias must have dtype float32")
        if correction_bias.shape != (router_weight.shape[0],):
            raise ValueError("correction_bias must have shape (num_experts,)")


def rocm_fused_router_gate(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score ``hidden_states @ router_weight.T`` and select the top experts.

    Args:
        hidden_states: Token rows, shape ``(num_tokens, hidden_size)``.
        router_weight: Gate weight, shape ``(num_experts, hidden_size)``.
        correction_bias: Per-expert selection bias, or None.
        topk: Number of experts to select per token.
        renormalize: Whether to normalize the selected weights to sum to one.
        routed_scaling_factor: Scale applied to the selected weights.
        indices_dtype: Dtype of the returned expert ids.

    Returns:
        The routing weights and expert ids, both shaped ``(num_tokens, topk)``.

    """
    _validate_inputs(hidden_states, router_weight, correction_bias, topk)
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError("indices_dtype must be int32 or int64")
    num_tokens = hidden_states.shape[0]
    num_experts = router_weight.shape[0]
    topk_weights = hidden_states.new_empty((num_tokens, topk), dtype=torch.float32)
    topk_ids = hidden_states.new_empty((num_tokens, topk), dtype=indices_dtype)
    if num_tokens == 0:
        return topk_weights, topk_ids

    block_n, block_k = 64, 128
    if num_tokens <= 16:
        block_m, target_splits = 16, 14
    elif num_tokens <= 64:
        block_m, target_splits = 32, 14
    elif num_tokens <= 128:
        block_m, target_splits = 32, 7
    elif num_tokens <= 256:
        block_m, target_splits = 64, 7
    elif num_tokens <= 512:
        block_m, target_splits = 32, 4
    else:
        block_m, block_k, target_splits = 64, 64, 7
        if num_tokens > 768:
            block_n = 128
    chunk_k = triton.cdiv(hidden_states.shape[1], target_splits * block_k) * block_k
    split_k = triton.cdiv(hidden_states.shape[1], chunk_k)
    partial_logits = hidden_states.new_empty(
        (split_k, num_tokens, num_experts), dtype=torch.float32
    )
    _router_gate_gemm[
        (triton.cdiv(num_tokens, block_m), triton.cdiv(num_experts, block_n), split_k)
    ](
        hidden_states,
        router_weight,
        partial_logits,
        M=num_tokens,
        K=hidden_states.shape[1],
        N=num_experts,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        CHUNK_K=chunk_k,
        num_warps=4,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    _router_gate_reduce_topk[(num_tokens,)](
        partial_logits,
        correction_bias,
        topk_weights,
        topk_ids,
        routed_scaling_factor,
        M=num_tokens,
        N=num_experts,
        SPLIT_K=split_k,
        TOPK=topk,
        HAS_BIAS=correction_bias is not None,
        RENORMALIZE=renormalize,
        BLOCK_N=triton.next_power_of_2(num_experts),
        BLOCK_S=triton.next_power_of_2(split_k),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=4,
    )
    return topk_weights, topk_ids
