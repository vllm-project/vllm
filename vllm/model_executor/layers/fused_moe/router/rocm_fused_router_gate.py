# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused router gate for gfx950: gate GEMM plus sqrtsoftplus expert selection.

The ROCm analogue of DeepGEMM's Mega-Gate (#56266). One program owns a token
row and holds all ``N`` scores in registers, so the fp32 logits never reach
memory and the selection kernel disappears. Every program streams the whole
gate weight, which is only affordable while that weight fits in the gfx950
last-level cache, so the shape allowlist and token cap below are deliberate.
"""

import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.triton_utils import tl, triton

# (hidden_size, num_experts) pairs whose gate weight fits the LLC budget:
#   (7168, 384) -> DeepSeek-V4.1-Flash, 5.5 MiB of bf16 weight
ROCM_FUSED_ROUTER_GATE_SUPPORTED_SHAPES = frozenset({(7168, 384)})
_MAX_TOKENS = 128


@triton.jit
def _rocm_fused_router_gate_kernel(
    hidden_states_ptr,
    router_weight_ptr,
    correction_bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    M,
    routed_scaling_factor,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    offsets_n = tl.arange(0, N)
    offsets_k = tl.arange(0, BLOCK_K)
    logits = tl.zeros((N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offsets_k_block = k_start + offsets_k
        mask_k = offsets_k_block < K
        hidden = tl.load(
            hidden_states_ptr + row * K + offsets_k_block, mask=mask_k, other=0.0
        ).to(tl.float32)
        weight = tl.load(
            router_weight_ptr + offsets_n[:, None] * K + offsets_k_block[None, :],
            mask=mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        logits += tl.sum(weight * hidden[None, :], axis=1)

    # sqrtsoftplus, in the numerically stable form the reference kernel uses:
    # softplus(x) = max(x, 0) + log1p(exp(-|x|)).
    scores = tl.sqrt(tl.maximum(logits, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(logits))))

    # Experts are selected by score plus correction bias; the routing weight is
    # the unbiased score.
    ranked = scores + tl.load(correction_bias_ptr + offsets_n) if HAS_BIAS else scores

    weight_sum = tl.zeros((), dtype=tl.float32)
    for slot in range(TOPK):
        expert = tl.argmax(ranked, axis=0)
        chosen = tl.sum(tl.where(offsets_n == expert, scores, 0.0), axis=0)
        weight_sum += chosen
        tl.store(topk_weights_ptr + row * TOPK + slot, chosen)
        tl.store(topk_ids_ptr + row * TOPK + slot, expert)
        ranked = tl.where(offsets_n == expert, float("-inf"), ranked)

    scale = routed_scaling_factor
    if RENORMALIZE:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    for slot in range(TOPK):
        offset = row * TOPK + slot
        tl.store(topk_weights_ptr + offset, tl.load(topk_weights_ptr + offset) * scale)


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
    num_tokens = hidden_states.shape[0]
    num_experts = router_weight.shape[0]
    topk_weights = hidden_states.new_empty((num_tokens, topk), dtype=torch.float32)
    topk_ids = hidden_states.new_empty((num_tokens, topk), dtype=indices_dtype)
    if num_tokens == 0:
        return topk_weights, topk_ids

    _rocm_fused_router_gate_kernel[(num_tokens,)](
        hidden_states,
        router_weight,
        correction_bias,
        topk_weights,
        topk_ids,
        num_tokens,
        routed_scaling_factor,
        K=hidden_states.shape[1],
        N=num_experts,
        TOPK=topk,
        HAS_BIAS=correction_bias is not None,
        RENORMALIZE=renormalize,
        BLOCK_K=1024,
        num_warps=8,
        num_stages=1,
    )
    return topk_weights, topk_ids
