# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Token-Routed Expert Dispatch — Fused Operations.

Optimized expert forward pass for i64 token-routed models.
Replaces Python per-expert loop with batched operations.

Two modes (auto-selected by batch size):
  - BMM mode  (N <= threshold): torch.bmm, fully parallel, zero loops
  - Chunked mode (N > threshold): sort-by-expert, memory-efficient

Since routing is deterministic (token_id % num_experts), no learned
gating is needed — dispatch is pure integer indexing.
"""

import torch
import torch.nn.functional as F


def fused_token_routed_forward(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    intermediate_per_tp: int,
    bmm_threshold: int = 48,
) -> torch.Tensor:
    """
    Fused expert forward with SwiGLU activation.

    Dispatches tokens to experts based on integer routing,
    then applies gate+up projection, SiLU gating, and down projection.

    Args:
        x: (num_tokens, hidden_size) — input hidden states
        gate_up_proj: (num_experts, hidden_size, 2 * intermediate_per_tp)
        down_proj: (num_experts, intermediate_per_tp, hidden_size)
        expert_ids: (num_tokens,) long — expert assignment per token
        num_experts: number of local experts
        intermediate_per_tp: intermediate dimension per TP shard
        bmm_threshold: max tokens for BMM mode (above → chunked)

    Returns:
        output: (num_tokens, hidden_size)
    """
    if x.shape[0] == 0:
        return x

    if x.shape[0] <= bmm_threshold:
        return _bmm_forward(x, gate_up_proj, down_proj, expert_ids,
                            intermediate_per_tp)
    return _chunked_forward(x, gate_up_proj, down_proj, expert_ids,
                            num_experts, intermediate_per_tp)


def _bmm_forward(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_ids: torch.Tensor,
    intermediate_per_tp: int,
) -> torch.Tensor:
    """
    Batched matmul forward — fully parallel across tokens.

    Selects each token's expert weights via indexing, then uses
    torch.bmm for zero-loop computation. Optimal for decode phase
    where batch sizes are small (1–48 tokens).
    """
    # Select each token's expert weights
    # (N,) → index into (E, H, 2I) → (N, H, 2I)
    sel_gu = gate_up_proj[expert_ids]
    sel_down = down_proj[expert_ids]

    # Gate+Up: (N, 1, H) @ (N, H, 2I) → (N, 2I)
    gu = torch.bmm(x.unsqueeze(1), sel_gu).squeeze(1)

    # SwiGLU activation
    gate = gu[..., :intermediate_per_tp]
    up = gu[..., intermediate_per_tp:]
    inter = F.silu(gate) * up

    # Down: (N, 1, I) @ (N, I, H) → (N, H)
    return torch.bmm(inter.unsqueeze(1), sel_down).squeeze(1)


def _chunked_forward(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    intermediate_per_tp: int,
) -> torch.Tensor:
    """
    Chunked forward — groups tokens by expert, processes in bulk.

    More memory-efficient than BMM for large batches (prefill phase).
    Sorts once, splits by expert boundaries.
    """
    N = x.shape[0]

    # Sort by expert for coalesced memory access
    sorted_idx = expert_ids.argsort(stable=True)
    sorted_x = x[sorted_idx]
    sorted_eid = expert_ids[sorted_idx]

    # Expert boundaries
    counts = torch.bincount(sorted_eid, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=x.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    output = torch.empty(N, x.shape[1], device=x.device, dtype=x.dtype)

    for eid in range(num_experts):
        s = offsets[eid].item()
        e = offsets[eid + 1].item()
        if s == e:
            continue

        chunk = sorted_x[s:e]

        # Fused gate+up projection
        gu = chunk @ gate_up_proj[eid]
        gate = gu[..., :intermediate_per_tp]
        up = gu[..., intermediate_per_tp:]

        # SwiGLU + down projection
        inter = F.silu(gate) * up
        output[s:e] = inter @ down_proj[eid]

    # Unsort to original token order
    result = torch.empty_like(output)
    result[sorted_idx] = output
    return result
