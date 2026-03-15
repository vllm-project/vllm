# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Fused I64 Token-Routed Expert Forward — CUDA Graph Safe.

Registered as custom op `vllm::i64_token_routed_forward` and added
to splitting_ops so the piecewise CUDA graph system executes it
in eager mode (not captured/frozen in the graph).

CRITICAL: The routing (token_ids % num_experts) MUST happen inside
this custom op. Because this op is a splitting_op, it runs in eager
mode during CUDA graph replay. If routing is done outside (in the
captured graph), it gets frozen with dummy token_ids = 0 from graph
capture, routing everything to expert 0 → immediate EOS.

Uses BMM (batched matmul) dispatch like the i64 engine:
  sel_gu = gate_up_proj[expert_ids]  — index each token's expert
  gu = torch.bmm(x, sel_gu)         — batched matmul, one expert per token
This is numerically exact (no mask-multiply accumulation artifacts).
"""

import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op


def fused_i64_experts(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    intermediate_per_tp: int,
) -> torch.Tensor:
    """
    Fused expert forward with SwiGLU — BMM dispatch.

    Each token selects its expert weights via indexing, then
    torch.bmm computes all tokens in parallel. No per-expert
    loop, no mask-multiply, numerically exact.

    Args:
        x: (num_tokens, hidden_size)
        gate_up_proj: (num_experts, hidden_size, 2 * intermediate_per_tp)
        down_proj: (num_experts, intermediate_per_tp, hidden_size)
        expert_ids: (num_tokens,) long — already routed
        num_experts: number of local experts
        intermediate_per_tp: intermediate dim per TP shard

    Returns:
        output: (num_tokens, hidden_size)
    """
    num_tokens = x.shape[0]
    if num_tokens == 0:
        return x

    # Select each token's expert weights
    sel_gu = gate_up_proj[expert_ids]    # (N, H, 2I)
    sel_down = down_proj[expert_ids]     # (N, I, H)

    # Gate+Up: (N, 1, H) @ (N, H, 2I) → (N, 1, 2I) → (N, 2I)
    gu = torch.bmm(x.unsqueeze(1), sel_gu).squeeze(1)

    # SwiGLU
    gate = gu[..., :intermediate_per_tp]
    up = gu[..., intermediate_per_tp:]
    inter = F.silu(gate) * up

    # Down: (N, 1, I) @ (N, I, H) → (N, 1, H) → (N, H)
    return torch.bmm(inter.unsqueeze(1), sel_down).squeeze(1)


# --- Custom op wrapper for splitting_ops integration ---
# Routing happens HERE (inside the splitting op = eager during graph replay)

def i64_token_routed_forward(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    token_ids: torch.Tensor,
    num_experts: int,
    intermediate_per_tp: int,
    vocab_size: int,
    mu_router_weight: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """
    Custom op entry point — computes routing + expert forward.

    Routing is done HERE so it runs in eager mode during CUDA graph
    replay (this op is in splitting_ops). If routing were done outside,
    it would be captured with dummy token_ids (all zeros) → expert 0.
    """
    # === I64 Routing (inside splitting op = eager) ===
    token_ids_clamped = token_ids.clamp(0, vocab_size - 1)
    expert_ids = (token_ids_clamped % num_experts).long()

    # Mu-guided bias — always compute (no CPU sync like .any())
    # When mu is zeros, mu_logits is zeros, base_one_hot * 10.0 dominates,
    # argmax returns the same expert as base → no-op, CUDA graph safe.
    mu_logits = F.linear(mu, mu_router_weight)  # (N, num_experts)
    base_one_hot = F.one_hot(expert_ids, num_experts).float()
    combined_logits = base_one_hot * 10.0 + mu_logits
    expert_ids = combined_logits.argmax(dim=-1)

    return fused_i64_experts(
        x, gate_up_proj, down_proj, expert_ids,
        num_experts, intermediate_per_tp,
    )


def i64_token_routed_forward_fake(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    token_ids: torch.Tensor,
    num_experts: int,
    intermediate_per_tp: int,
    vocab_size: int,
    mu_router_weight: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """Fake impl for torch.compile — returns tensor of correct shape."""
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="i64_token_routed_forward",
    op_func=i64_token_routed_forward,
    fake_impl=i64_token_routed_forward_fake,
)
