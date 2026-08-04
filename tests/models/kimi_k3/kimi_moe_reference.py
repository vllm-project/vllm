# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class KimiMoEReferenceOutput:
    router_logits: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    routed_input: torch.Tensor
    routed_combined: torch.Tensor
    routed_output: torch.Tensor
    shared_output: torch.Tensor
    output: torch.Tensor


def _situ(
    projected: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    gate, up = projected.chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def kimi_moe_reference(
    hidden_states: torch.Tensor,
    *,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_down_weight: torch.Tensor,
    routed_norm_weight: torch.Tensor,
    routed_up_weight: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    shared_w13_weight: torch.Tensor,
    shared_w2_weight: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    rms_norm_eps: float,
    situ_beta: float,
    situ_linear_beta: float | None,
) -> KimiMoEReferenceOutput:
    hidden_states = hidden_states.float()
    router_logits = F.linear(hidden_states, gate_weight.float())
    scores = router_logits.sigmoid()
    topk_ids = torch.topk(
        scores + correction_bias.float().unsqueeze(0),
        k=top_k,
        dim=-1,
        sorted=False,
    ).indices
    topk_weights = scores.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor

    routed_input = F.linear(hidden_states, routed_down_weight.float())
    routed_slots = torch.empty(
        hidden_states.shape[0],
        top_k,
        routed_input.shape[-1],
        dtype=torch.float32,
    )
    for token_idx in range(hidden_states.shape[0]):
        for slot_idx in range(top_k):
            expert_idx = int(topk_ids[token_idx, slot_idx])
            projected = F.linear(
                routed_input[token_idx],
                w13_weight[expert_idx].float(),
            )
            activated = _situ(projected, situ_beta, situ_linear_beta)
            routed_slots[token_idx, slot_idx] = F.linear(
                activated,
                w2_weight[expert_idx].float(),
            )
    routed_combined = (
        routed_slots * topk_weights.unsqueeze(-1)
    ).sum(dim=1)
    routed_normalized = F.rms_norm(
        routed_combined,
        (routed_combined.shape[-1],),
        routed_norm_weight.float(),
        rms_norm_eps,
    )
    routed_output = F.linear(routed_normalized, routed_up_weight.float())

    shared_projected = F.linear(hidden_states, shared_w13_weight.float())
    shared_activated = _situ(shared_projected, situ_beta, situ_linear_beta)
    shared_output = F.linear(shared_activated, shared_w2_weight.float())
    output = routed_output + shared_output

    return KimiMoEReferenceOutput(
        router_logits=router_logits,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        routed_input=routed_input,
        routed_combined=routed_combined,
        routed_output=routed_output,
        shared_output=shared_output,
        output=output,
    )
