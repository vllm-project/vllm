# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch
from torch.nn import functional as F

from vllm import envs


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    gating_output = gating_output.float()
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = scores.view(num_token, num_expert_group,
                                   -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(),
                                    float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(torch.int32)


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        return grouped_topk(hidden_states=hidden_states,
                            gating_output=router_logits,
                            topk=top_k,
                            renormalize=renormalize,
                            num_expert_group=num_expert_group,
                            topk_group=topk_group,
                            scoring_func=scoring_func,
                            routed_scaling_factor=routed_scaling_factor,
                            e_score_correction_bias=e_score_correction_bias)
    elif custom_routing_function is None:
        assert scoring_func == "softmax"
        topk_weights = torch.nn.functional.softmax(router_logits,
                                                   dim=1,
                                                   dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(topk_weights, top_k, dim=-1)
        if renormalize:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_ids.to(torch.int32)
    else:
        return custom_routing_function(hidden_states=hidden_states,
                                       gating_output=router_logits,
                                       topk=top_k,
                                       renormalize=renormalize)


class IPEXFusedMOE:

    def __init__(self, layer: torch.nn.Module) -> None:
        import intel_extension_for_pytorch as ipex
        layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
            layer.w13_weight,
            layer.w2_weight,
            use_prepack=envs.VLLM_CPU_MOE_PREPACK,
        )

    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", f"{activation} is not supported."
        assert not apply_router_weight_on_input
        assert routed_scaling_factor == 1.0, \
            f"routed_scaling_factor {routed_scaling_factor} is not supported."
        return layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            custom_routing_function,
            scoring_func,
            e_score_correction_bias,
        )


class SGLFusedMOE:

    def __init__(self, layer: torch.nn.Module) -> None:
        pass

    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", f"{activation} is not supported."
        assert not apply_router_weight_on_input
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        torch.ops._C.fused_experts_cpu(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            True,
            False,
            False,
            None,
            None,
            None,
            None,
            None,
            True,
        )
        return x


class CPUFusedMOE:

    def __init__(self, layer: torch.nn.Module) -> None:
        pass

    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", f"{activation} is not supported."
        assert not apply_router_weight_on_input
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        # Ref code from https://github.com/sgl-project/sglang/blob/716e682721397df103f347d22da8bd46c6016dab/python/sglang/srt/layers/moe/fused_moe_native.py#L53
        len_experts = global_num_experts

        cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
        cnts.scatter_(1, topk_ids.to(torch.int64), 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()

        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

            layer_w13_weight = layer.w13_weight[i]
            layer_w2_weight = layer.w2_weight[i]

            gate_up = F.linear(tokens_for_this_expert, layer_w13_weight)
            gate_up = silu_and_mul(gate_up)
            expert_out = F.linear(gate_up, layer_w2_weight)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs,
                         dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)

        new_x[idxs] = outs
        final_out = (new_x.view(
            *topk_ids.shape, -1).type(topk_weights.dtype).mul_(
                topk_weights.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype))
        return final_out
