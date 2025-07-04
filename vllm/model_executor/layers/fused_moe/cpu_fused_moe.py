# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch

from vllm import envs


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
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", f"{activation} is not supported."
        assert not apply_router_weight_on_input
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

    @staticmethod
    def _grouped_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
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
            # Store original scores before applying correction bias. We use
            # biased scores for expert selection but original scores for
            # routing weights
            original_scores = scores
            scores = scores + e_score_correction_bias.unsqueeze(0)
            group_scores = (scores.view(num_token, num_expert_group,
                                        -1).topk(2, dim=-1)[0].sum(dim=-1))
        else:
            group_scores = scores.view(num_token, num_expert_group,
                                       -1).max(dim=-1).values  # [n, n_group]
        group_idx = torch.topk(group_scores,
                               k=topk_group,
                               dim=-1,
                               sorted=False)[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = group_mask.unsqueeze(-1).expand(
            num_token, num_expert_group,
            scores.shape[-1] // num_expert_group).reshape(num_token,
                                                          -1)  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(),
                                        float("-inf"))  # [n, e]

        if e_score_correction_bias is not None:
            topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
            # Use original unbiased scores for the routing weights
            topk_weights = original_scores.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(tmp_scores,
                                                k=topk,
                                                dim=-1,
                                                sorted=False)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1,
                                                           keepdim=True)

        return topk_weights, topk_ids.to(torch.int32)

    @staticmethod
    def _select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = SGLFusedMOE._grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias)
        elif custom_routing_function is None:
            assert scoring_func == "softmax"
            topk_weights = torch.nn.functional.softmax(router_logits,
                                                       dim=1,
                                                       dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, top_k, dim=-1)
            if renormalize:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_ids = topk_ids.to(torch.int32)
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids

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
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", f"{activation} is not supported."
        assert not apply_router_weight_on_input
        topk_weights, topk_ids = SGLFusedMOE._select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
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
