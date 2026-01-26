# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_sigmoid
    return vllm_topk_sigmoid


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
        )

        return topk_weights, topk_ids
