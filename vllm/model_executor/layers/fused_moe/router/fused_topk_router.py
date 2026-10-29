# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


def _get_padding_mask(num_tokens: int) -> torch.Tensor | None:
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        is_padding = get_forward_context().is_padding
        return is_padding[:num_tokens] if is_padding is not None else None
    return None


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
        is_padding=_get_padding_mask(topk_indices.shape[0]),
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
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


# At E=512 / top-10, AITER's topk_gating is faster than the legacy launcher
# through this many tokens; its one-row generic prefill path regresses above it.
_AITER_TOPK_GATING_MAX_TOKENS = 4096


def _aiter_topk_gating_supported(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    num_shared_experts: int,
    shared_expert_scoring_func: str,
) -> bool:
    """Whether AITER's ``topk_gating`` can serve this softmax top-k launch.

    It needs contiguous gating rows and at most ``_AITER_TOPK_GATING_MAX_TOKENS``
    tokens. Fused shared experts are scored with sigmoid after the routed
    top-k: 1, 2, 4 or 8 of them, with ``topk_weights`` and ``topk_indices``
    sharing a row stride and ``topk_weights`` wide enough for the shared
    columns. Shared expert ids stay as the caller prefilled them.
    """
    if (
        not gating_output.is_contiguous()
        or gating_output.shape[0] > _AITER_TOPK_GATING_MAX_TOKENS
    ):
        return False
    if num_shared_experts == 0:
        return not shared_expert_scoring_func
    return (
        shared_expert_scoring_func == "sigmoid"
        and num_shared_experts in (1, 2, 4, 8)
        and topk_weights.stride(0) == topk_indices.stride(0)
        and topk_weights.shape[-1] >= topk_indices.shape[-1] + num_shared_experts
    )


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
    *,
    topk_weights: torch.Tensor | None = None,
    topk_indices: torch.Tensor | None = None,
    gating_output: torch.Tensor | None = None,
    num_shared_experts: int = 0,
    shared_expert_scoring_func: str = "",
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Pick the softmax top-k implementation for one launch.

    On ROCm with AITER, ``topk_gating`` is chosen on gfx942 and gfx950 when the
    launch described by the tensors and shared-expert arguments is one it supports;
    every other AITER launch uses the legacy ``topk_softmax``. The tensors are
    optional: without them the launch cannot be judged and the legacy launcher
    is used.
    """
    if use_rocm_aiter:
        if (
            rocm_aiter_ops.is_topk_gating_enabled()
            and topk_weights is not None
            and topk_indices is not None
            and gating_output is not None
            and _aiter_topk_gating_supported(
                topk_weights,
                topk_indices,
                gating_output,
                num_shared_experts,
                shared_expert_scoring_func,
            )
        ):
            return rocm_aiter_ops.topk_gating
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
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled(),
            topk_weights=topk_weights,
            topk_indices=topk_ids,
            gating_output=gating_output,
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
        scoring_func: str = "softmax",
        renormalize: bool = True,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
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
