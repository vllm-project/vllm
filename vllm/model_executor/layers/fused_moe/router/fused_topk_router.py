# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
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
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        aiter_topK_meta_data,
    )

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
        # Check if we can fuse the shared expert activation (sigmoid)
        # into the topk_softmax kernel, so routing softmax and shared
        # expert scoring happen in a single kernel launch.
        fuse_sigmoid_in_kernel = (
            rocm_aiter_ops.fuse_sigmoid_in_kernel(aiter_topK_meta_data)
            and num_fused_shared_experts > 0
        )
        if fuse_sigmoid_in_kernel:
            # gating_output is [M, num_experts + num_shared] from fused gate.
            # The kernel applies routing softmax on [:num_experts] and
            # shared expert sigmoid on the last num_shared columns,
            # writing results into the pre-allocated buffer.
            # None check is inside rocm_aiter_ops.fuse_sigmoid_in_kernel
            total_topk_weights, total_topk_ids = aiter_topK_meta_data  # type: ignore[misc]
            total_topk_weights_slice = total_topk_weights[:M]
            topk_ids_slice = total_topk_ids[:M, :topk]

            topk_func = dispatch_topk_softmax_func(use_rocm_aiter=True)
            topk_func(
                total_topk_weights_slice,
                topk_ids_slice,
                token_expert_indices,
                gating_output,
                renormalize,
                num_fused_shared_experts,
                "sigmoid",
            )
            return (total_topk_weights_slice, total_topk_ids[:M], token_expert_indices)
        else:
            # When num_fused_shared_experts > 0 but kernel fusion is
            # unavailable, gating_output may be [M, num_experts + num_shared]
            # from the fused gate matmul.  Standard topk_softmax must only
            # see the routing columns to compute softmax correctly.
            if num_fused_shared_experts > 0:
                routing_logits = gating_output[:, :-num_fused_shared_experts]
                shared_logits = gating_output[:, -num_fused_shared_experts:]
            else:
                routing_logits = gating_output
                shared_logits = None

            topk_func = dispatch_topk_softmax_func(
                use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
            )
            topk_weights, topk_ids = topk_func(
                topk_weights,
                topk_ids,
                token_expert_indices,
                routing_logits,
                renormalize,
            )

            # Non-fused fallback: compute shared expert activation
            # (sigmoid) and inject weights into buffer manually.
            if shared_logits is not None and aiter_topK_meta_data is not None:
                from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
                    inject_shared_expert_weights,
                )

                shared_weights = torch.sigmoid(shared_logits)
                topk_weights, topk_ids = inject_shared_expert_weights(
                    topk_weights,
                    topk_ids,
                    topk=topk,
                    num_fused_shared_experts=num_fused_shared_experts,
                    shared_expert_weights=shared_weights,
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
        num_fused_shared_experts: int = 0,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
            num_fused_shared_experts=num_fused_shared_experts,
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
            num_fused_shared_experts=self.num_fused_shared_experts,
        )

        return topk_weights, topk_ids
