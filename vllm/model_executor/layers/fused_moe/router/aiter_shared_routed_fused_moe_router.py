# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    dispatch_topk_softmax_func,
)


class AiterSharedRoutedFusedMoERouter(BaseRouter):
    """ROCm AITER router for models with fused shared experts.

    With a shared-expert gate (e.g. Qwen3-Next), the gate logits are appended
    to router_logits and each token's shared-expert weight is
    sigmoid(gate logit). When the AITER topk_softmax kernel supports sigmoid
    fusion, the routing softmax and shared-expert sigmoid are computed in a
    single kernel launch. Otherwise the shared-expert weights are injected into
    the pre-allocated AITER buffer via a fallback path.

    Without a shared-expert gate (e.g. DeepSeek-V2 style), router_logits only
    hold the routed experts and every token uses the shared experts with the
    constant shared_expert_weight. The routed top-k is written directly into
    the pre-allocated AITER buffer, whose shared-expert slots are pre-filled.

    Only instantiated when rocm_aiter fused-MoE is active and
    num_fused_shared_experts > 0.
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        num_fused_shared_experts: int,
        eplb_state: EplbLayerState | None = None,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        has_shared_expert_gate: bool = False,
        shared_expert_weight: float = 1.0,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.num_fused_shared_experts = num_fused_shared_experts
        self.has_shared_expert_gate = has_shared_expert_gate
        self.shared_expert_weight = shared_expert_weight

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
        assert hidden_states.size(0) == router_logits.size(0), (
            "Number of tokens mismatch"
        )

        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
            aiter_topK_meta_data,
        )

        M = hidden_states.size(0)
        topk = self.top_k
        num_fse = self.num_fused_shared_experts

        token_expert_indices = torch.empty(
            M, topk, dtype=torch.int32, device=hidden_states.device
        )

        if not self.has_shared_expert_gate:
            assert aiter_topK_meta_data is not None, (
                "AITER topK meta data is not initialized. Ensure "
                "init_aiter_topK_meta_data is called before routing."
            )
            total_topk_weights, total_topk_ids = aiter_topK_meta_data
            assert total_topk_weights.shape[0] >= M, (
                f"AITER topK meta data supports {total_topk_weights.shape[0]} "
                f"tokens, but got {M} tokens."
            )
            total_topk_weights = total_topk_weights[:M]
            total_topk_ids = total_topk_ids[:M]

            # Both kernels honor the row strides, so they fill the routed slots
            # in place and leave the pre-filled shared-expert slots untouched.
            topk_weights = total_topk_weights[:, :topk]
            topk_ids = total_topk_ids[:, :topk]
            topk_func = dispatch_topk_softmax_func(
                use_rocm_aiter=True,
                topk_weights=topk_weights,
                topk_indices=topk_ids,
                gating_output=router_logits,
            )
            topk_func(
                topk_weights,
                topk_ids,
                token_expert_indices,
                router_logits,
                self.renormalize,
            )
            # init_aiter_topK_meta_data pre-fills the shared slots with 1.0.
            if self.shared_expert_weight != 1.0:
                total_topk_weights[:, topk : topk + num_fse].fill_(
                    self.shared_expert_weight
                )
            return total_topk_weights, total_topk_ids

        if rocm_aiter_ops.fuse_sigmoid_in_kernel(aiter_topK_meta_data):
            total_topk_weights, total_topk_ids = aiter_topK_meta_data  # type: ignore[misc]
            total_topk_weights_slice = total_topk_weights[:M]
            topk_ids_slice = total_topk_ids[:M, :topk]

            topk_func = dispatch_topk_softmax_func(
                use_rocm_aiter=True,
                topk_weights=total_topk_weights_slice,
                topk_indices=topk_ids_slice,
                gating_output=router_logits,
                num_shared_experts=num_fse,
                shared_expert_scoring_func="sigmoid",
            )
            topk_func(
                total_topk_weights_slice,
                topk_ids_slice,
                token_expert_indices,
                router_logits,
                self.renormalize,
                num_fse,
                "sigmoid",
            )
            return total_topk_weights_slice, total_topk_ids[:M]

        routing_logits = router_logits[:, :-num_fse]
        shared_logits = router_logits[:, -num_fse:]

        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            M,
            topk,
            dtype=torch.int32 if indices_type is None else indices_type,
            device=hidden_states.device,
        )

        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled(),
            topk_weights=topk_weights,
            topk_indices=topk_ids,
            gating_output=routing_logits,
        )
        topk_weights, topk_ids = topk_func(
            topk_weights,
            topk_ids,
            token_expert_indices,
            routing_logits,
            self.renormalize,
        )

        if aiter_topK_meta_data is not None:
            from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
                inject_shared_expert_weights,
            )

            shared_weights = torch.sigmoid(shared_logits)
            topk_weights, topk_ids = inject_shared_expert_weights(
                topk_weights,
                topk_ids,
                topk=topk,
                num_fused_shared_experts=num_fse,
                shared_expert_weights=shared_weights,
            )

        return topk_weights, topk_ids
