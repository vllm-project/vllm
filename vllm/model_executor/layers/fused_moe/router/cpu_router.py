# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router for CPU MoE experts.

This is the routing logic that used to live inline in each CPU
FusedMoEExpertsMonolithic.apply() (see git history of
experts/cpu_moe.py); factoring it out into a proper FusedMoERouter is what
lets CPU MoE experts be FusedMoEExpertsModular like every other backend.
The math below is unchanged from that prior inline version -- this is a
relocation, not a rewrite.
"""

from collections.abc import Callable

import torch

from vllm._custom_ops import biased_topk_cpu, hash_topk_cpu
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


def _grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(torch.int32)


def _sqrtsoftplus_bias_topk(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
    input_ids: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek V4's routing: weight = sqrt(softplus(logit)); expert
    selection uses a bias-corrected score (weight + correction_bias), but
    the returned weight for each selected expert is the *unbiased* value
    (the "noaux_tc" pattern -- bias steers selection only). Always
    renormalizes selected weights to sum to ``routed_scaling_factor``,
    matching the reference Triton kernel
    (vllm/model_executor/layers/fused_moe/router/dsv4_topk.py).

    When ``hash_indices_table`` is given (hash-routed layers), expert
    selection instead comes from ``hash_indices_table[input_ids]`` --
    the bias-corrected-score topk is skipped entirely, matching the CUDA
    reference kernel (``dsv4HashTopkSoftplusSqrt``).

    Backed by the ported SGLang AVX512 kernels `biased_topk_cpu` (the
    correction-bias path) and `hash_topk_cpu` (the hash-routed-layer path);
    see csrc/cpu/sgl-kernels/topk.cpp.
    """
    assert renormalize, "DeepseekV4 routing always renormalizes"
    gating_output = router_logits.contiguous()
    num_experts = gating_output.shape[1]
    if hash_indices_table is not None:
        assert input_ids is not None
        # hash_indices_table is int32 by construction (see cpu/model.py) and
        # advanced indexing always allocates a fresh contiguous tensor.
        tid2eid = hash_indices_table[input_ids]
        return hash_topk_cpu(
            gating_output,
            tid2eid,
            top_k,
            "sqrtsoftplus",
            0,  # num_fused_shared_experts: never fused into routing on CPU
            num_experts,
            routed_scaling_factor,
        )
    assert e_score_correction_bias is not None
    return biased_topk_cpu(
        gating_output,  # only used for num_tokens/device (dtype forced fp32)
        gating_output,
        e_score_correction_bias.contiguous(),
        top_k,
        renormalize,
        "sqrtsoftplus",
        0,  # num_fused_shared_experts: never fused into routing on CPU
        routed_scaling_factor,
        True,  # apply_routed_scaling_factor_on_output
    )


def _softmax_topk(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_logit_vals, topk_idx = torch.topk(router_logits, k=top_k, dim=-1, sorted=False)
    if renormalize:
        topk_vals = torch.softmax(topk_logit_vals, dim=-1)
    else:
        logZ = torch.logsumexp(router_logits, dim=-1, keepdim=True)
        topk_vals = (topk_logit_vals - logZ).exp()
    return topk_vals.to(torch.float32), topk_idx.to(torch.int32)


class CPURouter(BaseRouter):
    """Router covering every routing scheme CPU MoE experts support: plain
    softmax top-k, grouped top-k (with an optional correction bias), an
    arbitrary custom_routing_function, and DeepSeek V4's sqrtsoftplus scheme
    (bias-corrected or hash-routed via the ported biased_topk_cpu/
    hash_topk_cpu kernels). All of the above except sqrtsoftplus are plain
    torch ops so Dynamo traces straight through into them like it did when
    this logic lived inline in FusedMoEExpertsMonolithic.apply() -- no new
    compile boundary is introduced here.
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        renormalize: bool = True,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        hash_indices_table: torch.Tensor | None = None,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.custom_routing_function = custom_routing_function
        self._hash_indices_table = hash_indices_table

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            has_e_score_bias=self.e_score_correction_bias is not None,
            routed_scaling_factor=self.routed_scaling_factor,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_grouped_topk:
            assert self.topk_group is not None
            assert self.num_expert_group is not None
            return _grouped_topk(
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
            )
        elif self.scoring_func == "sqrtsoftplus":
            assert (
                self._hash_indices_table is not None
                or self.e_score_correction_bias is not None
            )
            return _sqrtsoftplus_bias_topk(
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                e_score_correction_bias=self.e_score_correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                input_ids=input_ids,
                hash_indices_table=self._hash_indices_table,
            )
        elif self.custom_routing_function is None:
            assert self.scoring_func == "softmax"
            return _softmax_topk(router_logits, self.top_k, self.renormalize)
        else:
            topk_weights, topk_ids = self.custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
            # cpu_fused_moe reads routing tensors as contiguous float32/int32
            # buffers and does not account for tensor strides.
            topk_weights = topk_weights.to(torch.float32).contiguous()
            topk_ids = topk_ids.to(torch.int32).contiguous()
            return topk_weights, topk_ids


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing helper for the CPU MoE experts that still use the monolithic
    contract and so must compute routing themselves from ``router_logits``
    (the zentorch-accelerated path and the ARM W4A8 dynamic-quant kernel,
    kept monolithic since zentorch/Arm hardware isn't available to validate
    a modular migration here). Modular CPU experts get routing from
    ``CPURouter`` instead; this is a thin wrapper around the same logic for
    the two classes that can't use it.
    """
    router = CPURouter(
        top_k=top_k,
        global_num_experts=router_logits.shape[-1],
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        renormalize=renormalize,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        custom_routing_function=custom_routing_function,
        hash_indices_table=hash_indices_table,
    )
    return router._compute_routing(
        hidden_states, router_logits, None, input_ids=input_ids
    )
