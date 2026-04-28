# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable

import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
import vllm.envs as envs
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
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )

    return topk_weights, topk_indices


def _topk_softplus_sqrt_torch(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
    input_tokens: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
) -> None:
    # Reference implementation of csrc/moe/topk_softplus_sqrt_kernels.cu used
    # on platforms where the fused kernel is unavailable (e.g. ROCm). Math
    # mirrors the kernel exactly: weight_base = sqrt(softplus(x)) per expert,
    # bias is added only for ranking (subtracted back from output), then
    # optional renormalize + routed_scaling_factor.
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[-1]

    # softplus(x) with beta=1 and the same numerical-stability cutoff used by
    # the kernel ((val_b > 20) ? val : log1p(exp(val_b)) / beta).
    x_f32 = gating_output.to(torch.float32)
    softplus_x = torch.nn.functional.softplus(x_f32, beta=1.0, threshold=20.0)
    weights_base = torch.sqrt(softplus_x)  # (T, E)

    use_hash = (
        input_tokens is not None and hash_indices_table is not None
    )

    if use_hash:
        # tid2eid: (V, k); input_tokens: (T,) -> selected_experts: (T, k)
        tid2eid = hash_indices_table
        selected_experts = tid2eid[input_tokens.to(torch.long)]
        selected_weights = torch.gather(
            weights_base, -1, selected_experts.to(torch.long)
        )
        if renormalize:
            denom = selected_weights.sum(dim=-1, keepdim=True)
            denom = torch.where(
                denom > 0, denom, torch.ones_like(denom)
            )
            selected_weights = selected_weights / denom
        selected_weights = selected_weights * routed_scaling_factor

        topk_weights.copy_(selected_weights.to(topk_weights.dtype))
        topk_indices.copy_(selected_experts.to(topk_indices.dtype))
        # The CUDA kernel leaves token_expert_indices untouched in the hash
        # path, so we mirror that (caller treats it as scratch in this case).
        return

    if e_score_correction_bias is not None:
        ranking = weights_base + e_score_correction_bias.to(torch.float32)
    else:
        ranking = weights_base

    _, topk_ids = torch.topk(ranking, topk, dim=-1)
    out_weights = torch.gather(weights_base, -1, topk_ids)
    if renormalize:
        denom = out_weights.sum(dim=-1, keepdim=True)
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        out_weights = out_weights / denom
    out_weights = out_weights * routed_scaling_factor

    topk_weights.copy_(out_weights.to(topk_weights.dtype))
    topk_indices.copy_(topk_ids.to(topk_indices.dtype))

    # token_expert_indices[t, k_idx] = k_idx * T + t (matches kernel's
    # source_rows write at line 388 of topk_softplus_sqrt_kernels.cu).
    arange_t = torch.arange(
        num_tokens,
        device=gating_output.device,
        dtype=token_expert_indices.dtype,
    ).unsqueeze(-1)
    arange_k = torch.arange(
        topk,
        device=gating_output.device,
        dtype=token_expert_indices.dtype,
    ).unsqueeze(0)
    token_expert_indices.copy_(arange_k * num_tokens + arange_t)


def vllm_topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    # The fused topk_softplus_sqrt CUDA kernel is gated behind #ifndef USE_ROCM
    # in csrc/moe/torch_bindings.cpp and the .cu source isn't added to
    # VLLM_MOE_EXT_SRC for ROCm builds (CMakeLists.txt). Fall back to a torch
    # reference on platforms that don't ship the symbol.
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        ops.topk_hash_softplus_sqrt(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
        )
    else:
        _topk_softplus_sqrt_torch(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
        )

    return topk_weights, topk_indices


@functools.lru_cache(maxsize=8)
def _aiter_get_num_expert_group(num_experts: int) -> int:
    _AITER_MAX_EXPERTS_PER_GROUP = 32
    g = max(1, -(-num_experts // _AITER_MAX_EXPERTS_PER_GROUP))
    while num_experts % g != 0:
        g += 1
    assert num_experts % g == 0, f"{num_experts=} not divisible by {g=}"
    assert num_experts // g <= _AITER_MAX_EXPERTS_PER_GROUP, (
        f"group size {num_experts // g} exceeds limit {_AITER_MAX_EXPERTS_PER_GROUP}"
    )
    return g


def fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    scoring_func: str,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
):
    if not rocm_aiter_ops.is_fused_moe_enabled():
        assert hidden_states.size(0) == gating_output.size(0), (
            "Number of tokens mismatch"
        )

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
            topk_weights, topk_ids = vllm_topk_softmax(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
            )
            if routed_scaling_factor != 1.0:
                topk_weights *= routed_scaling_factor
            return topk_weights, topk_ids
        elif scoring_func == "sigmoid":
            topk_weights, topk_ids = vllm_topk_sigmoid(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
            )
            if routed_scaling_factor != 1.0:
                topk_weights *= routed_scaling_factor
            return topk_weights, topk_ids
        elif scoring_func == "sqrtsoftplus":
            return vllm_topk_softplus_sqrt(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
                input_tokens,
                hash_indices_table,
                routed_scaling_factor,
            )
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_func}")

    elif rocm_aiter_ops.is_fused_moe_enabled() and scoring_func == "sigmoid":
        M = hidden_states.size(0)
        num_experts = gating_output.shape[-1]
        num_expert_group = _aiter_get_num_expert_group(num_experts)
        if topk >= num_expert_group:
            topk_weights = torch.empty(
                M, topk, dtype=torch.float32, device=hidden_states.device
            )
            topk_ids = torch.empty(
                M,
                topk,
                dtype=torch.int32 if indices_type is None else indices_type,
                device=hidden_states.device,
            )
            rocm_aiter_ops.biased_grouped_topk(
                gating_output,
                e_score_correction_bias,
                topk_weights,
                topk_ids,
                num_expert_group=num_expert_group,
                topk_group=num_expert_group,
                need_renorm=renormalize,
            )
            if routed_scaling_factor != 1.0:
                topk_weights *= routed_scaling_factor
            return topk_weights, topk_ids

    n_routed_experts = gating_output.shape[-1]
    if scoring_func == "softmax":
        scores = gating_output.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = F.softplus(gating_output).sqrt()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")
    if e_score_correction_bias is not None:
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores.view(-1, n_routed_experts)
    # For batch invariance, use sorted=True to ensure deterministic expert selection
    if hash_indices_table is not None:
        topk_indices = hash_indices_table[input_tokens]
    else:
        use_sorted = envs.VLLM_BATCH_INVARIANT
        topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=use_sorted)[
            1
        ]
    topk_weights = scores.gather(1, topk_indices)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(torch.float32)
    if routed_scaling_factor != 1.0:
        topk_weights *= routed_scaling_factor
    return topk_weights, topk_indices.to(
        torch.int32 if indices_type is None else indices_type
    )


class FusedTopKBiasRouter(BaseRouter):
    """Router using fused top-k with e_score_correction_bias."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        e_score_correction_bias: torch.Tensor | None = None,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
        *,
        scoring_func: str = "sigmoid",
        hash_indices_table: torch.Tensor | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        self.e_score_correction_bias = e_score_correction_bias
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self._hash_indices_table = hash_indices_table

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=True,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using fused top-k with bias."""
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias.data
            if self.e_score_correction_bias is not None
            else None,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            input_tokens=input_ids,
            hash_indices_table=self._hash_indices_table,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        return topk_weights, topk_ids
