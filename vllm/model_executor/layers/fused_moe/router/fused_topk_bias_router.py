# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

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
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import (
    can_use_dsv4_topk,
    dsv4_topk,
)


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
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        routed_scaling_factor,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


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
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
) -> tuple[torch.Tensor, ...]:
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
        is_padding=_get_padding_mask(topk_indices.shape[0]),
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
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
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
):
    if (
        input_tokens is not None
        and hash_indices_table is not None
        and input_tokens.dtype != hash_indices_table.dtype
    ):
        input_tokens = input_tokens.to(dtype=hash_indices_table.dtype)

    if bias_vl is not None:
        # Image tokens carry five consecutive in-vocab sentinel ids starting
        # at image_sentinel_lo and select experts with bias_vl instead of the
        # regular routing path. image_sentinel_lo == 0 disables this.
        assert input_tokens is not None, "bias_vl routing requires input_tokens"

    if not rocm_aiter_ops.is_fused_moe_enabled():
        assert hidden_states.size(0) == gating_output.size(0), (
            "Number of tokens mismatch"
        )

        output_indices_dtype = torch.int32 if indices_type is None else indices_type
        if (
            scoring_func == "sqrtsoftplus"
            and hash_indices_table is None
            and can_use_dsv4_topk(
                gating_output,
                e_score_correction_bias,
                topk,
                renormalize,
                output_indices_dtype,
            )
        ):
            assert e_score_correction_bias is not None
            return dsv4_topk(
                gating_output,
                e_score_correction_bias,
                output_indices_dtype,
                routed_scaling_factor,
                input_ids=input_tokens,
                bias_vl=bias_vl,
                image_sentinel_lo=image_sentinel_lo,
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
                routed_scaling_factor,
            )
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
                bias_vl=bias_vl,
                image_sentinel_lo=image_sentinel_lo,
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

    if scoring_func == "sqrtsoftplus":
        M = hidden_states.size(0)
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
            bias_vl=bias_vl,
            image_sentinel_lo=image_sentinel_lo,
        )

    n_routed_experts = gating_output.shape[-1]
    if scoring_func == "softmax":
        scores = gating_output.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")
    if e_score_correction_bias is not None:
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores.view(-1, n_routed_experts)
    image_mask = None
    if bias_vl is not None and image_sentinel_lo > 0:
        # Image tokens (five consecutive sentinel ids starting at
        # image_sentinel_lo) select experts with bias_vl instead of
        # e_score_correction_bias / the hash table. Ids above the sentinel
        # block are regular special tokens and must not match.
        assert input_tokens is not None, "bias_vl routing requires input_tokens"
        image_mask = (
            (input_tokens >= image_sentinel_lo) & (input_tokens < image_sentinel_lo + 5)
        ).unsqueeze(-1)
        text_bias = (
            e_score_correction_bias
            if e_score_correction_bias is not None
            else torch.zeros_like(bias_vl)
        )
        row_bias = torch.where(image_mask, bias_vl, text_bias)
        scores_for_choice = scores.view(-1, n_routed_experts) + row_bias
    # For batch invariance, use sorted=True to ensure deterministic expert selection
    if hash_indices_table is not None:
        assert input_tokens is not None
        if image_mask is None:
            topk_indices = hash_indices_table[input_tokens]
        else:
            # Clamp sentinel rows (overwritten below) to keep the table
            # lookup well-defined.
            safe_ids = torch.where(image_mask.squeeze(-1), 0, input_tokens).to(
                hash_indices_table.dtype
            )
            topk_indices = hash_indices_table[safe_ids]
            vl_indices = torch.topk(scores_for_choice, k=topk, dim=-1)[1]
            topk_indices = torch.where(
                image_mask, vl_indices.to(topk_indices.dtype), topk_indices
            )
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
        e_score_correction_bias: torch.Tensor | None = None,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        eplb_state: EplbLayerState | None = None,
        *,
        scoring_func: str = "sigmoid",
        hash_indices_table: torch.Tensor | None = None,
        num_fused_shared_experts: int = 0,
        shared_expert_weight: float = 1.0,
        bias_vl: torch.Tensor | None = None,
        image_sentinel_lo: int = 0,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.e_score_correction_bias = e_score_correction_bias
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self._hash_indices_table = hash_indices_table
        # Vision bias: image sentinel tokens (five consecutive in-vocab ids
        # starting at image_sentinel_lo) select experts with bias_vl instead
        # of e_score_correction_bias / the hash table.
        self.bias_vl = bias_vl
        self.image_sentinel_lo = image_sentinel_lo
        # Fused shared experts: append constant slots (ids immediately after
        # the routed experts, [global, global+n)) routed to by every token at
        # ``shared_expert_weight``, AFTER the routed top-k is renormalized.
        self.num_fused_shared_experts = num_fused_shared_experts
        self.shared_expert_weight = shared_expert_weight

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=True,
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
            bias_vl=self.bias_vl.data if self.bias_vl is not None else None,
            image_sentinel_lo=self.image_sentinel_lo,
        )

        if self.num_fused_shared_experts > 0:
            m = topk_ids.shape[0]
            n = self.num_fused_shared_experts
            # global_num_experts counts only the routed experts; the fused
            # shared experts occupy the slots immediately after them, i.e. ids
            # [global_num_experts, global_num_experts + n).
            base = self.global_num_experts
            shared_ids = torch.arange(
                base, base + n, dtype=topk_ids.dtype, device=topk_ids.device
            ).expand(m, n)
            shared_w = torch.full(
                (m, n),
                self.shared_expert_weight,
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )
            topk_ids = torch.cat([topk_ids, shared_ids], dim=-1)
            topk_weights = torch.cat([topk_weights, shared_w], dim=-1)

        return topk_weights, topk_ids
