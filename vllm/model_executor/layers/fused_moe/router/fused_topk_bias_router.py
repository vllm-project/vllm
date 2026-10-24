# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import Any

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

# AITER instantiates biased_grouped_topk only for these NUM_GRP values, with at
# most this many experts per group.
AITER_SUPPORTED_NUM_GRP = (1, 2, 4, 8)
AITER_MAX_EXPERTS_PER_GROUP = 32

# A forward whose longest request query fits within this many tokens is
# treated as a decode step by ``_forward_allows_fused_shared_append``
# (plain decode = 1 query token; speculative decode = 1 + num_spec_tokens;
# prefill chunks are far larger). Deliberately below the heterogeneous
# path's default append threshold of 16 tokens.
_DECODE_MAX_QUERY_LEN = 8


def _forward_allows_fused_shared_append() -> bool:
    """Whether the current forward pass may append fused shared-expert slots.

    False only when the current v1 forward is unambiguously a decode step:
    the longest request query in the batch fits within
    ``_DECODE_MAX_QUERY_LEN`` tokens. Prefill, mixed, and chunked-prefill
    forwards return True. Callers without a forward context (unit tests,
    non-v1 paths) also return True, so the batch-size threshold stays the
    only gate there -- matching the stock behavior.

    Reads only CPU-side Python ints from the attention metadata
    (``max_query_len``), so it is capture-safe: no device synchronization,
    legal while a CUDA graph capture is active.
    """
    if not is_forward_context_available():
        return True
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, list):
        # DBO keeps one metadata per microbatch; the batch-level query
        # shape is shared, so the first element decides.
        if not attn_metadata:
            return True
        attn_metadata = attn_metadata[0]
    if not isinstance(attn_metadata, dict) or not attn_metadata:
        return True
    common = next(iter(attn_metadata.values()))
    max_query_len = getattr(common, "max_query_len", None)
    if not isinstance(max_query_len, int):
        return True
    return max_query_len > _DECODE_MAX_QUERY_LEN


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
    num_fused_shared_experts: int = 0,
    shared_expert_base: int = 0,
    shared_expert_weight: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """Softplus-sqrt biased top-k.

    With ``num_fused_shared_experts > 0`` the two top-k outputs must be
    (num_tokens, topk + n) buffers; the kernel writes the shared columns
    itself (see ops.topk_hash_softplus_sqrt).
    """
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
        num_fused_shared_experts=num_fused_shared_experts,
        shared_expert_base=shared_expert_base,
        shared_expert_weight=shared_expert_weight,
    )

    return topk_weights, topk_indices


@functools.lru_cache(maxsize=8)
def _aiter_get_num_expert_group(num_experts: int) -> int:
    g = max(1, -(-num_experts // AITER_MAX_EXPERTS_PER_GROUP))
    while num_experts % g != 0:
        g += 1
    assert num_experts % g == 0, f"{num_experts=} not divisible by {g=}"
    # Grouping is a no-op here (topk_group == num_expert_group), so any divisor
    # within the limit routes identically and we are free to pick one AITER has
    # a kernel for. If none fits, the unsupported value is returned as-is --
    # callers must gate on _aiter_can_use_biased_grouped_topk.
    if g not in AITER_SUPPORTED_NUM_GRP:
        g = next(
            (
                c
                for c in sorted(AITER_SUPPORTED_NUM_GRP, reverse=True)
                if num_experts % c == 0
                and num_experts // c <= AITER_MAX_EXPERTS_PER_GROUP
            ),
            g,
        )
    assert num_experts // g <= AITER_MAX_EXPERTS_PER_GROUP, (
        f"group size {num_experts // g} exceeds limit {AITER_MAX_EXPERTS_PER_GROUP}"
    )
    return g


def _aiter_can_use_biased_grouped_topk(num_experts: int, topk: int) -> bool:
    """Whether AITER has an instantiated kernel for this routing shape.

    Both checks are needed: an unsupported group count can still be small
    enough to satisfy ``topk >= g`` (num_experts=33 gives g=3).
    """
    g = _aiter_get_num_expert_group(num_experts)
    return topk >= g and g in AITER_SUPPORTED_NUM_GRP


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
    out_topk_weights: torch.Tensor | None = None,
    out_topk_ids: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
    shared_expert_base: int = 0,
    shared_expert_weight: float = 1.0,
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
        padding_mask = _get_padding_mask(gating_output.shape[0])
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
                is_padding=padding_mask,
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
        if _aiter_can_use_biased_grouped_topk(num_experts, topk):
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
        if out_topk_weights is not None or out_topk_ids is not None:
            # In-kernel fused-shared-experts fast path: the caller provided
            # persistent (M, topk + num_fused_shared_experts) buffers and the
            # top-k kernel writes the shared columns itself, replacing the
            # arange/full/cat append (4 tiny launch-bound kernels/layer).
            assert out_topk_weights is not None and out_topk_ids is not None
            assert num_fused_shared_experts > 0
            expected = (M, topk + num_fused_shared_experts)
            assert tuple(out_topk_weights.shape) == expected, (
                f"out_topk_weights shape {tuple(out_topk_weights.shape)} != {expected}"
            )
            assert tuple(out_topk_ids.shape) == expected, (
                f"out_topk_ids shape {tuple(out_topk_ids.shape)} != {expected}"
            )
            topk_weights, topk_ids = out_topk_weights, out_topk_ids
        else:
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
            num_fused_shared_experts=num_fused_shared_experts,
            shared_expert_base=shared_expert_base,
            shared_expert_weight=shared_expert_weight,
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


# ---------------------------------------------------------------------------
# Persistent top-k output buffers for the fused-shared-experts fast path.
#
# Appending the shared slots with torch.arange/full/cat costs four tiny
# launch-latency-bound kernels per MoE layer per decode step (+606 us/step
# measured on DSV4.1-Flash TP4 decode, all of it on the critical path).
# Instead the top-k kernel is handed a persistent (capacity, topk + n)
# buffer and writes the shared columns itself, so the append costs zero
# extra kernels.
#
# Buffers are keyed by (device, total width, indices dtype) and shared by all
# layers with the same routing shape. They grow on demand; superseded buffers
# stay alive because captured CUDA graphs may still reference their storage.
# The kernel rewrites every (row, column) of the returned view on each call,
# so cross-layer reuse on a single stream is safe.
# ---------------------------------------------------------------------------
_FUSED_SHARED_TOPK_BUFS: dict = {}
_FUSED_SHARED_TOPK_BUF_KEEPALIVE: list = []
_FUSED_SHARED_TOPK_BUF_INITIAL_ROWS = 1024


def _get_fused_shared_topk_buffers(
    num_tokens: int,
    topk: int,
    num_fused_shared: int,
    shared_expert_base: int,
    device: torch.device,
    indices_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return persistent (ids, weights) buffers of shape
    (capacity, topk + num_fused_shared), or None when growing is impossible
    (a CUDA graph capture is in progress and the capacity is insufficient —
    the caller then falls back to the cat-based append)."""
    total = topk + num_fused_shared
    key = (device.index or 0, total, indices_dtype)
    bufs = _FUSED_SHARED_TOPK_BUFS.get(key)
    if bufs is None or bufs[0].shape[0] < num_tokens:
        if torch.cuda.is_current_stream_capturing():
            # Never (re)allocate during graph capture: the captured graph
            # must not observe fresh addresses.
            return None
        if bufs is not None:
            # Keep the old storage alive: captured graphs may still write
            # into it (multi-shape capture sets replay sequentially).
            _FUSED_SHARED_TOPK_BUF_KEEPALIVE.append(bufs)
        rows = max(
            _FUSED_SHARED_TOPK_BUF_INITIAL_ROWS,
            num_tokens,
            bufs[0].shape[0] * 2 if bufs is not None else 0,
        )
        ids_buf = torch.empty(rows, total, dtype=indices_dtype, device=device)
        weights_buf = torch.empty(rows, total, dtype=torch.float32, device=device)
        # Pre-fill the shared id columns once (defensive only: the kernel
        # rewrites every exposed element on each call).
        ids_buf[:, topk:].copy_(
            torch.arange(
                shared_expert_base,
                shared_expert_base + num_fused_shared,
                dtype=indices_dtype,
                device=device,
            ).expand(rows, num_fused_shared)
        )
        bufs = (ids_buf, weights_buf)
        _FUSED_SHARED_TOPK_BUFS[key] = bufs
    return bufs


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
        # Optional batch-size gate for the shared-slot append (set by models
        # whose fused-shared-expert implementation engages only for
        # prefill-sized batches, e.g. the DeepSeek-V4 heterogeneous FHMoE
        # path). None (default) appends at every batch size, which keeps the
        # FP8->MXFP4 requant fusion and every other caller unchanged.
        self.shared_expert_append_min_tokens: int | None = None
        # Prefill-only append (set by models whose fused-shared-expert
        # implementation is a prefill optimization -- the DeepSeek-V4
        # heterogeneous FHMoE path): when True, unambiguous decode forwards
        # never append the shared slots, regardless of batch size, so
        # decode always keeps the stock separate overlapped shared expert.
        # The FP8->MXFP4 requant fusion and every other caller leave this
        # False and are unchanged.
        self.shared_expert_append_prefill_only: bool = False

    def _append_shared_experts_for(self, num_tokens: int) -> bool:
        """Whether fused shared-expert slots are appended at this batch size.

        Below ``shared_expert_append_min_tokens`` (when set) the router emits
        plain routed-width top-k outputs so decode-sized batches keep the
        stock overlapped separate shared-expert execution; the fused experts
        layer dispatches on the arrived route width. With
        ``shared_expert_append_prefill_only`` the append is additionally
        restricted to non-decode forwards (see
        ``_forward_allows_fused_shared_append``): fusion then engages only
        for prefill-shaped batches, and decode batches of any size keep the
        stock layout.
        """
        min_tokens = self.shared_expert_append_min_tokens
        prefill_only = self.shared_expert_append_prefill_only
        if min_tokens is None and not prefill_only:
            return True
        if prefill_only and not _forward_allows_fused_shared_append():
            return False
        return min_tokens is None or num_tokens >= min_tokens

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
        # Zero-kernel append fast path (ROCm AITER): write the shared slots
        # directly inside the top-k kernel, into persistent pre-allocated
        # buffers, instead of appending with arange/full/cat (4 tiny
        # launch-latency-bound kernels per layer per step). Falls back to the
        # cat-based append on other platforms/scoring funcs, when hashing, or
        # when the buffers cannot be grown (graph capture in progress).
        n = self.num_fused_shared_experts
        out_topk_weights = out_topk_ids = None
        fused_kwargs: dict[str, Any] = {}
        if (
            n > 0
            and self._append_shared_experts_for(hidden_states.size(0))
            and self.scoring_func == "sqrtsoftplus"
            and self._hash_indices_table is None
            and rocm_aiter_ops.is_fused_moe_enabled()
        ):
            bufs = _get_fused_shared_topk_buffers(
                num_tokens=hidden_states.size(0),
                topk=self.top_k,
                num_fused_shared=n,
                shared_expert_base=self.global_num_experts,
                device=hidden_states.device,
                indices_dtype=torch.int32 if indices_type is None else indices_type,
            )
            if bufs is not None:
                ids_buf, weights_buf = bufs
                m = hidden_states.size(0)
                out_topk_ids = ids_buf[:m]
                out_topk_weights = weights_buf[:m]
                fused_kwargs = {
                    "num_fused_shared_experts": n,
                    "shared_expert_base": self.global_num_experts,
                    "shared_expert_weight": self.shared_expert_weight,
                }

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
            out_topk_weights=out_topk_weights,
            out_topk_ids=out_topk_ids,
            **fused_kwargs,
        )

        # The in-kernel append already produced (M, top_k + n) outputs; the
        # width check detects any path that ignored the out buffers (e.g. a
        # dsv4_topk fast path) and appends with cat as before.
        if (
            self.num_fused_shared_experts > 0
            and self._append_shared_experts_for(topk_ids.shape[0])
            and topk_ids.shape[-1] == self.top_k
        ):
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
