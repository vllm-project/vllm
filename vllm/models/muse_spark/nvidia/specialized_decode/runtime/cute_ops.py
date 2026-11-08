# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark kernels the specialized decode and the DSpark draft call (the
CuTe DSL kernels of cutedsl/attention/ and cutedsl/moe/routing.py), with the
checks that decide where each one runs.

``attn2`` (``cutedsl/attention/attn2.py``)
    Paged decode attention of the 8-token verify step (one request, 1 + 7 draft
    tokens) fused with the attention output's weightless RMS norm and sigmoid
    gate: the o_proj input of a decoder layer. Contract: head_dim 64, 8 query
    heads per KV head, bf16 [pages, num_kv_heads, 16, 128] cache (K | V per
    slot), 8 query rows per request, any context length L >= 8, and the
    layer's sliding window if it has one. One launch covers the KV heads of
    the rank. With max_model_len past 2048 (or a window) a split kernel
    spreads a request's pages over ``attn2_splits`` z slices, merged through a
    workspace. The launch shape depends on max_model_len only, not on the
    step, so it is graph-capturable once ``prepare_attn2`` has compiled it.

    attn2 has no fallback of its own: the executor's plan() leaves every step
    it does not cover (prefill, other query lengths) to the stock forward.

``MuseRouting`` (``cutedsl/moe/routing.py``)
    Muse Spark routing for decode batches of at most 8 tokens. It writes, bit
    for bit, the expert ids and weights ``fused_musespark1x_routing`` writes
    for Muse Spark's router: sigmoid scores, the correction bias for the
    selection only, top-8 of 256 experts, weights
    ``score * route_scale / (sum + 1e-15)``. The router's output multiplier
    must already be folded into its weight, as ``fold_router_output_multiplier``
    does after loading. Optionally the same launch writes the TRT-LLM-gen
    routing metadata of this rank's experts for one GEMM tile
    (``RoutingMetadata``): the CTA map, the permutation and the padded size the
    routed experts' GEMMs read, as the routed MoE op's own routing kernel
    would write them for these ids.

``qkv_front`` (``cutedsl/attention/qkv_front.py``)
    For 1 to 16 tokens on the SM100 family the QKV front CuTe kernel replaces
    the QKV + o-gate projection GEMM and
    ``fused_muse_spark_qknorm_rope_kv_insert``. A served call is checked once
    (``supports_call``) and then launched (``launch_qkv_front``). Called
    eagerly (Muse Spark runs uncompiled), so the ``data_ptr()`` checks are not
    traced.
"""

from typing import NamedTuple

import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.muse_spark.common.moe import MuseSpark1xRouter
from vllm.platforms import current_platform

from ..cutedsl.attention import attn2 as _attn2
from ..cutedsl.attention import qkv_front as _front
from ..cutedsl.common.host import fp32
from ..cutedsl.moe import routing

# Both attn2 and the QKV front read head_dim 64 and 16-slot KV cache pages.
HEAD_DIM = _attn2.HEAD_DIM
BLOCK_SIZE = _attn2.PAGE

# attn2
ATTN2_KERNEL = "the attn2 CuTe kernel"
HEADS_PER_KV_HEAD = _attn2.HEADS
QUERY_LEN = _attn2.QUERY_ROWS
# Up to this max_model_len without a sliding window, the one-cluster kernel;
# beyond, the split kernel.
SHORT_MAX_LEN = _attn2.SHORT_MAX_LEN
# 16-row groups of a KV head's 8 query rows x 8 heads: clusters per z slice.
ROW_GROUPS = _attn2.GROUPS
# Clusters a long-context launch aims for per request. A constant, so the split
# (and the result) does not depend on the device.
SPLIT_TARGET_CLUSTERS = 16
MAX_SPLITS = _attn2.MAX_SPLITS
# The kernel's softmax scale (head_dim ** -0.5 * qk_scale_factor) and o-norm
# eps: the kernel has no arguments for them, so a layer with other values is
# refused.
SM_SCALE = _attn2.SM_SCALE
ATTN2_EPS = _attn2.EPS
attn2 = _attn2.attn2
prepare_attn2 = _attn2.prepare
# Whether attn2 launches the split kernel (which needs a workspace).
attn2_split_kernel = _attn2.split_kernel

# Routing short tail
ROUTING_KERNEL = "the routing CuTe kernel"
NUM_EXPERTS = routing.EXPERTS
TOP_K = routing.TOP_K
ROUTING_MAX_TOKENS = routing.MAX_TOKENS
# The division epsilon the kernel adds to the sum of the selected scores; Muse
# Spark builds its router with this value.
DIVISION_EPS = routing.DIVISION_EPS
# The routing metadata: int32 offsets of its fields.
META_PADDED_TOKENS = routing.META_PADDED_TOKENS
META_NUM_CTAS = routing.META_NUM_CTAS
META_EXPANDED_TO_PERMUTED = routing.META_EXPANDED_TO_PERMUTED
META_CTA_BATCH = routing.META_CTA_BATCH
META_CTA_LIMIT = routing.META_CTA_LIMIT
META_PERMUTED_TO_TOKEN = routing.META_PERMUTED_TO_TOKEN
METADATA_MAX_TILE = routing.MAX_TILE
# int32 elements of the metadata for GEMM tile tile_n: the fields above and the
# permuted-to-token map (64 CTAs of tile_n rows, plus a guard element).
metadata_size = routing.metadata_size

# QKV front
QKV_FRONT_KERNEL = "the QKV front CuTe kernel"
HIDDEN_SIZE = _front.HIDDEN
k_split = _front.k_split
supports_shape = _front.supports_shape
kv_cache_layout_ok = _front.kv_cache_layout_ok
prepare_qkv_front = _front.prepare
qkv_front = _front.qkv_front
# qkv_front without its checks, for a call supports_call took with its out.
launch_qkv_front = _front.launch


def sm100_family() -> bool:
    """Whether this is an SM100-family GPU, which each of the kernels needs
    (clusters of up to 16 CTAs, TMA, st.async, PDL)."""
    return current_platform.is_cuda() and current_platform.is_device_capability_family(
        100
    )


# Each provider asks for its own kernels.
attn2_available = qkv_front_available = routing_available = sm100_family


def unavailable(kernel: str) -> str:
    """The reason a deployment whose GPUs cannot run ``kernel`` is refused."""
    return f"GPUs outside the SM100 family ({kernel})"


def unsupported_attn2(layer: torch.nn.Module) -> str | None:
    """Why the kernel cannot compute this MuseSpark1xAttention layer, or None."""
    if not (layer.use_o_norm and layer.use_o_gate):
        return "the layer has no o-norm with a sigmoid o-gate"
    if layer.head_dim != HEAD_DIM:
        return f"head_dim {layer.head_dim} (the kernel has {HEAD_DIM})"
    if layer.num_heads != HEADS_PER_KV_HEAD * layer.num_kv_heads:
        return (
            f"{layer.num_heads} query heads for {layer.num_kv_heads} KV heads "
            f"(the kernel has {HEADS_PER_KV_HEAD} per KV head)"
        )
    if fp32(layer.scaling) != fp32(SM_SCALE):
        return f"softmax scale {layer.scaling} (the kernel is built for {SM_SCALE})"
    if fp32(layer.rms_norm_eps) != fp32(ATTN2_EPS):
        return f"o-norm eps {layer.rms_norm_eps} (the kernel is built for {ATTN2_EPS})"
    return None


def attn2_window(sliding_window: int | None, max_model_len: int) -> int:
    """The window attn2 applies for a layer: 0 (none) when the layer has none
    or the window covers max_model_len (no key is then outside a row's
    window)."""
    if sliding_window is None or sliding_window >= max_model_len:
        return 0
    if sliding_window < 1:
        raise ValueError(f"sliding window {sliding_window}")
    return sliding_window


def attn2_splits(max_model_len: int, window: int, num_kv_heads: int) -> int:
    """The z slices of an attn2 launch (ROW_GROUPS clusters each per request
    and KV head) for a deployment: 1 for the one-cluster kernel and for a
    window (it spans at most window + 7 keys), else about
    SPLIT_TARGET_CLUSTERS clusters per request over the rank's KV heads."""
    if window or max_model_len <= SHORT_MAX_LEN:
        return 1
    fill = SPLIT_TARGET_CLUSTERS // (ROW_GROUPS * num_kv_heads)
    return max(1, min(fill, MAX_SPLITS))


def attn2_workspace_bytes(num_reqs: int, num_kv_heads: int, num_splits: int) -> int:
    """Bytes of the workspace the split kernel needs for ``num_reqs``
    requests: the counters (16-byte padded) and a partial per row and
    cluster."""
    rows = num_reqs * QUERY_LEN * HEADS_PER_KV_HEAD * num_kv_heads
    return _attn2.workspace_bytes(rows, num_splits)


def split_qkv(
    qkv: torch.Tensor, num_kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The q and gate column views of the attention front's output rows
    ``[q | k | v | gate]`` (q and gate: 8 heads of 64 per KV head)."""
    q_width = HEADS_PER_KV_HEAD * num_kv_heads * HEAD_DIM
    kv_width = num_kv_heads * HEAD_DIM
    if qkv.dim() != 2 or qkv.shape[1] != 2 * q_width + 2 * kv_width:
        raise ValueError(
            f"qkv must be [tokens, {2 * q_width + 2 * kv_width}] for "
            f"{num_kv_heads} KV heads, got {tuple(qkv.shape)}"
        )
    return qkv[:, :q_width], qkv[:, q_width + 2 * kv_width :]


def unsupported_router(router: MuseSpark1xRouter) -> str | None:
    """Why the kernel cannot reproduce ``router``'s routing, or None if it
    can."""
    if router.router_score_normalization != "sigmoid":
        return f"{router.router_score_normalization} router scores"
    if router.top_k != TOP_K or router.global_num_experts != NUM_EXPERTS:
        return (
            f"top-{router.top_k} of {router.global_num_experts} experts "
            f"(the kernel routes top-{TOP_K} of {NUM_EXPERTS})"
        )
    if router.router_output_multiplier is not None:
        return "a router output multiplier not folded into the router weight"
    if router.router_score_division_eps != DIVISION_EPS:
        return f"router_score_division_eps {router.router_score_division_eps}"
    bias = router.expert_bias
    if bias.dtype != torch.float32 or not bias.is_contiguous():
        return f"a {bias.dtype} expert bias (the kernel reads contiguous fp32)"
    if bias.data_ptr() % 16:
        return "an expert bias not 16-byte aligned (the kernel reads it in 16 bytes)"
    return None


class RoutingMetadata(NamedTuple):
    """Where the routing writes the routing metadata of the local experts
    ``[local_expert_offset, local_expert_offset + num_local_experts)`` for GEMM
    tile ``tile_n``: ``buffer``, int32 of at least ``metadata_size(tile_n)``
    elements, not aliasing the ids or weights."""

    buffer: torch.Tensor
    local_expert_offset: int
    num_local_experts: int
    tile_n: int

    def fields(self, num_tokens: int) -> dict[str, torch.Tensor]:
        """The fields of ``num_tokens`` tokens, as 1-D int32 views of
        ``buffer`` at the extents the routed MoE op allocates for them (with
        8 num_tokens <= 256 routes, at most one CTA per route):
        total_num_padded_tokens [1], num_non_exiting_ctas [1],
        expanded_idx_to_permuted_idx [8 num_tokens] (-1 for another rank's
        expert), cta_idx_xy_to_batch_idx and cta_idx_xy_to_mn_limit
        [8 num_tokens], and permuted_idx_to_token_idx [8 num_tokens tile_n + 1]
        (-1 for a padding row; the last element is the guard the GEMMs may
        read, never written)."""
        if not 1 <= num_tokens <= ROUTING_MAX_TOKENS:
            raise ValueError(f"1 to {ROUTING_MAX_TOKENS} tokens, got {num_tokens}")
        m = self.buffer
        routes = num_tokens * TOP_K
        return dict(
            total_num_padded_tokens=m[META_PADDED_TOKENS : META_PADDED_TOKENS + 1],
            num_non_exiting_ctas=m[META_NUM_CTAS : META_NUM_CTAS + 1],
            expanded_idx_to_permuted_idx=m[
                META_EXPANDED_TO_PERMUTED : META_EXPANDED_TO_PERMUTED + routes
            ],
            cta_idx_xy_to_batch_idx=m[META_CTA_BATCH : META_CTA_BATCH + routes],
            cta_idx_xy_to_mn_limit=m[META_CTA_LIMIT : META_CTA_LIMIT + routes],
            permuted_idx_to_token_idx=m[
                META_PERMUTED_TO_TOKEN : META_PERMUTED_TO_TOKEN
                + routes * self.tile_n
                + 1
            ],
        )


class MuseRouting:
    """The routing of one Muse Spark MoE layer for 1 to 8 tokens, bound to its
    router's bias and route scale at build time, when the kernel compiles. A
    call allocates nothing, so it can be captured in a CUDA graph."""

    def __init__(self, router: MuseSpark1xRouter):
        reason = unsupported_router(router)
        if reason is not None:
            raise ValueError(f"Muse Spark routing does not support {reason}")
        self.bias = router.expert_bias
        self.route_scale = float(router.route_scale)
        routing.prepare()

    def __call__(
        self,
        logits: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        metadata: RoutingMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route fp32 router logits [b, 256] (1 <= b <= 8, 16-byte aligned
        rows) into ``ids`` (int32 [b, 8], global expert ids) and ``weights``
        (fp32 [b, 8]), and with ``metadata`` the routing metadata of its local
        experts; returns (ids, weights). Raises ValueError on arguments outside
        ``routing.route``'s contract."""
        if metadata is None:
            routing.route(logits, self.bias, ids, weights, self.route_scale)
        else:
            routing.route(
                logits,
                self.bias,
                ids,
                weights,
                self.route_scale,
                metadata=metadata.buffer,
                local_expert_offset=metadata.local_expert_offset,
                num_local_experts=metadata.num_local_experts,
                tile_n=metadata.tile_n,
            )
        return ids, weights


def unsupported_projection(attn: torch.nn.Module) -> str | None:
    """Why the QKV front cannot read this Muse Spark attention layer's loaded
    QKV projection and rotary table, or None."""
    proj = attn.qkv_proj
    weight = getattr(proj, "weight", None)
    n = 2 * (attn.num_heads + attn.num_kv_heads) * attn.head_dim
    if not (
        type(proj.quant_method) is UnquantizedLinearMethod
        and proj.bias is None
        and not (proj.gather_output and proj.tp_size > 1)
        and supports_shape(n, attn.num_kv_heads)
        and weight is not None
        and _front.hidden_rows_ok(weight)
        and weight.shape[0] == n
    ):
        return f"a QKV projection other than an unquantized bf16 [N, {HIDDEN_SIZE}]"
    emb = attn.rotary_emb
    if not attn.use_rope:
        return None if emb is None else "a rotary table on a layer without RoPE"
    cos_sin = getattr(emb, "cos_sin_cache", None)
    if not (
        emb is not None
        and emb.rotary_dim == attn.head_dim
        and not emb.is_neox_style
        and cos_sin is not None
        and cos_sin.dtype == torch.float32
        and cos_sin.dim() == 2
        and cos_sin.shape[1] == attn.head_dim
        and cos_sin.is_contiguous()
    ):
        return "a rotary table other than an interleaved fp32 one"
    return None


def supports_call(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    out: torch.Tensor | None = None,
) -> bool:
    """Whether the QKV front takes this call of a layer with ``num_kv_heads``
    (with ``out``, the full check before ``launch_qkv_front``); otherwise the
    caller keeps the unfused path."""
    return (
        _front.unsupported(
            x, weight, positions, cos_sin_cache, slot_mapping, kv_cache, out
        )
        is None
        and kv_cache.shape[1] == num_kv_heads
    )
