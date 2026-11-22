# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amd_kernels on the Kimi-K3.1 DSA decode path (TP8 / DCP8, MXFP4 indexer).

Six kernels, each behind its own switch (``VLLM_KIMI_AKR_<NAME>=0`` turns one off, on
top of the route in ``akr_config``), each called through its shape-agnostic entry
(``attention/mla_decode_common``, ``indexer/mxfp4_indexer_common``,
``collectives/topk_gather_merge_common``) and falling back to the stock op sequence
when the entry's ``*_unsupported_reason`` (or a vLLM-side precondition) rejects it:

==============  ==========  ===================================================
switch          route       replaces
==============  ==========  ===================================================
PROLOGUE        attention   q_a/kv_a norms + KV insert (+ wk/wv GEMMs, key LN,
                            key store, weights) and the indexer query prep
SCORE_TOPK      indexer     ``score_paged`` + the top-2048 select
GATHER_MERGE    collectives DCP score all-gather + ``merge_dcp_scores``
QGATHER         attention   q_b + W_UK + cat + DCP query all-gather
SPARSE_MLA      attention   Triton ``sparse_mla_fwd`` (DCP decode); not at the
                            rows where the FlyDSL 96-head kernel is faster
COMBINE         attention   DCP LSE combine + W_UV + reduce-scatter (+ gate)
==============  ==========  ===================================================

The cross-rank kernels (GATHER_MERGE, QGATHER, COMBINE) spin on their peers: they
need every workgroup resident on every rank, so the two aux-stream-capable ones
(QGATHER on main, GATHER_MERGE on aux) are ordered by an event (see ``dsa.py``)
and never overlap each other or the COMBINE.
"""

import os
from typing import Any

import torch

from vllm.logger import init_logger

from . import akr_config

logger = init_logger(__name__)

# switch -> route
SWITCHES: dict[str, str] = {
    "PROLOGUE": "attention",
    "SCORE_TOPK": "indexer",
    "GATHER_MERGE": "collectives",
    "QGATHER": "attention",
    "SPARSE_MLA": "attention",
    "COMBINE": "attention",
}

# rows of the shared gathered-query buffer at most (``ensure_q_buffer``: the server's
# decode rows)
Q_BUFFER_ROWS = 256
# the sparse MLA kernel's partials feed the combine kernel, whose communicator takes
# 256 rows (c48 with 4 speculative tokens: 48 x 5 = 240); past them the stock decode
# + v_up-first combine run as a pair
MAX_ROWS_SPARSE = 256
# rows where the stock FlyDSL 96-head sparse MLA (VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA)
# beats the kernel: 43-85 by 0.3-0.6 us, 129-215 by 0.5-6 us a layer (MI355X, graph
# replay, cold inputs); the combine stays ours at every row count
FLYDSL_SPARSE_ROWS = ((43, 85), (129, 215))


def on(name: str) -> bool:
    """Whether kernel `name` runs: its route is on and its env switch is not 0."""
    if os.environ.get(f"VLLM_KIMI_AKR_{name}", "1").strip().lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        return False
    return akr_config.on(SWITCHES[name])


def _fallback(name: str, reason: str | None) -> bool:
    """True (and a one-time warning) when `name` must use the stock path."""
    return akr_config.unsupported_reason(
        SWITCHES[name], f"{name}: {reason}" if reason else None
    )


def _call_fallback(name: str, reason: str | None) -> bool:
    """`_fallback` of one call (an out-of-range size is not warned)."""
    return akr_config.call_unsupported(
        SWITCHES[name], f"{name}: {reason}" if reason else None
    )


def _meta(*shape, dtype) -> torch.Tensor:
    """Stand-in of an operand that does not exist yet, for a host-only reason check."""
    return torch.empty(shape, dtype=dtype, device="meta")


def _decode_shape() -> tuple[int, int] | None:
    """(rows, query rows a request) of the server's largest decode batch: `max_num_seqs`
    requests of `max_decode_query_len` (1 + k) rows, the builders' decode / prefill
    split; rows capped at 256. The decode-only kernels (score + top-k, query gather,
    combine) never see more rows, so their buffers are sized by it (the entries'
    reasons guard the bound). None outside a server config."""
    from vllm.config import get_current_vllm_config_or_none
    from vllm.v1.attention.backend import max_decode_query_len

    cfg = get_current_vllm_config_or_none()
    if cfg is None:
        return None
    q = max_decode_query_len(cfg)
    return max(1, min(256, cfg.scheduler_config.max_num_seqs * q)), q


# ---------------------------------------------------------------------------
# Communicators: one per process, created collectively at the first DSA layer.
# ---------------------------------------------------------------------------


class _Comms:
    gather_merge: Any = None
    qgather: Any = None
    combine: Any = None
    done = False


_comms = _Comms()


def _eligible_deployment() -> str | None:
    """The DSA multi-stream wiring: the DCP group is the TP group (the query gather
    and the combine exchange heads over it)."""
    from vllm.distributed import get_dcp_group, get_tensor_model_parallel_world_size

    dcp = get_dcp_group().world_size
    tp = get_tensor_model_parallel_world_size()
    if dcp == 1 or dcp != tp:
        return f"DCP world size {dcp}, TP {tp} (needs DCP = TP > 1)"
    return None


def init_comms(attn) -> None:
    """Create the communicators of the enabled kernels for the dims of `attn`
    (KimiK3DSAAttention; a collective on the DCP group: every rank calls it from the
    first DSA layer's __init__)."""
    if _comms.done:
        return
    _comms.done = True
    want = {
        n: on(n)
        for n in (
            "GATHER_MERGE",
            "QGATHER",
            "COMBINE",
            "PROLOGUE",
            "SCORE_TOPK",
            "SPARSE_MLA",
        )
    }
    if not any(want.values()):
        return
    if _fallback("GATHER_MERGE", _eligible_deployment()):
        return
    from vllm.distributed import get_dcp_group

    dcp = get_dcp_group()
    rank, group, device = dcp.rank_in_group, dcp.cpu_group, dcp.device
    # (model init runs under the model's default dtype; the hosts' buffers are fp32)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        _create_comms(want, rank, group, device, attn, dcp.world_size)
    finally:
        torch.set_default_dtype(prev)


def _create_comms(want, rank, group, device, attn, world) -> None:
    if want["GATHER_MERGE"]:
        from amd_kernels.collectives import topk_gather_merge_common as gm

        kw = dict(topk=attn.dsa_indexer.topk_tokens)
        if not _fallback(
            "GATHER_MERGE",
            gm.topk_gather_merge_comm_unsupported_reason(rank, group, device, **kw),
        ):
            _comms.gather_merge = gm.topk_gather_merge_comm(rank, group, device, **kw)
            _comms.gather_merge.warmup()
    if want["QGATHER"]:
        from amd_kernels.attention import mla_decode_common as mla

        kw = dict(
            q_lora=attn.q_lora_rank,
            q_b_rows=attn.num_local_heads
            * (attn.qk_nope_head_dim + attn.qk_rope_head_dim),
            heads=attn.num_local_heads * world,
        )
        if not _fallback(
            "QGATHER",
            mla.q_gather_comm_unsupported_reason(rank, group, device, "q", **kw),
        ):
            _comms.qgather = mla.q_gather_comm(rank, group, device, "q", **kw)
    if want["COMBINE"]:
        from amd_kernels.attention import mla_decode_common as mla

        shape = _decode_shape()
        kw = dict(
            heads=attn.num_local_heads * world,
            v_head=attn.v_head_dim,
            kv_lora=attn.kv_lora_rank,
            max_tokens=None if shape is None else shape[0],
            is_lse_base_on_e=True,
        )
        if not _fallback(
            "COMBINE",
            mla.lse_combine_comm_unsupported_reason(rank, group, device, **kw),
        ):
            _comms.combine = mla.lse_combine_comm(rank, group, device, **kw)
    logger.info(
        "amd_kernels DSA: comms gather_merge=%s qgather=%s combine=%s",
        _comms.gather_merge is not None,
        _comms.qgather is not None,
        _comms.combine is not None,
    )


def gather_merge_comm():
    return _comms.gather_merge


def qgather_comm():
    return _comms.qgather


def combine_comm():
    return _comms.combine


def prewarm(kernel_of, *args, **kwargs) -> None:
    """Compile every variant of the entry `kernel_of(*args, **kwargs)` routes these
    arguments to (an entry's ``*_kernel``), for the mode of this call, without
    launching anything (``amd_kernels.warm_up``)."""
    from amd_kernels import warm_up

    spec = kernel_of(*args, **kwargs).entry
    key = (spec.name, spec.mode(*args, **kwargs))
    if key not in warm_up.warmed():
        warm_up._warm(spec, key, args, kwargs)


# ---------------------------------------------------------------------------
# (2) fused indexer score + top-2048
# ---------------------------------------------------------------------------


def score_topk_supported(
    batch: int, width: int, cache: torch.Tensor, out_indices: torch.Tensor
) -> bool:
    """One-launch score + select for a whole decode batch, where it beats the
    scorer + top-k pair (``paged_score_topk_prefer``); `score_topk` then checks
    the call itself."""
    if not on("SCORE_TOPK") or batch < 1:
        return False
    from amd_kernels.indexer.mxfp4_indexer_common import paged_score_topk_prefer

    return paged_score_topk_prefer(batch, width, cache=cache, out_indices=out_indices)


def score_topk(
    q_payload,
    q_scales,
    weights,
    cache,
    block_table,
    starts,
    row_ends,
    max_query_len,
    out_indices,
    out_scores,
) -> bool:
    """Scores + top-k in one launch; False (nothing launched) when the entry does not
    serve the call (the stock scorer + select run)."""
    from amd_kernels.indexer import mxfp4_indexer_common as ix

    args = (
        q_payload,
        q_scales,
        weights,
        cache,
        block_table,
        starts,
        row_ends,
        max_query_len,
        out_indices,
        out_scores,
    )
    if _call_fallback(
        "SCORE_TOPK", ix.mxfp4_paged_score_topk_unsupported_reason(*args)
    ):
        return False
    if _zero_ready_counts():
        # The kernel's chunk workgroups count themselves into the ready counter of
        # every (sequence, row tile) group of the grid, but only groups with rows
        # have a merge workgroup that resets it: padded requests (no rows) and tiles
        # beyond a short sequence leave counts behind, and a later batch in which the
        # group has rows then passes its wait early (corrupt selections). Zero the
        # counters this launch uses.
        ix.paged_score_topk_reset_counts(
            starts, max_query_len, cache=cache, out_indices=out_indices
        )
    ix.mxfp4_paged_score_topk(*args)
    return True


def _zero_ready_counts() -> bool:
    return os.environ.get("VLLM_KIMI_AKR_SCORE_TOPK_ZERO_COUNTS", "1") != "0"


def preallocate_score_topk_scratch(w, device, cache, out_indices) -> None:
    """The kernel's scratch is one set per (device, stream) (0.59 MB a row): create it
    now, not inside a graph capture, for the server's decode rows.

    With the multi-stream wrapper only the aux stream's set is kept: every decode batch
    it runs (all the captured ones) launches the kernel there, and the set the warm-up's
    host calls made on the load stream is released (an eager serial decode on another
    stream makes its own set later, after the KV cache is sized). No set at all when no
    decode batch of the server is the kernel's (e.g. 8 query rows a request); a shorter
    batch (fewer drafts) that the kernel takes then makes a set of the decode rows."""
    from amd_kernels.indexer.mxfp4_indexer_common import paged_score_topk_preallocate

    from vllm.utils.torch_utils import aux_stream

    ms = getattr(w, "ms_comm", None) is not None
    shape = _decode_shape()
    kw = dict(cache=cache, out_indices=out_indices)
    if shape is not None and _decode_reason(w, cache, out_indices, shape[1]):
        paged_score_topk_preallocate(
            device, [], **kw, rows=shape[0], release_others=True
        )
        return
    streams = (
        [aux_stream()] if ms else [torch.cuda.current_stream(device), aux_stream()]
    )
    paged_score_topk_preallocate(
        device,
        streams,
        **kw,
        rows=None if shape is None else shape[0],
        release_others=ms,
    )


def _decode_reason(w, cache, out_indices, qlen: int) -> str | None:
    """The entry's reason for one request of `qlen` decode rows (warned once)."""
    from amd_kernels.indexer.mxfp4_indexer_common import (
        mxfp4_paged_score_topk_unsupported_reason,
    )

    d = _dims(w)
    z = lambda *s, dtype: torch.zeros(s, dtype=dtype, device=cache.device)  # noqa: E731
    reason = mxfp4_paged_score_topk_unsupported_reason(
        z(qlen, d["index_h"], d["index_d"] // 2, dtype=torch.uint8),
        z(qlen, 1, 4, 16, 4, dtype=torch.uint8),
        z(qlen, d["index_h"], dtype=torch.float32),
        cache,
        z(1, 1, dtype=torch.int32),
        z(2, dtype=torch.int32),
        z(qlen, dtype=torch.int32),
        qlen,
        z(qlen, out_indices.shape[1], dtype=torch.int32),
        z(qlen, out_indices.shape[1], dtype=torch.float32),
    )
    _fallback(
        "SCORE_TOPK", reason and f"decode batches of {qlen} rows a request: {reason}"
    )
    return reason


# ---------------------------------------------------------------------------
# (3) DCP score gather + merge
# ---------------------------------------------------------------------------


def gather_merge(
    scores: torch.Tensor,
    local_idx: torch.Tensor,
    token_pages: torch.Tensor,
    page_size: int,
    block_stride_rows: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(slots, indptr)`` as ``merge_dcp_scores`` returns them, from this rank's
    local top-k ``scores`` / ``local_idx``; None when the call is not the
    kernel's (the stock all-gather + merge run)."""
    gm = _comms.gather_merge
    if gm is None or not on("GATHER_MERGE"):
        return None
    from amd_kernels.collectives.topk_gather_merge_common import (
        topk_gather_merge_unsupported_reason,
    )

    rows = scores.shape[0]
    slots = torch.empty(
        rows * scores.shape[-1], dtype=torch.int32, device=scores.device
    )
    indptr = torch.empty(rows + 1, dtype=torch.int32, device=scores.device)
    args = (scores, local_idx, token_pages, slots, indptr, page_size, block_stride_rows)
    if _call_fallback("GATHER_MERGE", topk_gather_merge_unsupported_reason(gm, *args)):
        return None
    gm(*args)
    return slots, indptr


# ---------------------------------------------------------------------------
# (5) sparse MLA decode
# ---------------------------------------------------------------------------


def _flydsl_sparse_wins(q) -> bool:
    """The stock FlyDSL kernel serves this call (its packed, lse-returning variant of
    these heads is built) and is faster at its row count."""
    from vllm._aiter_ops import rocm_aiter_ops as ops

    rows = q.shape[0]
    return (
        any(lo <= rows <= hi for lo, hi in FLYDSL_SPARSE_ROWS)
        and ops._FLYDSL_SPARSE_MLA
        and ops._AITER_ENABLED
        and q.shape[1] not in ops._FLYDSL_SPARSE_MLA_FAILED
        and (q.shape[1], False, True) in ops._FLYDSL_SPARSE_MLA_COMPILED
    )


def sparse_mla_decode(q, kv_rows, indptr, slots, scale, out_dtype):
    """``(out, lse)`` of the DCP sparse MLA decode kernel, or None when the case
    is not the kernel's (stock Triton runs)."""
    if not on("SPARSE_MLA"):
        return None
    if out_dtype != torch.bfloat16 or q.shape[0] > MAX_ROWS_SPARSE:
        return None
    if _flydsl_sparse_wins(q):
        return None
    from amd_kernels.attention.mla_decode_common import (
        fp8_sparse_mla_decode,
        fp8_sparse_mla_decode_unsupported_reason,
    )

    args = (q, kv_rows, indptr, slots, scale)
    if _fallback("SPARSE_MLA", fp8_sparse_mla_decode_unsupported_reason(*args)):
        return None
    return fp8_sparse_mla_decode(*args)


# ---------------------------------------------------------------------------
# Per-layer weights and the static (model build) decisions
# ---------------------------------------------------------------------------


class LayerWeights:
    """What one DSA layer keeps for the kernels (built after weight loading)."""

    w_qb: torch.Tensor | None = None  # packed q_b weight [q_b rows, q_lora]
    w_uk96: torch.Tensor | None = None  # packed W_UK of all heads (of the DCP group)
    w_uv_t: torch.Tensor | None = None  # W_UV of all heads, [heads, v_head, kv_lora]
    prologue: bool = False
    extra_bytes: int = 0


def _unquantized_bf16(linear) -> bool:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    return (
        isinstance(getattr(linear, "quant_method", None), UnquantizedLinearMethod)
        and linear.weight.dtype == torch.bfloat16
    )


def _specialized_decode() -> bool:
    from vllm.config import get_current_vllm_config_or_none

    from .specialized_decode.runtime import AUTO, decode_backend

    cfg = get_current_vllm_config_or_none()
    return cfg is not None and decode_backend(cfg) != AUTO


def prologue_static_ok(w) -> bool:
    """Model-build decision for the prologue kernels on wrapper `w`."""
    if not on("PROLOGUE"):
        return False
    from amd_kernels.attention import mla_decode_common as mla

    ind = w.indexer
    why = _eligible_deployment()
    if why is None and _specialized_decode():
        # its D1a / K2 read the 2112-column qkv_a output
        why = "decode_backend=latency_specialized"
    if why is None:
        if not mla.available():
            why = "needs gfx950 with FlyDSL"
        elif (w.q_lora_rank, w.kv_lora_rank) not in mla.qkv_a_norm_kv_insert_shapes():
            why = (
                f"no qkv_a_norm_kv_insert for q_lora {w.q_lora_rank}, "
                f"kv_lora {w.kv_lora_rank}"
            )
        elif (
            not ind.use_fp4
            or (
                ind.index_num_heads,
                ind.index_head_dim,
            )
            not in mla.indexer_prep_shapes()
        ):
            why = "no indexer_prep for this indexer (MXFP4, heads x d)"
        elif not hasattr(ind.k_index_norm, "weight_bf16"):
            why = "indexer key norm is not the LayerNorm"
        elif not all(
            _unquantized_bf16(m)
            for m in (w.fused_qkv_a_proj, ind.wk_index, ind.wv_index)
        ):
            why = "qkv_a / wk_index / wv_index are not plain bf16"
    return not _fallback("PROLOGUE", why)


def qgather_static_ok(w) -> bool:
    if not on("QGATHER"):
        return False
    why = _eligible_deployment()
    if why is None:
        if qgather_comm() is None:
            why = "no query gather communicator"
        elif w.q_b_proj is None or not _unquantized_bf16(w.q_b_proj):
            why = "q_b_proj is not plain bf16"
        elif w.mla_attn.is_aiter_triton_fp4_bmm_enabled or (
            w.mla_attn.is_aiter_triton_fp8_bmm_enabled
        ):
            why = "W_UK is an AITER FP4 / FP8 bmm weight"
    return not _fallback("QGATHER", why)


def build_weights(w) -> LayerWeights:
    """After weight loading (end of ``MLAAttention.process_weights_after_loading``):
    the extended qkv_a weight (wk_index / wv_index rows appended; their own weights
    become views of it), the packed q_b and replicated W_UK, the transposed W_UV."""
    from amd_kernels.attention import mla_decode_common as mla

    from vllm.distributed import get_dcp_group

    lw = LayerWeights()
    layer = w.mla_attn
    ind = w.indexer
    if getattr(w, "akr_pro", False):
        qkv = w.fused_qkv_a_proj
        wk, wv = ind.wk_index.weight, ind.wv_index.weight
        cols = qkv.weight.shape[0]
        assert wk.shape[0] == ind.index_head_dim and wv.shape[0] == ind.index_num_heads
        ext = torch.cat([qkv.weight.data, wk.data, wv.data], dim=0).contiguous()
        qkv.weight.data = ext
        wk.data = ext[cols : cols + wk.shape[0]]
        wv.data = ext[cols + wk.shape[0] :]
        norm = ind.k_index_norm
        norm.weight_bf16 = norm.weight.data.to(torch.bfloat16)
        norm.bias_bf16 = norm.bias.data.to(torch.bfloat16)
        lw.prologue = True
    if getattr(w, "akr_qg", False) and getattr(layer, "W_UK_T", None) is not None:
        w_uk_all = get_dcp_group().all_gather(layer.W_UK_T.contiguous(), dim=0)
        try:
            lw.w_qb = mla.pack_q_b(w.q_b_proj.weight.data)
            lw.w_uk96 = mla.pack_w_uk(w_uk_all)
        except ValueError as e:
            lw.w_qb = lw.w_uk96 = None
            _fallback("QGATHER", str(e))
        del w_uk_all
        if lw.w_qb is not None and lw.w_uk96 is not None:
            lw.extra_bytes += lw.w_qb.numel() * 2 + lw.w_uk96.numel() * 2
    if (
        combine_comm() is not None
        and on("COMBINE")
        and getattr(layer, "W_UV", None) is not None
    ):
        old = getattr(layer, "W_UV_dcp", None)
        w_uv_all = (
            old.data
            if old is not None
            else get_dcp_group().all_gather(layer.W_UV.contiguous(), dim=0)
        )
        try:
            lw.w_uv_t = mla.prepare_w_uv(w_uv_all)
        except ValueError as e:
            _fallback("COMBINE", str(e))
            return lw
        if old is not None:
            # the stock v_up-first combine stays for M > 256 on a
            # view of the kernel's copy ([96, 512, 128] = w_uv_t^T: no second copy;
            # dcp_vup_bmm takes any strides and reads its transpose = w_uv_t)
            layer.W_UV_dcp = torch.nn.Parameter(
                lw.w_uv_t.transpose(1, 2), requires_grad=False
            )
        else:
            lw.extra_bytes += lw.w_uv_t.numel() * 2
    return lw


# ---------------------------------------------------------------------------
# Warm-up (compile only; nothing launches)
# ---------------------------------------------------------------------------

_warmed_layers = False


def _dummy_kv_cache(device) -> torch.Tensor:
    return torch.zeros((2, 64, 656), dtype=torch.uint8, device=device)


def _dummy_index_cache(device) -> torch.Tensor:
    return torch.zeros((2, 4352), dtype=torch.uint8, device=device).as_strided(
        (2, 64, 68), (4352, 68, 1)
    )


def _dims(w) -> dict[str, int]:
    """The DSA dims of wrapper `w` (heads: all heads of the DCP group)."""
    from vllm.distributed import get_dcp_group

    ind = w.indexer
    return dict(
        q_lora=w.q_lora_rank,
        kv_lora=w.kv_lora_rank,
        rope=w.qk_rope_head_dim,
        local_heads=w.num_heads,
        heads=w.num_heads * get_dcp_group().world_size,
        v_head=w.mla_attn.v_head_dim,
        index_h=ind.index_num_heads,
        index_d=ind.index_head_dim,
        topk=ind.topk_tokens,
    )


def warm_up(w, lw: LayerWeights) -> None:
    """Compile every variant of the enabled entries (first layer only)."""
    global _warmed_layers
    if _warmed_layers:
        return
    _warmed_layers = True
    from amd_kernels import warm_up as akr_warm_up
    from amd_kernels.attention import mla_decode_common as mla
    from amd_kernels.indexer import mxfp4_indexer_common as ix

    # eight ranks each start one compile worker per CPU by default
    akr_warm_up.set_workers(int(os.environ.get("VLLM_KIMI_AKR_WARM_WORKERS", "8")))
    dev = w.fused_qkv_a_proj.weight.device
    ind = w.indexer
    d = _dims(w)
    z = lambda *s, dtype: torch.zeros(s, dtype=dtype, device=dev)  # noqa: E731
    bf = torch.bfloat16
    if lw.prologue:
        cache = _dummy_index_cache(dev)
        index = dict(
            ln_weight=ind.k_index_norm.weight_bf16,
            ln_bias=ind.k_index_norm.bias_bf16,
            cache=cache,
            slot_mapping=z(1, dtype=torch.int64),
            weights=z(1, d["index_h"], dtype=torch.float32),
            weight_scale=1.0,
        )
        prewarm(
            mla.qkv_a_norm_kv_insert_kernel,
            z(1, w.fused_qkv_a_proj.weight.shape[0], dtype=bf),
            w.q_a_layernorm.weight.data,
            w.kv_a_layernorm.weight.data,
            1e-5,
            z(1, d["q_lora"], dtype=bf),
            z(1, d["kv_lora"], dtype=bf),
            _dummy_kv_cache(dev),
            z(1, dtype=torch.int64),
            index,
        )
        prewarm(
            mla.indexer_prep_kernel,
            z(1, d["index_h"], d["index_d"], dtype=bf),
            None,
            None,
            None,
            None,
            1e-5,
            None,
            cache,
            z(1, d["topk"], dtype=torch.int32),
            1.0,
            z(1, d["index_h"], d["index_d"] // 2, dtype=torch.uint8),
            z(1, 1, 4, 16, 4, dtype=torch.uint8),
            None,
        )
    if on("SCORE_TOPK") and ind.use_fp4:
        cache = _dummy_index_cache(dev)
        args: tuple = (
            z(1, d["index_h"], d["index_d"] // 2, dtype=torch.uint8),
            z(1, 1, 4, 16, 4, dtype=torch.uint8),
            z(1, d["index_h"], dtype=torch.float32),
            cache,
            z(1, 1, dtype=torch.int32),
            z(2, dtype=torch.int32),
            z(1, dtype=torch.int32),
            1,
            z(1, d["topk"], dtype=torch.int32),
            z(1, d["topk"], dtype=torch.float32),
        )
        if not _fallback(
            "SCORE_TOPK", ix.mxfp4_paged_score_topk_unsupported_reason(*args)
        ):
            prewarm(ix.mxfp4_paged_score_topk_kernel, *args)
            preallocate_score_topk_scratch(w, dev, cache, args[8])
    if on("SPARSE_MLA") and _eligible_deployment() is None:
        args = (
            z(1, d["heads"], d["kv_lora"] + d["rope"], dtype=bf),
            _dummy_kv_cache(dev)[0],
            z(2, dtype=torch.int32),
            z(1, dtype=torch.int32),
            1.0,
        )
        if not _fallback(
            "SPARSE_MLA", mla.fp8_sparse_mla_decode_unsupported_reason(*args)
        ):
            prewarm(mla.fp8_sparse_mla_decode_kernel, *args)
    if lw.w_qb is not None and qgather_comm() is not None:
        buf = ensure_q_buffer(w)
        prewarm(
            mla.q_bproj_wuk_gather_kernel,
            qgather_comm(),
            z(1, d["q_lora"], dtype=bf),
            lw.w_qb,
            lw.w_uk96,
            out=buf[:1],
        )
    if lw.w_uv_t is not None:
        out = z(1, d["local_heads"], d["v_head"], dtype=bf)
        for gate in (None, z(1, d["local_heads"] * d["v_head"], dtype=bf)):
            prewarm(
                mla.lse_combine_vup_kernel,
                combine_comm(),
                z(1, d["heads"], d["kv_lora"], dtype=bf),
                z(1, d["heads"], dtype=torch.float32),
                lw.w_uv_t,
                out,
                None,
                None,
                gate,
            )


# ---------------------------------------------------------------------------
# (1) prologue: K_A (norms + KV insert + indexer key/weights) and K_B (query prep)
# ---------------------------------------------------------------------------


def _index_side(w, ind_cache, ind_slots, weights) -> dict:
    ind = w.indexer
    return dict(
        ln_weight=ind.k_index_norm.weight_bf16,
        ln_bias=ind.k_index_norm.bias_bf16,
        cache=ind_cache,
        slot_mapping=ind_slots,
        weights=weights,
        weight_scale=ind.indexer_weight_scale * ind.index_softmax_scale,
    )


def prologue_runtime_ok(
    w,
    rows: int,
    table_width: int,
    kv_cache: torch.Tensor,
    kv_slots,
    ind_cache: torch.Tensor,
    ind_slots,
) -> bool:
    """Per step (decided at capture, before the operands exist: the entries' reasons
    run on stand-ins of them and the real caches / slot mappings): the single indexer
    tile the prepped query path needs. `table_width`: keys of the indexer block
    table."""
    from amd_kernels.attention import mla_decode_common as mla

    from .ops.fp4_indexer import decode_tile_size

    lw = w.akr_w
    if lw is None or not lw.prologue or not on("PROLOGUE") or rows <= 0:
        return False
    d = _dims(w)
    bf = torch.bfloat16
    why = None
    if kv_slots is None or ind_slots is None:
        why = "no slot mapping"
    elif (
        decode_tile_size(w.indexer.max_model_len, table_width, d["topk"], d["index_h"])
        < rows
    ):
        why = "decode batch spans several indexer score tiles"
    else:
        why = mla.qkv_a_norm_kv_insert_unsupported_reason(
            _meta(rows, w.fused_qkv_a_proj.weight.shape[0], dtype=bf),
            w.q_a_layernorm.weight.data,
            w.kv_a_layernorm.weight.data,
            w.q_a_layernorm.variance_epsilon,
            _meta(rows, d["q_lora"], dtype=bf),
            _meta(rows, d["kv_lora"], dtype=bf),
            kv_cache,
            kv_slots,
            _index_side(
                w, ind_cache, ind_slots, _meta(rows, d["index_h"], dtype=torch.float32)
            ),
        ) or mla.indexer_prep_unsupported_reason(
            _meta(rows, d["index_h"], d["index_d"], dtype=bf),
            None,
            None,
            None,
            None,
            float(w.indexer.k_index_norm.eps),
            None,
            ind_cache,
            _meta(rows, d["topk"], dtype=torch.int32),
            1.0,
            _meta(rows, d["index_h"], d["index_d"] // 2, dtype=torch.uint8),
            _meta(rows, 1, 4, 16, 4, dtype=torch.uint8),
            None,
        )
    return not _fallback("PROLOGUE", why)


def prologue_a(w, qkv, kv_cache, kv_slots, ind_cache, ind_slots):
    """K_A: ``(q_c, kv_c_normed, index weights)`` and the KV + indexer key inserts."""
    from amd_kernels.attention.mla_decode_common import qkv_a_norm_kv_insert

    m = qkv.shape[0]
    q_c = torch.empty((m, w.q_lora_rank), dtype=qkv.dtype, device=qkv.device)
    kv_c = torch.empty((m, w.kv_lora_rank), dtype=qkv.dtype, device=qkv.device)
    idx_w = torch.empty(
        (m, w.indexer.index_num_heads), dtype=torch.float32, device=qkv.device
    )
    qkv_a_norm_kv_insert(
        qkv,
        w.q_a_layernorm.weight.data,
        w.kv_a_layernorm.weight.data,
        w.q_a_layernorm.variance_epsilon,
        q_c,
        kv_c,
        kv_cache,
        kv_slots,
        _index_side(w, ind_cache, ind_slots, idx_w),
    )
    return q_c, kv_c, idx_w


def prologue_b(w, index_q, topk, ind_cache):
    """K_B: MXFP4 query payload / scales and the top-k reset (aux stream)."""
    from amd_kernels.attention.mla_decode_common import indexer_prep

    ind = w.indexer
    m, heads, d = index_q.shape
    q_payload = torch.empty(
        (m, heads, d // 2), dtype=torch.uint8, device=index_q.device
    )
    q_scales = torch.empty((m, 1, 4, 16, 4), dtype=torch.uint8, device=index_q.device)
    indexer_prep(
        index_q,
        None,
        None,
        None,
        None,
        float(ind.k_index_norm.eps),
        None,
        ind_cache,
        topk[:m],
        ind.indexer_weight_scale * ind.index_softmax_scale,
        q_payload,
        q_scales,
        None,
    )
    return q_payload, q_scales


# ---------------------------------------------------------------------------
# (4) q_b + W_UK + DCP query gather
# ---------------------------------------------------------------------------

_q_buffer: torch.Tensor | None = None


def ensure_q_buffer(w) -> torch.Tensor:
    """The one gathered-query buffer [decode rows, heads, kv_lora + rope] shared by all
    layers (a layer's query is consumed by its sparse MLA before the next layer's
    gather); the gather runs on decode batches only."""
    global _q_buffer
    if _q_buffer is None:
        d = _dims(w)
        shape = _decode_shape()
        _q_buffer = torch.empty(
            (
                Q_BUFFER_ROWS if shape is None else shape[0],
                d["heads"],
                d["kv_lora"] + d["rope"],
            ),
            dtype=torch.bfloat16,
            device=w.fused_qkv_a_proj.weight.device,
        )
    return _q_buffer


def qgather_runtime_ok(w, rows: int) -> bool:
    """Per step (decided at capture, on a stand-in of ``q_c``)."""
    lw = w.akr_w
    comm = qgather_comm()
    if (
        lw is None
        or not on("QGATHER")
        or lw.w_qb is None
        or comm is None
        or not 0 < rows <= ensure_q_buffer(w).shape[0]
    ):
        return False
    from amd_kernels.attention.mla_decode_common import (
        q_bproj_wuk_gather_unsupported_reason,
    )

    reason = q_bproj_wuk_gather_unsupported_reason(
        comm,
        _meta(rows, w.q_lora_rank, dtype=torch.bfloat16),
        lw.w_qb,
        lw.w_uk96,
        out=ensure_q_buffer(w)[:rows],
    )
    return not _call_fallback("QGATHER", reason)


def qgather(w, q_c: torch.Tensor) -> torch.Tensor:
    """The latent query of all heads, [rows, heads, kv_lora + rope] (view of the shared
    buffer)."""
    from amd_kernels.attention.mla_decode_common import q_bproj_wuk_gather

    buf = ensure_q_buffer(w)[: q_c.shape[0]]
    q_bproj_wuk_gather(qgather_comm(), q_c, w.akr_w.w_qb, w.akr_w.w_uk96, out=buf)
    return buf


# ---------------------------------------------------------------------------
# (6) LSE combine + W_UV (+ output gate)
# ---------------------------------------------------------------------------


def combine(layer, attn_out, lse, out3d, seq_lens, query_start_loc) -> bool:
    """Write this rank's combined, up-projected heads into `out3d` ([rows, local heads,
    v_head]); with a pending DSA aux-stream gate also multiply by sigmoid(gate). False
    when the case is not the kernel's (the stock combine runs)."""
    comm = combine_comm()
    w_uv_t = getattr(layer, "akr_w_uv_t", None)
    if comm is None or w_uv_t is None or not on("COMBINE") or seq_lens is None:
        return False
    from amd_kernels.attention.mla_decode_common import (
        lse_combine_vup,
        lse_combine_vup_unsupported_reason,
    )

    rows = attn_out.shape[0]
    args = (comm, attn_out, lse, w_uv_t, out3d, seq_lens, query_start_loc)
    gate = None
    st = getattr(layer, "_akr_ms_state", None)
    if st is not None and st.gate is not None and st.gate.shape[0] >= rows:
        gate = st.gate[:rows]
        if lse_combine_vup_unsupported_reason(*args, gate) is not None:
            gate = None
    if _call_fallback("COMBINE", lse_combine_vup_unsupported_reason(*args, gate)):
        return False
    if gate is not None:
        assert st is not None  # gate is only set from st
        # join the aux stream (g_proj) before the launch
        torch.cuda.current_stream().wait_event(st.ev_done)
        st.joined = True
    lse_combine_vup(*args, gate)
    if gate is not None and st is not None:
        st.folded = True
    return True
