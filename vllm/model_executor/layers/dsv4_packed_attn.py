# SPDX-License-Identifier: Apache-2.0
"""Head-packed sparse attention for DeepSeek-V4 C128A prefill layers.

At TP8 DeepSeek-V4 has 8 real heads per rank, but FlashMLA's sparse prefill
kernel only accepts h_q=64, so the stock path pads q with 56 zero heads and
throws away 7/8 of every tensor-core tile. For C128A layers the per-token
index row is structurally

    [0, n_c)                    compressed prefix, shared by nearby tokens
    [w, w + swa)                a contiguous sliding window

and n_c is constant over runs of exactly 128 query tokens, so 16 adjacent
tokens x 8 real heads pack into ONE 128-row tile whose union index list is
barely longer than a single token's (measured blow-up 1.002x). The packed
kernel takes two per-query ranges instead of a bitmask, which is what makes
the union exact: query g attends union ranks [0, pref_g) U [lo_g, hi_g).

This module derives those ranges directly from positions -- the same inputs
combine_topk_swa_indices consumes -- so the [num_tokens, topk+window] index
matrix is never materialized, and the result is cached per (metadata, chunk)
because it depends only on positions, not on the layer.

Enable with VLLM_DSV4_PACKED_ATTN=1 and point VLLM_DSV4_PACKED_SO at the
separately built kernel module. Any shape the packed path cannot serve
exactly, or a missing module, falls back to the stock kernel.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

B_H = 128  # packed rows per tile -- fixed by the kernel
B_TOPK = 64  # kernel index-width granularity
MAX_G = 16  # kernel limit: at most 16 queries share a tile

_ENABLED = os.environ.get("VLLM_DSV4_PACKED_ATTN", "0") == "1"
# This experimental TVM-FFI module is not part of the vLLM wheel build. Make
# the artifact choice explicit instead of probing an untracked source-tree
# file that cannot exist in a standard wheel installation.
_SO = os.environ.get("VLLM_DSV4_PACKED_SO", "")
_MOD = None
_FN = None
_FAILED = False
_SCRATCH: dict = {}
_EXECUTION_LOGGED = False


_CHECK = os.environ.get("VLLM_DSV4_PACKED_CHECK", "0") == "1"
_CHECK_STATS = {"n": 0, "worst": 0.0}


def enabled() -> bool:
    return _ENABLED


def checking() -> bool:
    """Run the stock path too and compare (validation only; halves speed)."""
    return _CHECK


def _is_sm100a_device(device: torch.device) -> bool:
    """Return whether ``device`` can run the sm_100a-only packed kernel."""
    return (
        torch.cuda.is_available()
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) == (10, 0)
    )


def report_check(packed, stock, n_local_heads):
    """Compare a packed result against the stock kernel's, in place."""
    if packed is None:
        return
    h = int(n_local_heads)
    packed_real = packed[:, :h].float()
    stock_real = stock[:, :h].float()
    if not (torch.isfinite(packed_real).all() and torch.isfinite(stock_real).all()):
        raise RuntimeError("DSv4 packed-attention check found non-finite output")
    d = (packed_real - stock_real).abs()
    scale = stock_real.abs().max().clamp(min=1e-6)
    rel = float((d.max() / scale).item())
    _CHECK_STATS["n"] += 1
    if rel > _CHECK_STATS["worst"]:
        _CHECK_STATS["worst"] = rel
    if _CHECK_STATS["n"] % 20 == 0 or rel > 0.05:
        logger.info(
            "[dsv4-packed] check n=%d rel=%.5f worst=%.5f maxabs=%.5f",
            _CHECK_STATS["n"],
            rel,
            _CHECK_STATS["worst"],
            float(d.max().item()),
        )


def _fn():
    """Load the packed-attention .so once per process."""
    global _MOD, _FN, _FAILED
    if _FN is None and not _FAILED:
        if not _SO:
            _FAILED = True
            logger.warning(
                "DSv4 packed attention requested without "
                "VLLM_DSV4_PACKED_SO; falling back to the stock kernel"
            )
            return None
        try:
            import tvm_ffi

            _MOD = tvm_ffi.load_module(_SO)
            _FN = _MOD["dsv4_masked_mla_bf16"]
            logger.info("DSv4 packed attention loaded from %s", _SO)
        except Exception as e:  # noqa: BLE001
            _FAILED = True
            logger.warning(
                "DSv4 packed attention unavailable (%s); "
                "falling back to the stock kernel",
                e,
            )
    return _FN


def _build_ranges(
    query_start_loc,
    seq_lens,
    gather_lens,
    chunk_M,
    chunk_N,
    window_size,
    compress_ratio,
    top_k,
    num_tokens,
    G,
):
    """Union list + per-query ranges for one chunk, from positions alone.

    Mirrors _combine_topk_swa_indices_kernel exactly: token at absolute
    position ``pos`` of request ``b`` attends compressed entries
    ``M*b + [0, min((pos+1)//ratio, top_k))`` and window entries starting at
    ``M*b + N + pos - swa + 1 - (seq_len - gather_len)``.
    """
    dev = seq_lens.device
    qsl = query_start_loc - query_start_loc[0]
    query_len = qsl[1:] - qsl[:-1]  # [num_reqs]
    start_pos = seq_lens - query_len

    tok = torch.arange(num_tokens, dtype=torch.int64, device=dev)
    # request id per token (chunk-local)
    b = torch.searchsorted(qsl[1:].to(torch.int64), tok, right=True)
    pos = start_pos.to(torch.int64)[b] + (tok - qsl.to(torch.int64)[b])

    topk_len = torch.clamp(
        torch.div(pos + 1, compress_ratio, rounding_mode="floor"), max=top_k
    )
    swa_len = torch.clamp(pos + 1, max=window_size)
    gather_start = (seq_lens - gather_lens).to(torch.int64)[b]
    base = chunk_M * b
    win_start = base + chunk_N + pos - swa_len + 1 - gather_start

    ng = num_tokens // G
    pl = topk_len.view(ng, G)
    ws = win_start.view(ng, G)
    we = ws + swa_len.view(ng, G)
    P = pl.max(1).values  # [ng]
    wlo = ws.min(1).values
    whi = we.max(1).values
    ulen = P + (whi - wlo)
    # Avoid a GPU->host synchronization while planning each new chunk. A
    # group of G adjacent queries can span at most ``window_size + G - 1``
    # window entries, in addition to the bounded compressed prefix.
    cap_bound = top_k + window_size + G - 1
    cap = (cap_bound + B_TOPK - 1) // B_TOPK * B_TOPK

    # union rank -> workspace slot: prefix first (offset by this request's
    # base), then the shared window span.
    gbase = base.view(ng, G)[:, 0].view(-1, 1)
    rank = torch.arange(cap, dtype=torch.int64, device=dev).view(1, cap)
    u = torch.where(
        rank < P.view(-1, 1), gbase + rank, wlo.view(-1, 1) + (rank - P.view(-1, 1))
    )
    u = (
        torch.where(rank < ulen.view(-1, 1), u, torch.full_like(u, -1))
        .to(torch.int32)
        .contiguous()
    )

    pref = pl.to(torch.int32).contiguous()
    w_lo = (P.view(-1, 1) + (ws - wlo.view(-1, 1))).to(torch.int32).contiguous()
    w_hi = (w_lo + swa_len.view(ng, G).to(torch.int32)).contiguous()
    counts = ulen.to(torch.int32).contiguous()
    return u.view(ng, 1, cap), pref, w_lo, w_hi, counts


def try_packed_prefill(
    *,
    q,
    kv,
    out,
    attn_sink,
    sm_scale,
    query_start_loc,
    seq_lens,
    gather_lens,
    chunk_M,
    chunk_N,
    window_size,
    compress_ratio,
    top_k,
    n_local_heads,
    cache_owner,
    cache_key,
):
    """Run one C128A prefill chunk on the packed kernel.

    Returns True when it ran; False means the caller must use the stock path.
    ``q``/``out`` may be padded [num_tokens, 64, 512] views; only the first
    ``n_local_heads`` entries carry data.
    """
    if not _is_sm100a_device(q.device):
        return False

    fn = _fn()
    if fn is None:
        return False
    num_tokens = q.shape[0]
    num_reqs = seq_lens.shape[0]
    # One tile is B_H rows = G query tokens x n_local_heads real heads, so the
    # packing factor follows the TP degree: G = 128 / heads-per-rank. Anything
    # that does not tile the 128 rows exactly (or would need more than the
    # kernel's 16 queries per tile) falls back.
    h = int(n_local_heads)
    if h <= 0 or B_H % h != 0:
        return False
    G = B_H // h
    if G > MAX_G:
        return False
    # Groups must not straddle a request (each request has its own workspace
    # base offset).
    if num_reqs != 1 or num_tokens % G != 0 or num_tokens == 0:
        return False
    if q.shape[-1] != 512 or kv.shape[-1] != 512:
        return False

    ng = num_tokens // G
    cached = getattr(cache_owner, "_dsv4_packed_cache", None)
    if cached is None:
        cached = {}
        cache_owner._dsv4_packed_cache = cached
    plan = cached.get(cache_key)
    if plan is None:
        plan = _build_ranges(
            query_start_loc,
            seq_lens,
            gather_lens,
            chunk_M,
            chunk_N,
            window_size,
            compress_ratio,
            top_k,
            num_tokens,
            G,
        )
        cached[cache_key] = plan
    idxp, pref, w_lo, w_hi, counts = plan

    # Persistent per-(device, shape) scratch: these are re-created for every
    # C128A layer of every chunk, and o_packed alone is 64 MiB at TP8.
    key = (str(q.device), ng, B_H, h)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = {
            "o": torch.empty((ng, B_H, 512), dtype=q.dtype, device=q.device),
            "lse": torch.empty((ng, B_H), dtype=torch.float32, device=q.device),
            "q": torch.empty((ng, B_H, 512), dtype=q.dtype, device=q.device),
            "sink": torch.empty(B_H, dtype=attn_sink.dtype, device=q.device),
        }
        _SCRATCH[key] = buf
    o_packed, lse, qp, sink = buf["o"], buf["lse"], buf["q"], buf["sink"]
    # attn_sink is a distinct learned parameter for every C128A layer. The
    # large Q/O scratch is shape-shared, but its tiny row-expanded sink must
    # be refreshed before each layer rather than retaining the first layer's.
    sink.view(G, h).copy_(attn_sink[:h])
    # The packed row layout (token-major within a group, then head) is a
    # two-level stride over the 64-head tensor, so it cannot be a view; this
    # copy and the output scatter below are what a 3D Q tensor map and a
    # strided epilogue would remove.
    qp.view(ng, G, h, 512).copy_(q[:, :h].view(ng, G, h, 512))
    kvp = kv.view(-1, 1, kv.shape[-1])

    fn(qp, kvp, idxp, pref, w_lo, w_hi, counts, float(sm_scale), h, o_packed, lse, sink)

    out[:, :h].copy_(o_packed.view(num_tokens, h, 512))
    global _EXECUTION_LOGGED
    if not _EXECUTION_LOGGED:
        logger.info("DSV4_PACKED_KERNEL_EXECUTED packed BF16 MLA kernel dispatched")
        _EXECUTION_LOGGED = True
    return True
