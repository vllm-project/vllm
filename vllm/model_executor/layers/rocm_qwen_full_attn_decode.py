# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Qwen3.5-family gated full-attention layer for ROCm decode.

The layer from its input RMSNorm to its post-attention RMSNorm is split into
three custom ops so that only attention runs outside piecewise CUDA graphs:

* front: residual add + input RMSNorm + QKV GEMV + q/gate split + q/k
  RMSNorm + (M)RoPE + K/V cache write (ROCM_ATTN head-major layout).
* core (graph split point): split-KV paged decode attention + sigmoid output
  gate for pure decode steps, the ROCM_ATTN kernel otherwise.
* back: o_proj GEMV, then residual add + post-attention RMSNorm.

Steps of at most ``MAX_ROWS`` rows run front/back as fused Triton kernels
(four launches per layer for decode, 12 in the stock chain on MI355X);
larger steps run the stock modules inside the same ops.

Programs hand work to each other through arrival counters: the last program
to finish a unit of work (a q/k head, a KV-split group) runs its epilogue
and resets the counter, so no program waits on another and the counters are
back at zero after every launch (HIP-graph safe).
"""

from typing import cast

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionImpl,
    RocmAttentionMetadata,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention

MAX_ROWS = 32
_MAX_SPLITS = 32
_ATTN_PROGRAMS = 256
# (TILE, num_warps, num_stages), tuned on MI355X.
_ATTN_CFG = (64, 8, 2)
_O_SPLITS = 4
# Per padded row count, tuned on MI355X: front (BM, num_warps, num_stages,
# software-prefetch the next K step), o_proj (BM, num_warps, num_stages).
_FRONT_CFG = {1: (64, 8, 1, True), 16: (32, 4, 2, False), 32: (64, 4, 2, False)}
_OPROJ_CFG = {1: (16, 8, 2), 16: (32, 4, 2), 32: (64, 4, 1)}

_FrontOut = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

_LAYERS: dict[str, torch.nn.Module] = {}
_SCRATCH: dict[tuple, dict[str, torch.Tensor]] = {}


@triton.jit
def _front_kernel(
    hid_ptr,
    res_ptr,
    res_out_ptr,
    ln_w_ptr,
    w_ptr,
    qk_scr_ptr,
    gate_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    kc_ptr,
    vc_ptr,
    slot_ptr,
    pos_ptr,
    cs_ptr,
    qn_w_ptr,
    kn_w_ptr,
    cnt_ptr,
    n_tok,
    eps,
    block_size,
    kc_bs,
    kc_hs,
    kc_ds,
    vc_bs,
    vc_hs,
    vc_ds,
    cs_stride,
    pos_stride_m,
    HID: tl.constexpr,
    NQH: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    ROT: tl.constexpr,
    X: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    NP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    RES_CH: tl.constexpr,
    PREFETCH: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.arange(0, NP)
    tmask = t < n_tok
    kk = tl.arange(0, BK)
    QG: tl.constexpr = NQH * 2 * HD
    KOFF: tl.constexpr = QG + NKV * HD
    HALF: tl.constexpr = ROT // 2
    TPH: tl.constexpr = HD // BM
    SCR: tl.constexpr = (NQH + NKV) * HD
    KVD: tl.constexpr = NKV * HD
    dt = q_ptr.dtype.element_ty
    r0 = pid * BM

    dd = tl.arange(0, HD)
    partner = tl.where(dd < HALF, dd + HALF, dd - HALF)
    rmask = tmask[:, None] & (dd < ROT)[None, :]
    # Epilogue operands are fetched before the weight stream starts.
    j = dd % HALF
    pos = tl.load(pos_ptr + t, mask=tmask, other=0).to(tl.int64)[:, None]
    if HAS_MROPE:
        pos_h = tl.load(pos_ptr + pos_stride_m + t, mask=tmask, other=0)
        pos_w = tl.load(pos_ptr + 2 * pos_stride_m + t, mask=tmask, other=0)
        is_h = ((j % 3 == 1) & (j < 3 * MROPE_H))[None, :]
        is_w = ((j % 3 == 2) & (j < 3 * MROPE_W))[None, :]
        pos = tl.where(
            is_h,
            pos_h.to(tl.int64)[:, None],
            tl.where(is_w, pos_w.to(tl.int64)[:, None], pos),
        )
    cos = tl.load(cs_ptr + pos * cs_stride + j[None, :], mask=rmask, other=1.0).to(
        tl.float32
    )
    sin = tl.load(
        cs_ptr + pos * cs_stride + HALF + j[None, :], mask=rmask, other=0.0
    ).to(tl.float32)
    nw_ptr = qn_w_ptr if r0 < QG else kn_w_ptr
    nw = tl.load(nw_ptr + dd).to(tl.float32)
    nwp = tl.load(nw_ptr + partner).to(tl.float32)

    if pid * RES_CH < HID:
        c = pid * RES_CH + tl.arange(0, RES_CH)
        offs = t[:, None] * HID + c[None, :]
        cm = tmask[:, None] & (c < HID)[None, :]
        s = tl.load(hid_ptr + offs, mask=cm, other=0.0).to(tl.float32)
        s += tl.load(res_ptr + offs, mask=cm, other=0.0).to(tl.float32)
        tl.store(res_out_ptr + offs, s.to(res_out_ptr.dtype.element_ty), mask=cm)

    # GEMV on (h + r) * (1 + w); 1/rms is applied to the accumulator.
    rm = r0 + tl.arange(0, BM)
    ss_k = tl.zeros([NP, BK], dtype=tl.float32)
    if NP == 1:
        acc_k = tl.zeros([BM, BK], dtype=tl.float32)
    else:
        acc = tl.zeros([BM, NP], dtype=tl.float32)
    w_row = w_ptr + rm[:, None].to(tl.int64) * HID + kk[None, :]
    a_offs = t[:, None] * HID + kk[None, :]
    if PREFETCH:
        w_n = tl.load(w_row, cache_modifier=".cg")
        h_n = tl.load(hid_ptr + a_offs, mask=tmask[:, None], other=0.0)
        r_n = tl.load(res_ptr + a_offs, mask=tmask[:, None], other=0.0)
        lw_n = tl.load(ln_w_ptr + kk)
    for k0 in range(0, HID, BK):
        if PREFETCH:
            w, hh, rr, lw = w_n, h_n, r_n, lw_n
            if k0 + BK < HID:
                w_n = tl.load(w_row + k0 + BK, cache_modifier=".cg")
                h_n = tl.load(
                    hid_ptr + a_offs + k0 + BK, mask=tmask[:, None], other=0.0
                )
                r_n = tl.load(
                    res_ptr + a_offs + k0 + BK, mask=tmask[:, None], other=0.0
                )
                lw_n = tl.load(ln_w_ptr + k0 + BK + kk)
        else:
            w = tl.load(w_row + k0, cache_modifier=".cg")
            hh = tl.load(hid_ptr + a_offs + k0, mask=tmask[:, None], other=0.0)
            rr = tl.load(res_ptr + a_offs + k0, mask=tmask[:, None], other=0.0)
            lw = tl.load(ln_w_ptr + k0 + kk)
        s = hh.to(tl.float32) + rr.to(tl.float32)
        ss_k += s * s
        a = (s * (1.0 + lw.to(tl.float32)[None, :])).to(w.dtype)
        if NP == 1:
            acc_k += w.to(tl.float32) * a.to(tl.float32)
        else:
            acc += tl.dot(w, tl.trans(a))
    if NP == 1:
        acc = tl.sum(acc_k, axis=1)[:, None]
    rstd = tl.rsqrt(tl.sum(ss_k, axis=1) / HID + eps)
    out = (acc * rstd[None, :]).to(dt)  # [BM, NP]

    slot = tl.load(slot_ptr + t, mask=tmask, other=-1).to(tl.int64)
    smask = tmask & (slot >= 0)
    blk = slot // block_size
    boff = slot % block_size

    if r0 >= KOFF:
        vh = (r0 - KOFF) // HD
        d = (r0 - KOFF) % HD + tl.arange(0, BM)
        dst = blk[None, :] * vc_bs + vh * vc_hs + d[:, None] * vc_ds + boff[None, :]
        tl.store(vc_ptr + dst, out, mask=smask[None, :])
        tl.store(
            v_ptr + t[None, :] * KVD + vh * HD + d[:, None],
            out,
            mask=tmask[None, :],
        )
    elif (r0 < QG) and ((r0 % (2 * HD)) >= HD):
        h = r0 // (2 * HD)
        d = (r0 % (2 * HD)) - HD + tl.arange(0, BM)
        tl.store(
            gate_ptr + t[None, :] * (NQH * HD) + h * HD + d[:, None],
            out,
            mask=tmask[None, :],
        )
    else:
        if r0 < QG:
            head = r0 // (2 * HD)
            d0 = r0 % (2 * HD)
        else:
            head = NQH + (r0 - QG) // HD
            d0 = (r0 - QG) % HD
        d = d0 + tl.arange(0, BM)
        tl.store(
            qk_scr_ptr + t[None, :] * SCR + head * HD + d[:, None],
            out,
            mask=tmask[None, :],
        )
        tl.debug_barrier()
        old = tl.atomic_add(cnt_ptr + head, 1, sem="release", scope="gpu")
        if old == TPH - 1:
            tl.atomic_xchg(cnt_ptr + head, 0, sem="acquire", scope="gpu")
            base = qk_scr_ptr + t[:, None] * SCR + head * HD
            v = tl.load(base + dd[None, :], mask=tmask[:, None], other=0.0).to(
                tl.float32
            )
            vp = tl.load(base + partner[None, :], mask=rmask, other=0.0).to(tl.float32)
            r = tl.rsqrt(tl.sum(v * v, axis=1) / HD + eps)[:, None]
            vx = (v * r * (1.0 + nw[None, :])).to(dt).to(tl.float32)
            vp = (vp * r * (1.0 + nwp[None, :])).to(dt).to(tl.float32)
            sign = tl.where(dd < HALF, -1.0, 1.0)
            rot = vx * cos + sign[None, :] * vp * sin
            res = tl.where((dd < ROT)[None, :], rot, vx).to(dt)
            if head < NQH:
                tl.store(
                    q_ptr + t[:, None] * (NQH * HD) + head * HD + dd[None, :],
                    res,
                    mask=tmask[:, None],
                )
            else:
                kh = head - NQH
                tl.store(
                    k_ptr + t[:, None] * KVD + kh * HD + dd[None, :],
                    res,
                    mask=tmask[:, None],
                )
                dst = (
                    blk[:, None] * kc_bs
                    + kh * kc_hs
                    + (dd // X)[None, :] * kc_ds
                    + boff[:, None] * X
                    + (dd % X)[None, :]
                )
                tl.store(kc_ptr + dst, res, mask=smask[:, None])


@triton.jit
def _attn_kernel(
    q_ptr,
    gate_ptr,
    out_ptr,
    kc_ptr,
    vc_ptr,
    bt_ptr,
    sl_ptr,
    po_ptr,
    pml_ptr,
    cnt_ptr,
    scale,
    num_seqs,
    block_size,
    kc_bs,
    kc_hs,
    kc_ds,
    vc_bs,
    vc_hs,
    vc_ds,
    bt_stride,
    NQH: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    X: tl.constexpr,
    GQA: tl.constexpr,
    GP: tl.constexpr,
    TILE: tl.constexpr,
    NSPLIT: tl.constexpr,
    NSP2: tl.constexpr,
):
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    b = pid // NKV
    kvh = pid % NKV
    seq_len = tl.load(sl_ptr + b, mask=b < num_seqs, other=0)
    chunk = tl.maximum(tl.cdiv(tl.cdiv(seq_len, NSPLIT), TILE), 1) * TILE
    n_active = tl.cdiv(seq_len, chunk)
    start = sp * chunk
    end = tl.minimum(seq_len, start + chunk)

    g = tl.arange(0, GP)
    gmask = g < GQA
    heads = kvh * GQA + g
    d = tl.arange(0, HD)
    gofs = b * (NQH * HD) + heads[:, None] * HD + d[None, :]
    q = tl.load(q_ptr + gofs, mask=gmask[:, None], other=0.0)
    gt = tl.load(gate_ptr + gofs, mask=gmask[:, None], other=0.0).to(tl.float32)
    m = tl.full([GP], float("-inf"), dtype=tl.float32)
    lsum = tl.zeros([GP], dtype=tl.float32)
    acc = tl.zeros([GP, HD], dtype=tl.float32)
    bt_row = bt_ptr + b.to(tl.int64) * bt_stride
    for t0 in range(start, end, TILE):
        tok = t0 + tl.arange(0, TILE)
        tmask = tok < end
        phys = tl.load(bt_row + tok // block_size, mask=tmask, other=0).to(tl.int64)
        off = tok % block_size
        k = tl.load(
            kc_ptr
            + phys[None, :] * kc_bs
            + kvh * kc_hs
            + (d // X)[:, None] * kc_ds
            + off[None, :] * X
            + (d % X)[:, None],
            mask=tmask[None, :],
            other=0.0,
        )
        v_ptrs = (
            vc_ptr
            + phys[:, None] * vc_bs
            + kvh * vc_hs
            + d[None, :] * vc_ds
            + off[:, None]
        )
        v = tl.load(v_ptrs, mask=tmask[:, None], other=0.0)
        sc = tl.dot(q, k) * scale
        sc = tl.where(tmask[None, :], sc, float("-inf"))
        m_new = tl.maximum(m, tl.max(sc, axis=1))
        p = tl.exp(sc - m_new[:, None])
        alpha = tl.exp(m - m_new)
        lsum = lsum * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    # Only the n_active splits that own tokens arrive; one split (or an empty
    # pad row) writes its result directly.
    if n_active <= 1:
        if sp == 0:
            o = acc / tl.where(lsum > 0, lsum, 1.0)[:, None]
            o = o * tl.sigmoid(gt)
            tl.store(
                out_ptr + gofs, o.to(out_ptr.dtype.element_ty), mask=gmask[:, None]
            )
    elif sp < n_active:
        grp = b * NKV + kvh
        pbase = (grp * NSPLIT + sp) * GQA
        tl.store(
            po_ptr + (pbase + g[:, None]) * HD + d[None, :],
            acc,
            mask=gmask[:, None],
        )
        tl.store(pml_ptr + (pbase + g) * 2, m, mask=gmask)
        tl.store(pml_ptr + (pbase + g) * 2 + 1, lsum, mask=gmask)
        tl.debug_barrier()
        old = tl.atomic_add(cnt_ptr + grp, 1, sem="release", scope="gpu")
        if old == n_active - 1:
            tl.atomic_xchg(cnt_ptr + grp, 0, sem="acquire", scope="gpu")
            si = tl.arange(0, NSP2)
            ml = pml_ptr + ((grp * NSPLIT + si)[:, None] * GQA + g[None, :]) * 2
            pm = (si < n_active)[:, None] & gmask[None, :]
            m_max = tl.max(tl.load(ml, mask=pm, other=float("-inf")), axis=0)
            m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
            l_tot = tl.zeros([GP], dtype=tl.float32)
            o_tot = tl.zeros([GP, HD], dtype=tl.float32)
            for s in range(n_active):
                row = (grp * NSPLIT + s) * GQA + g
                ms = tl.load(pml_ptr + row * 2, mask=gmask, other=float("-inf"))
                ls = tl.load(pml_ptr + row * 2 + 1, mask=gmask, other=0.0)
                wgt = tl.exp(ms - m_max)
                l_tot += ls * wgt
                o_s = tl.load(
                    po_ptr + row[:, None] * HD + d[None, :],
                    mask=gmask[:, None],
                    other=0.0,
                )
                o_tot += o_s * wgt[:, None]
            o = o_tot / tl.where(l_tot > 0, l_tot, 1.0)[:, None]
            o = o * tl.sigmoid(gt)
            tl.store(
                out_ptr + gofs, o.to(out_ptr.dtype.element_ty), mask=gmask[:, None]
            )


@triton.jit
def _oproj_kernel(
    x_ptr,
    w_ptr,
    h_ptr,
    n_tok,
    HID: tl.constexpr,
    KIN: tl.constexpr,
    NP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    pid = tl.program_id(0)
    ks = tl.program_id(1)
    t = tl.arange(0, NP)
    tmask = t < n_tok
    kk = tl.arange(0, BK)
    rm = pid * BM + tl.arange(0, BM)
    if NP == 1:
        acc_k = tl.zeros([BM, BK], dtype=tl.float32)
    else:
        acc = tl.zeros([BM, NP], dtype=tl.float32)
    KC: tl.constexpr = KIN // SK
    w_row = w_ptr + rm[:, None].to(tl.int64) * KIN + kk[None, :]
    x_row = x_ptr + t[:, None] * KIN + kk[None, :]
    for k0 in range(ks * KC, ks * KC + KC, BK):
        w = tl.load(w_row + k0, cache_modifier=".cg")
        x = tl.load(x_row + k0, mask=tmask[:, None], other=0.0)
        if NP == 1:
            acc_k += w.to(tl.float32) * x.to(tl.float32)
        else:
            acc += tl.dot(w, tl.trans(x))
    if NP == 1:
        acc = tl.sum(acc_k, axis=1)[:, None]
    tl.store(
        h_ptr + (ks * NP + t[None, :]) * HID + rm[:, None],
        acc,
        mask=tmask[None, :],
    )


@triton.jit
def _add_norm_kernel(
    h_ptr,
    res_ptr,
    x_out_ptr,
    res_out_ptr,
    ln_w_ptr,
    eps,
    HID: tl.constexpr,
    NP: tl.constexpr,
    SK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    cmask = c < HID
    offs = row * HID + c
    o = tl.zeros([BLOCK], dtype=tl.float32)
    for ks in tl.static_range(SK):
        o += tl.load(h_ptr + (ks * NP + row) * HID + c, mask=cmask, other=0.0)
    s = o.to(x_out_ptr.dtype.element_ty).to(tl.float32)
    s += tl.load(res_ptr + offs, mask=cmask, other=0.0).to(tl.float32)
    tl.store(res_out_ptr + offs, s.to(res_out_ptr.dtype.element_ty), mask=cmask)
    rstd = tl.rsqrt(tl.sum(s * s, axis=0) / HID + eps)
    lw = tl.load(ln_w_ptr + c, mask=cmask, other=0.0).to(tl.float32)
    y = s * rstd * (1.0 + lw)
    tl.store(x_out_ptr + offs, y.to(x_out_ptr.dtype.element_ty), mask=cmask)


@triton.jit
def _sigmoid_gate_kernel(x_ptr, gate_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    tl.store(x_ptr + offs, (x * tl.sigmoid(g)).to(x_ptr.dtype.element_ty), mask=mask)


def _scratch(
    name: str, numel: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Persistent per-process buffer; counters start (and stay) at zero."""
    key = (name, numel, dtype, device)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = torch.zeros(numel, dtype=dtype, device=device)
        _SCRATCH[key] = buf
    return buf


def _padded_rows(n: int) -> int:
    # 1 row uses an FMA mainloop, 16/32 use tl.dot.
    return 1 if n == 1 else (16 if n <= 16 else 32)


def fused_front(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    in_ln_w: torch.Tensor,
    qkv_w: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | None,
    eps: float,
) -> _FrontOut:
    """Returns (q, gate, k, v, residual + hidden) and writes K/V to the cache.

    ``positions`` is [n] or [3, n]; the latter with ``mrope_section`` applies
    interleaved MRoPE.
    """
    n, hid = hidden.shape
    np_ = _padded_rows(n)
    nqk = (num_heads + num_kv_heads) * head_dim
    qk_scr = _scratch("qk", MAX_ROWS * nqk, hidden.dtype, hidden.device)
    cnt = _scratch("front_cnt", num_heads + num_kv_heads, torch.int32, hidden.device)
    has_mrope, mrope_h, mrope_w = False, 0, 0
    if positions.ndim == 2:
        if mrope_section:
            has_mrope, mrope_h, mrope_w = True, mrope_section[1], mrope_section[2]
        else:
            positions = positions[0]

    res_mid = torch.empty_like(residual)
    q = hidden.new_empty(n, num_heads * head_dim)
    gate = torch.empty_like(q)
    k = hidden.new_empty(n, num_kv_heads * head_dim)
    v = torch.empty_like(k)
    bm, warps, stages, prefetch = _FRONT_CFG[np_]
    grid = qkv_w.shape[0] // bm
    _front_kernel[(grid,)](
        hidden, residual, res_mid, in_ln_w, qkv_w, qk_scr, gate, q, k, v,
        key_cache, value_cache, slot_mapping, positions, cos_sin_cache,
        q_norm_w, k_norm_w, cnt, n, eps, value_cache.shape[3],
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        cos_sin_cache.stride(0), positions.stride(0) if has_mrope else 0,
        HID=hid, NQH=num_heads, NKV=num_kv_heads, HD=head_dim, ROT=rotary_dim,
        X=key_cache.shape[4], HAS_MROPE=has_mrope, MROPE_H=mrope_h,
        MROPE_W=mrope_w, NP=np_, BM=bm, BK=256,
        RES_CH=triton.next_power_of_2(triton.cdiv(hid, grid)),
        PREFETCH=prefetch, num_warps=warps, num_stages=stages,
    )  # fmt: skip
    return q, gate, k, v, res_mid


def fused_attn(
    q: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> None:
    """Gated decode attention (one query row per sequence) into ``out``."""
    n = q.shape[0]
    gqa = num_heads // num_kv_heads
    tile, warps, stages = _ATTN_CFG
    groups = MAX_ROWS * num_kv_heads
    nsplit = max(1, min(_MAX_SPLITS, _ATTN_PROGRAMS // (n * num_kv_heads)))
    slots = max(_ATTN_PROGRAMS, groups) * gqa
    f32 = torch.float32
    po = _scratch("po", slots * head_dim, f32, q.device)
    pml = _scratch("pml", slots * 2, f32, q.device)
    cnt = _scratch("attn_cnt", groups, torch.int32, q.device)
    _attn_kernel[(n * num_kv_heads, nsplit)](
        q, gate, out, key_cache, value_cache, block_table, seq_lens,
        po, pml, cnt, scale,
        seq_lens.shape[0], value_cache.shape[3],
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_table.stride(0),
        NQH=num_heads, NKV=num_kv_heads, HD=head_dim, X=key_cache.shape[4],
        GQA=gqa, GP=triton.next_power_of_2(max(gqa, 16)), TILE=tile, NSPLIT=nsplit,
        NSP2=triton.next_power_of_2(nsplit),
        num_warps=warps,
        num_stages=stages,
    )  # fmt: skip


def fused_back(
    attn: torch.Tensor,
    res_mid: torch.Tensor,
    o_w: torch.Tensor,
    post_ln_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (post_norm(o_proj(attn) + res_mid), o_proj(attn) + res_mid)."""
    n = attn.shape[0]
    hid = o_w.shape[0]
    np_ = _padded_rows(n)
    h = _scratch("o", _O_SPLITS * MAX_ROWS * hid, torch.float32, attn.device)
    bm, warps, stages = _OPROJ_CFG[np_]
    _oproj_kernel[(hid // bm, _O_SPLITS)](
        attn, o_w, h, n,
        HID=hid, KIN=o_w.shape[1], NP=np_, BM=bm, BK=256, SK=_O_SPLITS,
        num_warps=warps, num_stages=stages,
    )  # fmt: skip
    x_out, res_out = torch.empty_like(res_mid), torch.empty_like(res_mid)
    _add_norm_kernel[(n,)](
        h, res_mid, x_out, res_out, post_ln_w, eps,
        HID=hid, NP=np_, SK=_O_SPLITS, BLOCK=triton.next_power_of_2(hid),
        num_warps=8,
    )  # fmt: skip
    return x_out, res_out


def _add_norm(
    h: torch.Tensor, res: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    hid = h.shape[1]
    x_out, res_out = torch.empty_like(h), torch.empty_like(res)
    _add_norm_kernel[(h.shape[0],)](
        h, res, x_out, res_out, w, eps,
        HID=hid, NP=1, SK=1, BLOCK=triton.next_power_of_2(hid), num_warps=8,
    )  # fmt: skip
    return x_out, res_out


def supports_fused_full_attn_decode(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    min_bm = min(cfg[0] for cfg in _FRONT_CFG.values())
    max_bm = max(cfg[0] for cfg in _FRONT_CFG.values())
    qkv_rows = 2 * (num_heads + num_kv_heads) * head_dim
    return (
        head_dim == triton.next_power_of_2(head_dim)
        and head_dim % max_bm == 0
        and num_heads % num_kv_heads == 0
        and num_heads // num_kv_heads <= 16
        and hidden_size % 256 == 0
        and qkv_rows % min_bm == 0
        and (num_heads * head_dim) % (256 * _O_SPLITS) == 0
        and rotary_dim % 2 == 0
        and rotary_dim <= head_dim
    )


def register_layer(layer_name: str, layer: torch.nn.Module) -> None:
    _LAYERS[layer_name] = layer


def _kv_caches(attn_layer: torch.nn.Module, kv_cache: torch.Tensor):
    return PagedAttention.split_kv_cache(
        kv_cache.transpose(0, 1), attn_layer.num_kv_heads, attn_layer.head_size
    )


def rocm_qwen_full_attn_front(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    q_size: int,
    kv_size: int,
    layer_name: LayerNameType,
) -> _FrontOut:
    name = _resolve_layer_name(layer_name)
    layer = _LAYERS[name]
    sa = layer.self_attn
    attn_layer = sa.attn
    eps = layer.input_layernorm.variance_epsilon
    mrope_section = getattr(sa.rotary_emb, "mrope_section", None)
    slot_maps = get_forward_context().slot_mapping
    slot_mapping = slot_maps.get(name) if isinstance(slot_maps, dict) else None
    n = hidden_states.shape[0]
    impl = cast(RocmAttentionImpl, attn_layer.impl)
    if n <= MAX_ROWS and not impl._use_interleaved_v_cache:
        if slot_mapping is None:
            slot_mapping = torch.full(
                (n,), -1, dtype=torch.int64, device=hidden_states.device
            )
        key_cache, value_cache = _kv_caches(attn_layer, attn_layer.kv_cache)
        return fused_front(
            hidden_states, residual, positions[..., :n],
            layer.input_layernorm.weight, sa.qkv_proj.weight, sa.q_norm.weight,
            sa.k_norm.weight, sa.rotary_emb.cos_sin_cache, key_cache,
            value_cache, slot_mapping[:n], sa.num_heads, sa.num_kv_heads,
            sa.head_dim, sa.rotary_emb.rotary_dim, mrope_section, eps,
        )  # fmt: skip

    x, res_mid = _add_norm(hidden_states, residual, layer.input_layernorm.weight, eps)
    qkv, _ = sa.qkv_proj(x)
    q_gate, k, v = qkv.split([sa.q_size * 2, sa.kv_size, sa.kv_size], dim=-1)
    if positions.ndim == 2 and not mrope_section:
        positions = positions[0]
    q, k, gate = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        sa.q_norm.weight,
        sa.k_norm.weight,
        sa.rotary_emb.cos_sin_cache,
        positions,
        sa.q_norm.variance_epsilon,
        sa.num_heads,
        sa.num_kv_heads,
        sa.head_dim,
        sa.rotary_emb.rotary_dim,
        mrope_section=mrope_section if positions.ndim == 2 else None,
        norm_beta=1.0,
    )
    v = v.contiguous()
    if slot_mapping is not None:
        impl.do_kv_cache_update(
            attn_layer,
            k.view(-1, sa.num_kv_heads, sa.head_dim),
            v.view(-1, sa.num_kv_heads, sa.head_dim),
            attn_layer.kv_cache,
            slot_mapping,
        )
    return q, gate, k, v, res_mid


def _front_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    q_size: int,
    kv_size: int,
    layer_name: LayerNameType,
) -> _FrontOut:
    n = hidden_states.shape[0]
    q = hidden_states.new_empty(n, q_size)
    k = hidden_states.new_empty(n, kv_size)
    return q, torch.empty_like(q), k, torch.empty_like(k), torch.empty_like(residual)


@eager_break_during_capture
@maybe_transfer_kv_layer
def rocm_qwen_full_attn_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    md, attn_layer, kv_cache, _ = get_attention_context(_resolve_layer_name(layer_name))
    if md is None:
        output.fill_(0)
        return
    impl = cast(RocmAttentionImpl, attn_layer.impl)
    n = q.shape[0]
    if (
        isinstance(md, RocmAttentionMetadata)
        and md.max_query_len == 1
        and n <= MAX_ROWS
        and not impl._use_interleaved_v_cache
    ):
        key_cache, value_cache = _kv_caches(attn_layer, kv_cache)
        fused_attn(
            q, gate, output, key_cache, value_cache, md.block_table, md.seq_lens,
            impl.num_heads, impl.num_kv_heads, impl.head_size, impl.scale,
        )  # fmt: skip
        return
    nh, nkv, hd = impl.num_heads, impl.num_kv_heads, impl.head_size
    impl.forward(
        attn_layer,
        q.view(-1, nh, hd),
        k.view(-1, nkv, hd),
        v.view(-1, nkv, hd),
        kv_cache,
        md,
        output=output.view(-1, nh, hd),
    )
    numel = output.numel()
    _sigmoid_gate_kernel[(triton.cdiv(numel, 2048),)](
        output, gate, numel, BLOCK=2048, num_warps=4
    )


def rocm_qwen_full_attn_back(
    attn: torch.Tensor,
    res_mid: torch.Tensor,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = _LAYERS[_resolve_layer_name(layer_name)]
    eps = layer.post_attention_layernorm.variance_epsilon
    w = layer.post_attention_layernorm.weight
    if attn.shape[0] <= MAX_ROWS:
        return fused_back(attn, res_mid, layer.self_attn.o_proj.weight, w, eps)
    h, _ = layer.self_attn.o_proj(attn)
    return _add_norm(h, res_mid, w, eps)


def _back_fake(
    attn: torch.Tensor,
    res_mid: torch.Tensor,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(res_mid), torch.empty_like(res_mid)


direct_register_custom_op(
    op_name="rocm_qwen_full_attn_front",
    op_func=rocm_qwen_full_attn_front,
    fake_impl=_front_fake,
)
direct_register_custom_op(
    op_name="rocm_qwen_full_attn_core",
    op_func=rocm_qwen_full_attn_core,
    mutates_args=["output"],
)
direct_register_custom_op(
    op_name="rocm_qwen_full_attn_back",
    op_func=rocm_qwen_full_attn_back,
    fake_impl=_back_fake,
)
