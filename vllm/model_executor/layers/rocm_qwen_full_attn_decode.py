# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused projections around Qwen3.5-family gated full attention on ROCm.

Two custom ops replace the stock chain on either side of the attention
backend, which runs unchanged in between:

* front: residual add + input RMSNorm + QKV GEMV + q/gate split + q/k
  RMSNorm + (M)RoPE, in one launch.
* back: sigmoid output gate + o_proj GEMV, then residual add +
  post-attention RMSNorm, in two launches.

Steps of at most ``MAX_ROWS`` rows run the fused Triton kernels; larger steps
run the stock modules inside the same ops.

Front programs hand each q/k head to whichever of them finishes it last
through an arrival counter, which that program resets, so no program waits
on another and the counters are back at zero after every launch (HIP-graph
safe).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)

MAX_ROWS = 32
_O_SPLITS = 4
# Per padded row count, tuned on MI355X: front (BM, num_warps, num_stages,
# software-prefetch the next K step), o_proj (BM, num_warps, num_stages).
_FRONT_CFG = {1: (64, 8, 1, True), 16: (32, 4, 2, False), 32: (64, 4, 2, False)}
_OPROJ_CFG = {1: (16, 8, 2), 16: (32, 4, 2), 32: (64, 4, 1)}

_FrontOut = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

_LAYERS: dict[str, torch.nn.Module] = {}
_SCRATCH: dict[tuple, torch.Tensor] = {}


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
    pos_ptr,
    cs_ptr,
    qn_w_ptr,
    kn_w_ptr,
    cnt_ptr,
    n_tok,
    eps,
    cs_stride,
    pos_stride_m,
    HID: tl.constexpr,
    NQH: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    ROT: tl.constexpr,
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

    if r0 >= KOFF:
        d = (r0 - KOFF) + tl.arange(0, BM)
        tl.store(v_ptr + t[None, :] * KVD + d[:, None], out, mask=tmask[None, :])
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
                dst = q_ptr + t[:, None] * (NQH * HD) + head * HD + dd[None, :]
            else:
                dst = k_ptr + t[:, None] * KVD + (head - NQH) * HD + dd[None, :]
            tl.store(dst, res, mask=tmask[:, None])


@triton.jit
def _oproj_kernel(
    x_ptr,
    gate_ptr,
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
    x_offs = t[:, None] * KIN + kk[None, :]
    for k0 in range(ks * KC, ks * KC + KC, BK):
        w = tl.load(w_row + k0, cache_modifier=".cg")
        x = tl.load(x_ptr + x_offs + k0, mask=tmask[:, None], other=0.0)
        g = tl.load(gate_ptr + x_offs + k0, mask=tmask[:, None], other=0.0)
        xg = (x.to(tl.float32) * tl.sigmoid(g.to(tl.float32))).to(w.dtype)
        if NP == 1:
            acc_k += w.to(tl.float32) * xg.to(tl.float32)
        else:
            acc += tl.dot(w, tl.trans(xg))
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
def _sigmoid_gate_kernel(x_ptr, gate_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    o = x * tl.sigmoid(g)
    tl.store(out_ptr + offs, o.to(out_ptr.dtype.element_ty), mask=mask)


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
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | None,
    eps: float,
) -> _FrontOut:
    """Returns (q, gate, k, v, residual + hidden).

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
        positions, cos_sin_cache, q_norm_w, k_norm_w, cnt, n, eps,
        cos_sin_cache.stride(0), positions.stride(0) if has_mrope else 0,
        HID=hid, NQH=num_heads, NKV=num_kv_heads, HD=head_dim, ROT=rotary_dim,
        HAS_MROPE=has_mrope, MROPE_H=mrope_h, MROPE_W=mrope_w, NP=np_, BM=bm,
        BK=256, RES_CH=triton.next_power_of_2(triton.cdiv(hid, grid)),
        PREFETCH=prefetch, num_warps=warps, num_stages=stages,
    )  # fmt: skip
    return q, gate, k, v, res_mid


def fused_back(
    attn: torch.Tensor,
    gate: torch.Tensor,
    res_mid: torch.Tensor,
    o_w: torch.Tensor,
    post_ln_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (post_norm(h + res_mid), h + res_mid) for
    h = o_proj(attn * sigmoid(gate))."""
    n = attn.shape[0]
    hid = o_w.shape[0]
    np_ = _padded_rows(n)
    h = _scratch("o", _O_SPLITS * MAX_ROWS * hid, torch.float32, attn.device)
    bm, warps, stages = _OPROJ_CFG[np_]
    _oproj_kernel[(hid // bm, _O_SPLITS)](
        attn, gate, o_w, h, n,
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
        and hidden_size % 256 == 0
        and qkv_rows % min_bm == 0
        and (num_heads * head_dim) % (256 * _O_SPLITS) == 0
        and rotary_dim % 2 == 0
        and rotary_dim <= head_dim
    )


def register_layer(layer_name: str, layer: torch.nn.Module) -> None:
    _LAYERS[layer_name] = layer


def rocm_qwen_full_attn_front(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    q_size: int,
    kv_size: int,
    layer_name: LayerNameType,
) -> _FrontOut:
    layer = _LAYERS[_resolve_layer_name(layer_name)]
    sa = layer.self_attn
    eps = layer.input_layernorm.variance_epsilon
    n = hidden_states.shape[0]
    if n <= MAX_ROWS:
        return fused_front(
            hidden_states, residual, positions[..., :n],
            layer.input_layernorm.weight, sa.qkv_proj.weight, sa.q_norm.weight,
            sa.k_norm.weight, sa.rotary_emb.cos_sin_cache, sa.num_heads,
            sa.num_kv_heads, sa.head_dim, sa.rotary_emb.rotary_dim,
            getattr(sa.rotary_emb, "mrope_section", None), eps,
        )  # fmt: skip
    x, res_mid = _add_norm(hidden_states, residual, layer.input_layernorm.weight, eps)
    qkv, _ = sa.qkv_proj(x)
    q, k, v, gate = sa._project_qkv_gate(qkv, positions)
    return q, gate, k.contiguous(), v.contiguous(), res_mid


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


def rocm_qwen_full_attn_back(
    attn: torch.Tensor,
    gate: torch.Tensor,
    res_mid: torch.Tensor,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = _LAYERS[_resolve_layer_name(layer_name)]
    eps = layer.post_attention_layernorm.variance_epsilon
    w = layer.post_attention_layernorm.weight
    if attn.shape[0] <= MAX_ROWS:
        return fused_back(attn, gate, res_mid, layer.self_attn.o_proj.weight, w, eps)
    gated = torch.empty_like(attn)
    numel = attn.numel()
    _sigmoid_gate_kernel[(triton.cdiv(numel, 2048),)](
        attn, gate, gated, numel, BLOCK=2048, num_warps=4
    )
    h, _ = layer.self_attn.o_proj(gated)
    return _add_norm(h, res_mid, w, eps)


def _back_fake(
    attn: torch.Tensor,
    gate: torch.Tensor,
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
    op_name="rocm_qwen_full_attn_back",
    op_func=rocm_qwen_full_attn_back,
    fake_impl=_back_fake,
)
