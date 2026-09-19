"""Combine kernel for the split-KV mega decode path (Triton stand-in).

Merges the fp32 per-split partials the mega kernel writes, applies the
attention sink exactly once, and runs the epilogue: fp8 e4m3 cast with packed
ue8m0 per-32 scales into the kernel's permuted
[s_q, n_wv_group, wv_group_size*d_v] layout.
"""

import torch
import triton
import triton.language as tl

WV = tl.constexpr(8)
GRAN = tl.constexpr(32)
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _combine(
    o_accum,
    lse_accum,
    out_fp8,
    out_sf_u8,
    sink,
    s_o_split,
    s_l_split,
    sf_stride_wv,
    sf_stride_hd,
    H_Q: tl.constexpr,
    NS: tl.constexpr,
    DV: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.arange(0, NS)
    lse = tl.load(lse_accum + sp * s_l_split + s * H_Q + h)
    m = tl.max(lse, 0)
    empty = m == float("-inf")
    m = tl.where(empty, 0.0, m)
    g = m + tl.log2(tl.sum(tl.exp2(lse - m), 0))
    g = tl.where(empty, float("-inf"), g)
    if HAS_SINK:
        sk = tl.load(sink + h).to(tl.float32) * LOG2E
        g = tl.where(empty, sk, g + tl.log2(1.0 + tl.exp2(sk - g)))
    sc = tl.exp2(lse - g)
    sc = tl.where(sc != sc, 0.0, sc)

    d = tl.arange(0, DV)
    acc = tl.zeros([DV], dtype=tl.float32)
    for i in tl.static_range(NS):
        v = tl.load(o_accum + i * s_o_split + (s * H_Q + h) * DV + d)
        acc += v * tl.sum(tl.where(sp == i, sc, 0.0), 0)

    grp = d // GRAN
    a2 = tl.reshape(tl.abs(acc), (DV // GRAN, GRAN))
    amax = tl.maximum(tl.max(a2, 1), 1e-4)
    sf = amax / 448.0
    exp_sf = ((sf.to(tl.int32, bitcast=True) - 1) >> 23) + (1 - 127)
    sf_inv = ((127 - exp_sf) << 23).to(tl.float32, bitcast=True)
    q = acc * tl.sum(
        tl.where(
            tl.arange(0, DV // GRAN)[None, :] == grp[:, None], sf_inv[None, :], 0.0
        ),
        1,
    )
    off = (
        (h // WV) * (WV * DV) + (d // GRAN) * (WV * GRAN) + (h % WV) * GRAN + (d % GRAN)
    )
    tl.store(out_fp8 + s * (H_Q * DV) + off, q.to(tl.float8e4nv))

    k = tl.arange(0, DV // GRAN)
    blk = (h % WV) + k * WV
    idx = (s + (h // WV) * sf_stride_wv + (blk // 4) * sf_stride_hd) * 4 + (blk % 4)
    tl.store(out_sf_u8 + idx, (exp_sf + 127).to(tl.uint8))


def mega_combine(o_accum, lse_accum, out_fp8, out_sf, attn_sink=None):
    ns, s_q, h_q, dv = o_accum.shape
    n_i32 = out_sf.untyped_storage().size() // 4
    sf_u8 = torch.as_strided(out_sf, (n_i32,), (1,), 0).view(torch.uint8)
    base = out_sf.storage_offset()
    _combine[(s_q, h_q)](
        o_accum,
        lse_accum,
        out_fp8,
        sf_u8[base * 4 :],
        attn_sink,
        o_accum.stride(0),
        lse_accum.stride(0),
        out_sf.stride(1),
        out_sf.stride(2),
        H_Q=h_q,
        NS=ns,
        DV=dv,
        HAS_SINK=attn_sink is not None,
        num_warps=4,
    )
