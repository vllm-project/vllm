# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 parts of the DeepSeek-V4.1 mono decode kernels
(vllm/models/deepseek_v41/amd/mono) on the GPU.

  key lists  keylist.launch, which writes K2's key lists on an index layer,
             against a reference built in torch from its documented format.
  entry seam AITER's delayed seam on the expanded residual, which the first
             layer uses with the mono layers on, against vLLM's torch seam
             on the folded weight.
  GEMV       the attention GEMV core (attention/gemv.py) and the MoE side's
             FP8 GEMV (stages/gemv.py) on the gfx942 weight copy
             (weights942.dense_copy), against a float64 product.
  MoE        the FNUZ conversion helpers, the bf16 MFMA built from two
             MFMAs, the routed experts' up/gate GEMV and their down
             projection on the gfx942 copy, against float64 references.

The test kernels call the same device functions as K1 and K2, one function
at a time, on random inputs of the model's shapes.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


if not _on_gfx942():
    pytest.skip("the gfx942 mono kernels need a gfx942 GPU", allow_module_level=True)

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from flydsl.expr import const_expr, gpu, range_constexpr  # noqa: E402
from flydsl.expr.typing import Int64, T  # noqa: E402

from vllm.models.deepseek_v41.amd.mono.attention import gemv as G  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.attention.device import (  # noqa: E402
    THREADS,
    WAVES,
    fp8x8_bf16,
    rsrc,
)
from vllm.models.deepseek_v41.amd.mono.common.gfx942 import (  # noqa: E402
    fp4x8_fnuz,
    mfma_bf16_k32,
    ocp_to_fnuz,
)
from vllm.models.deepseek_v41.amd.mono.stages import gemv as SG  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.stages import moe as M  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.weights942 import (  # noqa: E402
    EVEN_ODD,
    codes_tile_rounds,
    dense_copy,
    fp4_drop_negative_zero_,
    fp4_tile_major,
)

HIDDEN = 5120
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
)


def _ld(ptr, i, n=1):
    v = bo.buffer_load(rsrc(ptr), i, vec_width=n, dtype=T.i32)
    return fx.Int32(v) if n == 1 else fx.Vector(v)


def _st(v, ptr, i):
    bo.buffer_store(v, rsrc(ptr), i)


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return (got.double() - ref).abs().max().item() / ref.abs().max().item()


# Key lists


def _keylist_reference(slot, pos, swa_idx, swa_lens, t2r, ratio, topk, bt, cb):
    from vllm.models.deepseek_v41.amd.mono.attention.plan import KEYS, TOPK

    S = slot.shape[0]
    kt = torch.full((S, KEYS), -1, dtype=torch.int64)
    klen = torch.zeros(2 * S, dtype=torch.int32)
    for t in range(S):
        live = int(slot[t]) >= 0
        nswa = int(swa_lens[t]) if live else 0
        ntopk = min((int(pos[t]) + 1) // ratio, TOPK) if live and ratio else 0
        for k in range(ntopk):
            local = int(topk[t, k])
            if local >= 0:
                # The kernel sets bit 31 of the int32 slot. For a slot below
                # 2^31 that is the slot minus 2^31.
                blk = int(bt[int(t2r[t]), local // cb])
                kt[t, k] = blk * cb + local % cb - (1 << 31)
        kt[t, ntopk : ntopk + nswa] = swa_idx[t, 0, :nswa].long()
        klen[t], klen[S + t] = ntopk + nswa, ntopk
    return kt.to(torch.int32).view(-1), klen


@pytest.mark.parametrize("ratio", [0, 1, 2, 4])
def test_keylist_matches_torch_reference(ratio):
    from vllm.models.deepseek_v41.amd.mono import keylist
    from vllm.models.deepseek_v41.amd.mono.attention.plan import KEYS, TOPK, WINDOW

    g = torch.Generator().manual_seed(ratio)
    S, reqs, cb, blocks = 6, 2, 64, 3000
    # Short and long positions, and a padded token with slot -1.
    pos = torch.tensor([3, 40, 700, 131071, 131072, 2047], dtype=torch.int64)
    slot = torch.randint(0, 1 << 30, (S,), generator=g, dtype=torch.int64)
    slot[2] = -1
    swa_lens = torch.minimum(pos + 1, torch.tensor(WINDOW)).to(torch.int32)
    swa_idx = torch.randint(0, 1 << 24, (S, 1, WINDOW), generator=g, dtype=torch.int32)
    t2r = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    topk = torch.full((S, TOPK), -1, dtype=torch.int32)
    for t in range(S):
        n = min((int(pos[t]) + 1) // max(ratio, 1), TOPK)
        topk[t, :n] = torch.randperm(max(n, (int(pos[t]) + 1) // max(ratio, 1)))[:n]
    topk[4, 10:20] = -1
    bt = torch.randint(0, 1 << 20, (reqs, blocks), generator=g, dtype=torch.int32)
    kt = torch.full((S * KEYS,), -7, dtype=torch.int32, device="cuda")
    klen = torch.full((2 * S,), -7, dtype=torch.int32, device="cuda")
    keylist.launch(
        kt,
        klen,
        slot.cuda(),
        pos.cuda(),
        swa_idx.cuda(),
        swa_lens.cuda(),
        t2r.cuda(),
        ratio,
        topk.cuda() if ratio else None,
        bt.cuda() if ratio else None,
        cb,
    )
    torch.accelerator.synchronize()
    ref_kt, ref_klen = _keylist_reference(
        slot, pos, swa_idx, swa_lens, t2r, ratio, topk, bt, cb
    )
    assert torch.equal(klen.cpu(), ref_klen)
    assert torch.equal(kt.cpu(), ref_kt)


# Entry seam


def test_entry_seam_matches_torch():
    """VLLM projects the first layer's 2-D embedding x with the sum of
    hc_attn_fn over the hc copies. With the mono layers on, the seam projects
    the expanded residual (hc equal copies of x) with hc_attn_fn itself. The
    two are the same sum with the products added in another order, and the
    layer input is the same bit for bit."""
    from vllm.model_executor.kernels.mhc.aiter import mhc_pre_delayed_aiter
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    hc, iters = 4, 20
    torch.manual_seed(0)
    hc3 = hc * (2 + hc)
    fn = torch.randn(hc3, hc * HIDDEN, device="cuda") * 0.02
    scale = torch.tensor([0.5, 0.7, 0.9], device="cuda")
    base = torch.randn(hc3, device="cuda") * 0.1
    x = (torch.randn(6, HIDDEN, device="cuda") * 3).to(torch.bfloat16)
    residual = x.unsqueeze(1).expand(-1, hc, -1).contiguous()
    folded = fn.view(hc3, hc, HIDDEN).sum(dim=1)
    eps = (1e-6, 1e-6, 1e-6)
    ref = mhc_pre_delayed_torch(
        residual, folded, scale, base, *eps, 2.0, iters, pre_mix=None, x=x
    )
    new = mhc_pre_delayed_aiter(
        residual, fn, scale, base, *eps, 2.0, iters, pre_mix=None
    )
    names = ("post_mix", "comb_mix", "layer_input", "next_pre_mix")
    for name, a, b in zip(names, ref, new):
        a, b = a.float().reshape(-1), b.float().reshape(-1)
        if name == "layer_input":
            assert torch.equal(a, b), name
        else:
            assert _rel_err(b, a.double()) < 1e-4, name


# GEMV


def _build_gemv(n, k, s, stage):
    tail = SG.lds_tail(k) + 8

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        xl: fx.Array[fx.Int32, s * G.lds_row(k // 4) + tail, 16]
        xsl: fx.Array[fx.Int32, s * G.lds_row(k // 32) + tail, 16]

    @fx.union
    class U:
        t: Lds

    @flyc.kernel(
        name=f"test_gemv942_{n}_{k}_{s}_{int(stage)}",
        known_block_size=[THREADS, 1, 1],
    )
    def kern(w: Int64, ws: Int64, x8: Int64, x8s: Int64, out: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(U).t.peek()
        c = {"tid": tid, "lane": tid % 64, "wave": tid // 64, "S": s}
        if const_expr(stage):
            ops = SG.gemv_fp8_loads(c, w, ws, k, bid, row_major=True)
        else:
            ops = G.gemv_loads(c, w, ws, k, bid * 16)
        # A plain copy of the rows into LDS, a dword a thread a round.
        for src, dst, width in ((x8, lds.xl.ptr, k // 4), (x8s, lds.xsl.ptr, k // 32)):
            for i in range_constexpr(-(-s * width // THREADS)):
                u = tid + THREADS * i
                if u < s * width:
                    v = fx.Int32(bo.buffer_load(rsrc(src), u, vec_width=1, dtype=T.i32))
                    fx.ptr_store(v, dst + (u // width * G.lds_row(width) + u % width))
        gpu.barrier()
        if const_expr(stage):
            SG.gemv_fp8_mfmas(c, k, lds.xl.ptr, lds.xsl.ptr, lds.red.ptr, ops, rows=s)
        else:
            G.gemv_mfmas(c, k, lds.xl.ptr, lds.xsl.ptr, lds.red.ptr, ops, rows=s)
        gpu.barrier()
        if tid < 16 * s:
            r = tid % 16
            t = tid // 16
            _st(G.row_sum(lds.red.ptr, r, t), out, t * n + bid * 16 + r)

    @flyc.jit
    def launch(
        w: Int64,
        ws: Int64,
        x8: Int64,
        x8s: Int64,
        out: Int64,
    ):
        kern(w, ws, x8, x8s, out).launch(grid=(n // 16,), block=(THREADS,))

    return launch


def _fnuz_rows(x, q_order):
    """Activations [S, K] as the gfx942 kernels quantize them: FNUZ e4m3
    bytes with maximum 224 and an int32 code a group of 32, optionally with
    each 8-byte chunk in even / odd order. Also returns the float64 values."""
    s, k = x.shape
    amax = x.view(s, k // 32, 32).abs().amax(dim=2).clamp_min(1e-30)
    code = (torch.ceil(torch.log2(amax / 224.0)) + 127).clamp(1, 254)
    mul = torch.exp2(code - 127).repeat_interleave(32, 1)
    q = (x / mul).clamp(-224, 224).to(torch.float8_e4m3fnuz)
    b = q.view(torch.uint8)
    if q_order:
        b = b.view(s, k // 8, 8)[..., EVEN_ODD].reshape(s, k)
    return b.contiguous(), code.to(torch.int32), q.double() * mul.double()


@pytest.mark.parametrize("stage", [False, True], ids=["attention", "moe_side"])
def test_gemv_matches_float64(stage):
    n, k, s = 64, 1280, 6
    torch.manual_seed(0)
    # Weights with a different magnitude a block, and values that quantize
    # to -0, which the gfx942 copy must turn into +0.
    w = torch.randn(n, k, device="cuda") * torch.exp2(
        torch.randint(-6, 3, (n // 32, k // 32), device="cuda").float()
    ).repeat_interleave(32, 0).repeat_interleave(32, 1)
    w[:, ::7] = -1e-9
    blk = w.view(n // 32, 32, k // 32, 32).abs().amax(dim=(1, 3)).clamp_min(1e-30)
    code = (torch.ceil(torch.log2(blk / 448.0)) + 127).clamp(1, 254)
    mul = torch.exp2(code - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    w8 = (w / mul).clamp(-448, 448).to(torch.float8_e4m3fn)
    wdeq = w8.double() * mul.double()
    x = torch.randn(s, k, device="cuda") * 3
    x[:, 5] = 400.0
    x8, xcode, xdeq = _fnuz_rows(x, q_order=False)
    wc, wcc = dense_copy(w8.view(torch.uint8), code.to(torch.uint8))
    out = torch.zeros(s, n, device="cuda", dtype=torch.float32)
    _build_gemv(n, k, s, stage)(
        wc.data_ptr(),
        wcc.data_ptr(),
        x8.data_ptr(),
        xcode.data_ptr(),
        out.data_ptr(),
    )
    torch.accelerator.synchronize()
    assert not torch.isnan(out).any()
    assert _rel_err(out, xdeq @ wdeq.t()) < 1e-5


# MoE


def _build_helpers():
    @flyc.kernel(name="test_helpers942", known_block_size=[64, 1, 1])
    def kern(fp8: Int64, ocp: Int64, fp4: Int64, ab: Int64, out: Int64):
        lane = fx.thread_idx.x
        # fp8x8_bf16: 2 FNUZ words and a power-of-two scale (2^(lane % 8 - 3)).
        scale = ((lane % 8 + 124) << 23).bitcast(fx.Float32)
        w = _ld(fp8, 2 * lane, 2)
        _st(fp8x8_bf16(w[0], w[1], scale), out, 4 * lane)
        _st(ocp_to_fnuz(_ld(ocp, lane)), out, 256 + lane)
        even, odd = fp4x8_fnuz(_ld(fp4, lane))
        _st(fx.Vector.from_elements([even, odd], fx.Int32), out, 320 + 2 * lane)
        # Lane l holds A row l % 16 and B column l % 16, K 8 (l / 16) to
        # 8 (l / 16) + 7, as 4 dwords of bf16 pairs each.
        a = _ld(ab, 4 * lane, 4).bitcast(fx.BFloat16)
        b = _ld(ab, 256 + 4 * lane, 4).bitcast(fx.BFloat16)
        cc = mfma_bf16_k32(a, b, fx.Vector.filled(4, 0.0, fx.Float32))
        _st(cc.bitcast(fx.Int32), out, 448 + 4 * lane)

    @flyc.jit
    def launch(
        fp8: Int64,
        ocp: Int64,
        fp4: Int64,
        ab: Int64,
        out: Int64,
    ):
        kern(fp8, ocp, fp4, ab, out).launch(grid=(1,), block=(64,))

    return launch


def test_gfx942_conversion_helpers():
    g = torch.Generator(device="cuda").manual_seed(1)
    u8 = dict(device="cuda", dtype=torch.uint8, generator=g)
    fnuz = torch.randint(0, 256, (64 * 8,), **u8)
    fnuz[fnuz == 0x80] = 0  # 0x80 is the FNUZ NaN, which a record never holds.
    ocp = torch.randint(0, 256, (64 * 4,), **u8)
    ocp[(ocp & 0x7F) == 0x7F] = 0  # These are the OCP NaN codes.
    ocp[:8] = 0x80  # These are -0.
    fp4 = torch.randint(0, 256, (64 * 4,), **u8)
    fp4_drop_negative_zero_(fp4)
    a = torch.randn(16, 32, device="cuda").bfloat16()
    b = torch.randn(16, 32, device="cuda").bfloat16()
    a_l = a.view(16, 4, 8).permute(1, 0, 2).reshape(64, 8)
    b_l = b.view(16, 4, 8).permute(1, 0, 2).reshape(64, 8)
    ab = torch.cat([a_l.reshape(-1), b_l.reshape(-1)]).contiguous()
    out = torch.zeros(448 + 256, device="cuda", dtype=torch.int32)
    _build_helpers()(
        fnuz.data_ptr(),
        ocp.data_ptr(),
        fp4.data_ptr(),
        ab.data_ptr(),
        out.data_ptr(),
    )
    torch.accelerator.synchronize()
    lanes = torch.arange(64, device="cuda")
    scale = torch.exp2((lanes % 8 - 3).double()).repeat_interleave(8)
    ref = fnuz.view(torch.float8_e4m3fnuz).double() * scale
    assert torch.equal(out[:256].view(torch.bfloat16).double(), ref), "fp8x8_bf16"
    # An FNUZ byte is half the OCP value of the same bits.
    got = out[256:320].contiguous().view(torch.uint8).view(torch.float8_e4m3fnuz)
    ref = ocp.view(torch.float8_e4m3fn).double()
    assert torch.equal(got.double() * 2, ref), "ocp_to_fnuz"
    # 2^7 times the FNUZ bytes are the e2m1 values, the even K first.
    got = out[320:448].contiguous().view(torch.uint8).view(64, 8)
    nib = torch.stack([fp4 & 0xF, fp4 >> 4], dim=1).view(64, 8).long()
    ref = E2M1.cuda()[nib][:, EVEN_ODD]
    assert torch.equal(got.view(torch.float8_e4m3fnuz).double() * 128, ref), "fp4"
    # Lane l holds C rows 4 (l / 16) to 4 (l / 16) + 3 of column l % 16.
    cc = out[448:].view(torch.float32).view(4, 16, 4)
    got = cc.permute(0, 2, 1).reshape(16, 16).double()
    err = (got - a.double() @ b.double().t()).abs().max().item()
    assert err < 1e-4, "mfma_bf16_k32"


def _fp4_weights(rows, k, g):
    """Random MXFP4 rows in vLLM's layout: bytes [rows, k / 2] with the low
    nibble the lower K and -0 removed, e8m0 codes [rows, k / 32], and the
    float64 values."""
    u8 = dict(device="cuda", dtype=torch.uint8, generator=g)
    wq = torch.randint(0, 256, (rows, k // 2), **u8)
    fp4_drop_negative_zero_(wq)
    code = torch.randint(112, 128, (rows, k // 32), **u8)
    nib = torch.stack([wq & 0xF, wq >> 4], dim=2).view(rows, k).long()
    scale = torch.exp2(code.double() - 127).repeat_interleave(32, 1)
    return wq, code, E2M1.cuda()[nib] * scale


def _build_ug(s, n_rows, k, slices):
    x_row, s_row = G.lds_row(k // 4), G.lds_row(k // 32)
    pairs_cta = WAVES // slices

    @fx.struct
    class Lds:
        xl: fx.Array[fx.Int32, s * x_row + 8, 16]
        xsl: fx.Array[fx.Int32, s * s_row + 8, 16]

    @fx.union
    class U:
        t: Lds

    @flyc.kernel(name=f"test_ug942_{s}_{n_rows}_{k}", known_block_size=[THREADS, 1, 1])
    def kern(w: Int64, ws: Int64, x8: Int64, x8s: Int64, out: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane, wave = tid % 64, tid // 64
        lds = fx.SharedAllocator().allocate(U).t.peek()
        for src, dst, width, row in (
            (x8, lds.xl.ptr, k // 4, x_row),
            (x8s, lds.xsl.ptr, k // 32, s_row),
        ):
            for i in range_constexpr(-(-s * width // THREADS)):
                u = tid + THREADS * i
                if u < s * width:
                    fx.ptr_store(_ld(src, u), dst + (u // width * row + u % width))
        gpu.barrier()
        p = bid * pairs_cta + wave // slices
        e = p // (n_rows // 32)
        rg0 = 2 * (p % (n_rows // 32))
        ks = wave % slices
        col = fx.min(lane % 16, s - 1)
        c = {"lane": lane, "rs": SimpleNamespace(ug_k_slices=slices)}
        acc = M.gemv_fp4_pair_942(
            c, w, ws, e, n_rows, k, rg0, ks, lds.xl.ptr, lds.xsl.ptr, col
        )
        if lane % 16 < s:
            for t in range_constexpr(2):
                for q in range_constexpr(4):
                    row = (rg0 + t) * 16 + 4 * (lane // 16) + q
                    dst = ((ks * 64 + e) * s + lane % 16) * n_rows + row
                    _st(acc[4 * t + q], out, dst)

    @flyc.jit
    def launch(
        w: Int64,
        ws: Int64,
        x8: Int64,
        x8s: Int64,
        out: Int64,
        grid: fx.Int32,
    ):
        kern(w, ws, x8, x8s, out).launch(grid=(grid,), block=(THREADS,))

    return launch


def test_moe_up_gate_matches_float64():
    """gemv_fp4_pair_942 with FP4 rows in the copy's fp4_tile_major byte
    order, e8m0 codes in codes_tile_rounds order, and activations in LDS as
    FNUZ bytes with each 8-byte chunk in even / odd order."""
    s, experts, n_rows, k, slices = 6, 2, 1152, HIDDEN, 2
    g = torch.Generator(device="cuda").manual_seed(2)
    wq, code, deq = _fp4_weights(experts * n_rows, k, g)
    x = torch.randn(s, k, device="cuda", generator=g) * 2
    x[:, 3] = 300.0
    x8, xcode, xdeq = _fnuz_rows(x, q_order=True)
    out = torch.zeros(slices, 64, s, n_rows, device="cuda", dtype=torch.float32)
    grid = experts * (n_rows // 32) // (WAVES // slices)
    _build_ug(s, n_rows, k, slices)(
        fp4_tile_major(wq).data_ptr(),
        codes_tile_rounds(code, M.UG_ROUND_942).data_ptr(),
        x8.data_ptr(),
        xcode.data_ptr(),
        out.data_ptr(),
        grid,
    )
    torch.accelerator.synchronize()
    got = out.sum(0)[:experts]
    ref = torch.einsum("sk,erk->esr", xdeq, deq.view(experts, n_rows, k))
    assert not torch.isnan(got).any()
    assert _rel_err(got, ref) < 1e-5


def _build_down(s, experts):
    d = SimpleNamespace(inter_real=576, inter=640, mid_words=160)

    @flyc.kernel(name=f"test_down942_{s}", known_block_size=[THREADS, 1, 1])
    def kern(w2: Int64, w2s: Int64, mid: Int64, mids: Int64, out: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane, wave = tid % 64, tid // 64
        task = bid * WAVES + wave
        e = task // (HIDDEN // 16)
        tt = task % (HIDDEN // 16)
        c = {
            "lane": lane,
            "args": {"w2": w2, "w2_s": w2s},
            "d": d,
            "mid": mid,
            "mids": mids,
        }
        wd, sc = M.down_weights_942(c, e, tt)
        bvs, sxs = M.mid_operands_942(c, 0, s)
        kc = d.inter_real // 32
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for b in range_constexpr(kc):
            codes = [M.code_byte(sc, q * kc + b) for q in range(4)]
            acc = M.fp4_mfma_942(acc, wd[b], bvs[b], codes, sxs[b] - 120)
        if lane % 16 < s:
            for q in range_constexpr(4):
                row = e * HIDDEN + tt * 16 + 4 * (lane // 16) + q
                _st(acc[q], out, (lane % 16) * (experts * HIDDEN) + row)

    @flyc.jit
    def launch(
        w2: Int64,
        w2s: Int64,
        mid: Int64,
        mids: Int64,
        out: Int64,
        grid: fx.Int32,
    ):
        kern(w2, w2s, mid, mids, out).launch(grid=(grid,), block=(THREADS,))

    return launch


def test_moe_down_matches_float64():
    """down_weights_942, mid_operands_942 and fp4_mfma_942 with w2 in the
    copy's fp4_tile_major byte order, against intermediate rows in the layout
    that ug_mid_store_942 writes."""
    s, experts = 6, 2
    g = torch.Generator(device="cuda").manual_seed(3)
    wq, code, deq = _fp4_weights(experts * HIDDEN, 576, g)
    y = torch.randn(s, 576, device="cuda", generator=g)
    y[:, 7] = -50.0
    y8, ycode, ydeq = _fnuz_rows(y, q_order=False)
    mid = torch.zeros(s, 640, device="cuda", dtype=torch.uint8)
    mid[:, :576] = y8
    # Within each 128-byte step, the four 8-byte chunks of lane group j are
    # together, and each chunk holds the even K first.
    mid = mid.view(s, 5, 4, 4, 8)[..., EVEN_ODD].permute(0, 1, 3, 2, 4)
    mid = mid.contiguous().view(s, 640)
    mids = torch.full((s, 20), 127, device="cuda", dtype=torch.int32)
    mids[:, :18] = ycode
    out = torch.zeros(s, experts * HIDDEN, device="cuda", dtype=torch.float32)
    _build_down(s, experts)(
        fp4_tile_major(wq).data_ptr(),
        code.data_ptr(),
        mid.data_ptr(),
        mids.data_ptr(),
        out.data_ptr(),
        experts * HIDDEN // 16 // WAVES,
    )
    torch.accelerator.synchronize()
    assert not torch.isnan(out).any()
    assert _rel_err(out, ydeq @ deq.t()) < 1e-5
