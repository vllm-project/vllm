# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exactness of the Qwen3.8-27B dense mono decode kernels (ROCm / FlyDSL,
gfx950) against the stock decode path, on random MXFP4 weights.

``DenseFfnTest`` (K2, ``dense_ffn``): out_proj / o_proj + the residual add +
post_attention_layernorm + gate_up, SiLU-mul and down, against dynamic_mxfp4_quant
+ Triton ``gemm_afp4wfp4``, the ROCm Triton Gemma norm and SiluAndMul. At TP2
(from the stock norm's output) each rank's down partial before the all-reduce,
from its half of gate_up's rows and down's K columns.

``RowCapsTest``: K1's and K2's row caps by TP and their env overrides.

``StockAccumulationTest``: on typical inputs at K2's K sizes the stock MXFP4
GEMM's output is bf16(fp32(exact sum)), so a mismatch there is a K2 bug rather
than a rounding corner. ``StockSplitTest``: the split counts K2 follows are the
stock GEMM's at every width. ``StockChainTest``: K2 is exact by construction, on
rows whose exact sums need more than 24 bits, where the same GEMMs summed in K
quarters differ from stock.

``DenseGdnTest`` (K1, ``dense_gdn``): the residual add + input_layernorm ->
in_proj_qkvz / in_proj_ba -> conv1d -> the gated delta rule -> the gated norm,
against the ROCm Triton Gemma norm, the MXFP4 linears, ``causal_conv1d_update``,
``fused_recurrent_gated_delta_rule_packed_decode`` and RMSNormGated. The core,
the residual and both states must match, on the first layer (no residual) and
with pad rows (slot -1); other slots stay untouched. At TP2 (from the stock input
norm's output, as the runtime runs it there) each rank's half of the heads, its
column-parallel slices of the same layer, against the stock path on them, with
out_proj folded in (its partial, from the rank's K columns) and without.

Widths 1..8 at TP1 and 1..16 at TP2, each with its own weights and Mailbox
layer tag, over many input draws: a norm reduced in another order than the stock
kernel's still matches on most rows, and shows on about one row in 640.
Run directly with ``pytest tests/models/qwen3_5/test_amd_dense_mono.py``.
"""

import os
import unittest
from contextlib import contextmanager

import torch


@contextmanager
def _env(name, value):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


BF = torch.bfloat16
EPS = 1e-6
FFN_DRAWS = 512
GDN_DRAWS = 512
# Mailbox tag slots: the layer index a launch passes as ``layer``.
LAYER_TAGS = (0, 17, 63)


def _gfx950() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    )


def _mxfp4(g, n, k):
    """Random MXFP4 [n, k]: fp4 pairs and e8m0 scales 2^-7 .. 2^-1."""
    w = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    ws = torch.randint(
        120, 127, (n, k // 32), dtype=torch.uint8, device="cuda", generator=g
    )
    return w, ws


def _widths(tp):
    from vllm.models.qwen3_5.amd.mono import dense_ffn as D

    return range(1, D.max_tokens(tp) + 1)


def _rnd(g, *shape, scale=1.0, dtype=BF):
    return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)


def _linear(x, w, ws):
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    xq, xs = dynamic_mxfp4_quant(x)
    y = torch.empty(x.size(0), w.size(0), dtype=BF, device=x.device)
    gemm_afp4wfp4(xq, w, xs, ws, BF, y)
    return y


def _k_cols(w, ws, k, tp, rank):
    """Row-parallel: the rank's input columns of an MXFP4 [n, k]."""
    n = k // tp
    return (
        w[:, rank * n // 2 : (rank + 1) * n // 2].contiguous(),
        ws[:, rank * n // 32 : (rank + 1) * n // 32].contiguous(),
    )


_FP4 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _dequant(wq, ws):
    """MXFP4 [n, k / 2] (low nibble the even column), e8m0 [n, k / 32] -> fp64."""
    lut = torch.tensor(_FP4, dtype=torch.float64, device=wq.device)
    v = torch.stack([lut[(wq & 0xF).long()], lut[(wq >> 4).long()]], -1)
    s = torch.pow(2.0, ws.double() - 127.0).repeat_interleave(32, dim=1)
    return v.reshape(wq.size(0), -1) * s


def _gemma_norm(x, residual, w):
    """GemmaRMSNorm as the stock path runs it on ROCm with aiter -> x, residual."""
    import vllm.models.minimax_m3.amd.ops.gemma_rmsnorm as M

    if residual is None:
        return M.gemma_rmsnorm(x, w, EPS), x
    return M.gemma_fused_add_rmsnorm(x, residual, w, EPS)


@unittest.skipUnless(_gfx950(), "needs FlyDSL and aiter on gfx950")
class DenseFfnTest(unittest.TestCase):
    def test_bit_exact(self):
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D
        from vllm.model_executor.layers.activation import SiluAndMul

        act = SiluAndMul()
        for s in _widths(1):
            g = torch.Generator(device="cuda").manual_seed(s)
            layer = LAYER_TAGS[s % len(LAYER_TAGS)]
            w_o, w_os = _mxfp4(g, D.HIDDEN, D.CORE)
            w13, w13s = _mxfp4(g, 2 * D.INTER, D.HIDDEN)
            w2, w2s = _mxfp4(g, D.HIDDEN, D.INTER)
            ln_w = _rnd(g, D.HIDDEN, scale=0.1)
            weights = dict(
                w_o=D.shuffle_w(w_o),
                w_os=w_os,
                ln_w=ln_w,
                w13=D.shuffle_w(w13),
                w13s=w13s,
                w2=D.shuffle_w(w2),
                w2s=w2s,
            )
            key = D.FfnBuild(tokens=s, eps=EPS, shuffled=True, oproj=True)
            out = torch.empty(s, D.HIDDEN, dtype=BF, device="cuda")
            res_out = torch.empty_like(out)
            epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
            # One scratch for every launch: the epoch tags must keep it valid.
            scratch = torch.zeros(
                D.scratch_bytes(key), dtype=torch.uint8, device="cuda"
            )
            bad = []
            for draw in range(FFN_DRAWS):
                core = _rnd(g, s, D.CORE, scale=0.5)
                residual = _rnd(g, s, D.HIDDEN, scale=2.0)
                x, res_ref = _gemma_norm(_linear(core, w_o, w_os), residual, ln_w)
                out_ref = _linear(act(_linear(x, w13, w13s)), w2, w2s)
                epoch.add_(1)
                D.dense_ffn(
                    key,
                    hidden=core,
                    residual=residual,
                    out=out,
                    res_out=res_out,
                    scratch=scratch,
                    epoch=epoch,
                    layer=layer,
                    **weights,
                )
                if not (torch.equal(res_out, res_ref) and torch.equal(out, out_ref)):
                    bad.append(draw)
            with self.subTest(s=s, layer=layer):
                self.assertEqual(bad, [])

    def test_bit_exact_tp2(self):
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D
        from vllm.model_executor.layers.activation import SiluAndMul

        act = SiluAndMul()
        inter = D.FGeo(2).inter
        for rank in range(2):
            for s in _widths(2):
                g = torch.Generator(device="cuda").manual_seed(200 + s)
                layer = LAYER_TAGS[s % len(LAYER_TAGS)]
                w13, w13s = _mxfp4(g, 2 * D.INTER, D.HIDDEN)
                w2, w2s = _mxfp4(g, D.HIDDEN, D.INTER)
                ln_w = _rnd(g, D.HIDDEN, scale=0.1)
                # Column-parallel gate_up: the rank's half of gate, then of up.
                rows = [
                    slice(o + rank * inter, o + (rank + 1) * inter)
                    for o in (0, D.INTER)
                ]
                w13 = torch.cat([w13[r] for r in rows]).contiguous()
                w13s = torch.cat([w13s[r] for r in rows]).contiguous()
                w2, w2s = _k_cols(w2, w2s, D.INTER, 2, rank)
                w13_sh, w2_sh = D.shuffle_w(w13), D.shuffle_w(w2)
                key = D.FfnBuild(tokens=s, shuffled=True, tp=2, normed=True)
                out = torch.empty(s, D.HIDDEN, dtype=BF, device="cuda")
                epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
                scratch = torch.zeros(
                    D.scratch_bytes(key), dtype=torch.uint8, device="cuda"
                )
                bad = []
                for draw in range(FFN_DRAWS // 2):
                    h = _rnd(g, s, D.HIDDEN, scale=0.5)
                    x, _ = _gemma_norm(h, _rnd(g, s, D.HIDDEN, scale=2.0), ln_w)
                    out_ref = _linear(act(_linear(x, w13, w13s)), w2, w2s)
                    epoch.add_(1)
                    D.dense_ffn(
                        key,
                        hidden=x,
                        residual=None,
                        ln_w=None,
                        w13=w13_sh,
                        w13s=w13s,
                        w2=w2_sh,
                        w2s=w2s,
                        out=out,
                        res_out=None,
                        scratch=scratch,
                        epoch=epoch,
                        layer=layer,
                    )
                    if not torch.equal(out, out_ref):
                        bad.append(draw)
                with self.subTest(rank=rank, s=s, layer=layer):
                    self.assertEqual(bad, [])


@unittest.skipUnless(_gfx950(), "needs FlyDSL and aiter on gfx950")
class RowCapsTest(unittest.TestCase):
    def test_tp_defaults_and_overrides(self):
        from vllm.models.qwen3_5.amd.dense_mono import row_caps

        cases = (
            (None, None, (4, 4), (8, 8)),
            ("6", "2", (6, 2), (6, 2)),
            (None, "16", (4, 8), (8, 16)),
            ("0", "99", (1, 8), (1, 16)),
        )
        keys = (
            "VLLM_ROCM_QWEN38_DENSE_MONO_K1_MAX_ROWS",
            "VLLM_ROCM_QWEN38_DENSE_MONO_K2_MAX_ROWS",
        )
        for v1, v2, want1, want2 in cases:
            with self.subTest(k1=v1, k2=v2), _env(keys[0], v1), _env(keys[1], v2):
                self.assertEqual(row_caps(1), want1)
                self.assertEqual(row_caps(2), want2)


@unittest.skipUnless(_gfx950(), "needs aiter on gfx950")
class StockAccumulationTest(unittest.TestCase):
    # K2's GEMM K sizes: gate_up, down at TP1, down at TP2.
    KS = (5120, 17408, 8704)

    def test_stock_gemm_is_exact_sum_rounded(self):
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        g = torch.Generator(device="cuda").manual_seed(7)
        for k in self.KS:
            w, ws = _mxfp4(g, 512, k)
            wd = _dequant(w, ws)
            bad = 0
            for draw in range(8):
                x = torch.randn(8, k, generator=g, device="cuda")
                if draw % 2:  # heavy-tailed rows
                    x = x * torch.randn(8, k, generator=g, device="cuda").mul(2).exp()
                x = x.to(BF)
                xq, xs = dynamic_mxfp4_quant(x)
                exact = _dequant(xq.view(torch.uint8), xs.view(torch.uint8)) @ wd.T
                bad += (_linear(x, w, ws) != exact.float().to(BF)).sum().item()
            with self.subTest(k=k):
                self.assertEqual(bad, 0)


def _linear_parts(x, w, ws, splits=None, block_k=128):
    """The stock GEMM's fp32 split partials [n, M, N] (``splits``: forced to
    that many equal K splits), or its bf16 output [M, N] for one split."""
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    xq, xs = dynamic_mxfp4_quant(x)
    cfg = None
    if splits is not None:
        cfg = dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=1,
            num_warps=4,
            num_stages=2,
            waves_per_eu=2,
            matrix_instr_nonkdim=16,
            cache_modifier=None,
            NUM_KSPLIT=splits,
        )
    return gemm_afp4wfp4(xq, w, xs, ws, BF, config=cfg, skip_reduce=True)


def _linear_quartered(x, w, ws):
    """The GEMM as four K quarters, each chained from zero, summed in order."""
    p = _linear_parts(x, w, ws, splits=4)
    assert p.dim() == 3 and p.size(0) == 4
    return (((p[0] + p[1]) + p[2]) + p[3]).to(BF)


@unittest.skipUnless(_gfx950(), "needs aiter on gfx950")
class StockSplitTest(unittest.TestCase):
    def test_k2_follows_stock_splits(self):
        """K2's split counts are the stock GEMM's at every width (from the
        partials its config yields)."""
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D

        g = torch.Generator(device="cuda").manual_seed(3)
        for tp in D.TP_SIZES:
            inter = D.FGeo(tp).inter
            shapes = [("gate_up", 2 * inter, D.HIDDEN, D.gu_splits)]
            shapes.append(("down", D.HIDDEN, inter, lambda s: D.d_splits(s, tp)))
            if tp == 1:
                shapes.append(("o_proj", D.HIDDEN, D.CORE, lambda s: D.O_SPLITS))
            for name, n, k, want in shapes:
                w, ws = _mxfp4(g, n, k)
                for s in _widths(tp):
                    p = _linear_parts(_rnd(g, s, k), w, ws)
                    got = p.size(0) if p.dim() == 3 else 1
                    with self.subTest(tp=tp, gemm=name, s=s):
                        self.assertEqual(got, want(s))


def _cancel_blocks(w, ws, rows, blocks, g):
    """Rows ``rows`` of an MXFP4 [n, k] zeroed but for ``blocks``: (block, kind),
    kind "big" (random fp4, scale 2^30), "neg" (the previous big block negated)
    or "small" (random, scale 2^0)."""
    k_blocks = ws.size(1)
    for r in rows:
        w[r] = 0
        ws[r] = 127
        prev = None
        for b, kind in blocks:
            assert 0 <= b < k_blocks
            cols = slice(16 * b, 16 * b + 16)
            if kind == "neg":
                w[r, cols] = w[r, 16 * prev : 16 * prev + 16] ^ 0x88
                ws[r, b] = ws[r, prev]
                continue
            w[r, cols] = torch.randint(
                0, 256, (16,), dtype=torch.uint8, device="cuda", generator=g
            )
            ws[r, b] = 157 if kind == "big" else 127
            prev = b


def _copy_block(t, dst, src):
    """Columns 32 dst .. of ``t`` (bf16 [.., k]) := columns 32 src .."""
    t[..., 32 * dst : 32 * dst + 32] = t[..., 32 * src : 32 * src + 32]


@unittest.skipUnless(_gfx950(), "needs FlyDSL and aiter on gfx950")
class StockChainTest(unittest.TestCase):
    """K2 reproduces the stock GEMMs' arithmetic, not only their exact sums: on
    rows whose exact sums span more than 24 bits (a 2^30-scaled block and its
    negation, small blocks next to them), stock differs from the same GEMMs
    summed in K quarters, and K2 still matches stock. Gate rows (seen through
    down rows reading only their columns) and down rows each get a pair across
    a quarter boundary inside a split (P1) and across the split boundary (P2)."""

    def _weights(self, g, hidden, inter):
        w13, w13s = _mxfp4(g, 2 * inter, hidden)
        w2, w2s = _mxfp4(g, hidden, inter)
        for w, ws, rows, k in ((w13, w13s, (0, 4), hidden), (w2, w2s, (0, 4), inter)):
            nb = k // 32
            qb, sb = nb // 4, nb // 2
            p1 = [(qb - 1, "big"), (qb, "neg"), (qb + 1, "small")]
            p2 = [(1, "small"), (sb - 1, "big"), (sb, "neg"), (nb - 1, "small")]
            _cancel_blocks(w, ws, range(rows[0], rows[0] + 4), p1, g)
            _cancel_blocks(w, ws, range(rows[1], rows[1] + 4), p2, g)
        # down's big blocks read equal H blocks: gate / up rows copied
        nb = inter // 32
        for dst, src in ((nb // 4, nb // 4 - 1), (nb // 2, nb // 2 - 1)):
            for base in (0, inter):
                w13[base + 32 * dst : base + 32 * dst + 32] = w13[
                    base + 32 * src : base + 32 * src + 32
                ]
                w13s[base + 32 * dst : base + 32 * dst + 32] = w13s[
                    base + 32 * src : base + 32 * src + 32
                ]
        # down rows 8 .. 11 read the crafted gate rows' columns (H block 0) only
        w2[8:12, 16:] = 0
        return w13, w13s, w2, w2s

    def test_stock_chains(self):
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D
        from vllm.model_executor.layers.activation import SiluAndMul

        act = SiluAndMul()
        for tp in D.TP_SIZES:
            inter = D.FGeo(tp).inter
            g = torch.Generator(device="cuda").manual_seed(40 + tp)
            w13, w13s, w2, w2s = self._weights(g, D.HIDDEN, inter)
            w13_sh, w2_sh = D.shuffle_w(w13), D.shuffle_w(w2)
            nb = D.HIDDEN // 32
            for s in _widths(tp):
                h = _rnd(g, s, D.HIDDEN, scale=0.5)
                r = _rnd(g, s, D.HIDDEN, scale=2.0)
                ln_w = _rnd(g, D.HIDDEN, scale=0.1)
                # the gate rows' big blocks read equal x blocks
                for t in (h, r, ln_w):
                    _copy_block(t, nb // 4, nb // 4 - 1)
                    _copy_block(t, nb // 2, nb // 2 - 1)
                x, res_ref = _gemma_norm(h, r, ln_w)
                ref = _linear(act(_linear(x, w13, w13s)), w2, w2s)
                quart = _linear_quartered(act(_linear_quartered(x, w13, w13s)), w2, w2s)
                epoch = torch.ones(1, dtype=torch.int32, device="cuda")
                out = torch.empty(s, D.HIDDEN, dtype=BF, device="cuda")
                if tp == 1:
                    key = D.FfnBuild(tokens=s, eps=EPS, shuffled=True)
                    args = dict(hidden=h, residual=r, ln_w=ln_w)
                    res_out = torch.empty_like(out)
                else:
                    key = D.FfnBuild(tokens=s, shuffled=True, tp=2, normed=True)
                    args = dict(hidden=x, residual=None, ln_w=None)
                    res_out = None
                scratch = torch.zeros(
                    D.scratch_bytes(key), dtype=torch.uint8, device="cuda"
                )
                D.dense_ffn(
                    key,
                    **args,
                    w13=w13_sh,
                    w13s=w13s,
                    w2=w2_sh,
                    w2s=w2s,
                    out=out,
                    res_out=res_out,
                    scratch=scratch,
                    epoch=epoch,
                    layer=0,
                )
                with self.subTest(tp=tp, s=s):
                    self.assertFalse(torch.equal(quart, ref))
                    self.assertTrue(torch.equal(out, ref))
                    if tp == 1:
                        self.assertTrue(torch.equal(res_out, res_ref))


def _gdn_weights(g, K):
    from vllm.model_executor.layers.layernorm import RMSNormGated

    w = {
        "ln_w": _rnd(g, K.HIDDEN, scale=0.1),
        "conv_w": _rnd(g, K.CONV, 1, K.CONV_W, scale=0.4),
        "a_log": _rnd(g, K.NV, scale=0.5, dtype=torch.float32),
        "dt_bias": _rnd(g, K.NV, scale=0.5),
        "norm_w": (1 + _rnd(g, K.HD, scale=0.1, dtype=torch.float32)).to(BF),
    }
    w["w_qkvz"], w["w_qkvzs"] = _mxfp4(g, K.QKVZ, K.HIDDEN)
    w["w_ba"], w["w_bas"] = _mxfp4(g, K.BA, K.HIDDEN)
    w["w_o"], w["w_os"] = _mxfp4(g, K.HIDDEN, K.VAL)
    gn = RMSNormGated(
        K.HD, eps=EPS, group_size=None, norm_before_gate=True, dtype=BF
    ).cuda()
    gn.weight.data.copy_(w["norm_w"])
    w["gated_norm"] = gn
    return w


def _rank_slice(K, w, tp, rank):
    """The rank's column-parallel share of a GDN layer: each of q | k | v | z,
    b | a, the conv's q | k | v and the heads' A_log / dt_bias cut in tp."""
    if tp == 1:
        return w
    geo = K.Geo(tp)

    def rows(t, parts):
        return torch.cat(
            [t[o + rank * n : o + (rank + 1) * n] for o, n in parts]
        ).contiguous()

    qkvz = [(0, geo.key), (K.KEY, geo.key), (2 * K.KEY, geo.val), (K.CONV, geo.val)]
    heads = slice(rank * geo.nv, (rank + 1) * geo.nv)
    out = dict(w)
    out["w_qkvz"] = rows(w["w_qkvz"], qkvz)
    out["w_qkvzs"] = rows(w["w_qkvzs"], qkvz)
    out["w_ba"] = rows(w["w_ba"], [(0, geo.nv), (K.NV, geo.nv)])
    out["w_bas"] = rows(w["w_bas"], [(0, geo.nv), (K.NV, geo.nv)])
    out["conv_w"] = rows(w["conv_w"], qkvz[:3])
    out["a_log"] = w["a_log"][heads].contiguous()
    out["dt_bias"] = w["dt_bias"][heads].contiguous()
    out["w_o"], out["w_os"] = _k_cols(w["w_o"], w["w_os"], K.VAL, tp, rank)
    return out


def _stock_gdn(K, geo, w, h, residual, conv, ssm, idx):
    """The rank's heads (``geo``) -> core, residual out; ``conv`` / ``ssm``
    updated in place."""
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    x, res = _gemma_norm(h, residual, w["ln_w"])
    qkvz = _linear(x, w["w_qkvz"], w["w_qkvzs"])
    ba = _linear(x, w["w_ba"], w["w_bas"])
    q, k, v, z = qkvz.split([geo.key, geo.key, geo.val, geo.val], dim=-1)
    b, a = ba.split([geo.nv, geo.nv], dim=-1)
    mixed = causal_conv1d_update(
        torch.cat([q, k, v], dim=-1),
        conv,
        w["conv_w"].view(geo.conv, K.CONV_W),
        None,
        "silu",
        conv_state_indices=idx,
        null_block_id=-1,
    )
    o = torch.empty(h.size(0), 1, geo.nv, K.HD, dtype=BF, device="cuda")
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed,
        a=a.contiguous(),
        b=b.contiguous(),
        A_log=w["a_log"],
        dt_bias=w["dt_bias"],
        scale=K.HD**-0.5,
        initial_state=ssm,
        out=o,
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
    )
    core = w["gated_norm"](o.view(-1, K.HD), z.reshape(-1, K.HD))
    return core.view(h.size(0), geo.val), res


@unittest.skipUnless(_gfx950(), "needs FlyDSL and aiter on gfx950")
class DenseGdnTest(unittest.TestCase):
    SLOTS = 20  # slots 2 .. 17 at width 16, the rest untouched

    def test_bit_exact(self):
        self._check(tp=1, rank=0, normed=False)

    def test_bit_exact_tp2(self):
        for rank in range(2):
            self._check(tp=2, rank=rank, normed=True, oproj=True)

    def test_bit_exact_tp2_stock_out_proj(self):
        self._check(tp=2, rank=0, normed=True, draws=GDN_DRAWS // 4)

    def _check(self, tp, rank, normed, oproj=False, draws=GDN_DRAWS):
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D
        from vllm.models.qwen3_5.amd.mono import dense_gdn as K

        geo = K.Geo(tp)
        for s in _widths(tp):
            first = s in (1, 5) and not normed
            g = torch.Generator(device="cuda").manual_seed(100 + s)
            layer = LAYER_TAGS[s % len(LAYER_TAGS)]
            w = _rank_slice(K, _gdn_weights(g, K), tp, rank)
            w_qkvz, w_ba = D.shuffle_w(w["w_qkvz"]), D.shuffle_w(w["w_ba"])
            key = K.K1Build(
                tokens=s,
                first=first,
                eps=EPS,
                norm_eps=EPS,
                tp=tp,
                normed=normed,
                oproj=oproj,
            )
            fold = {}
            if oproj:
                out = torch.empty(s, K.HIDDEN, dtype=BF, device="cuda")
                fold = dict(w_o=D.shuffle_w(w["w_o"]), w_os=w["w_os"], out=out)
            core = torch.empty(s, geo.val, dtype=BF, device="cuda")
            res_out = torch.empty(s, K.HIDDEN, dtype=BF, device="cuda")
            epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
            scratch = torch.zeros(
                K.scratch_bytes(key), dtype=torch.uint8, device="cuda"
            )
            idx = torch.arange(2, 2 + s, dtype=torch.int32, device="cuda")
            if s > 1:
                idx[1] = -1  # a pad row
            bad = []
            for draw in range(draws):
                h = _rnd(g, s, K.HIDDEN, scale=0.5)
                residual = None if first else _rnd(g, s, K.HIDDEN, scale=2.0)
                # The mamba pool's layout: conv (slots, dim, 3) dim-first, SSM fp32.
                conv = _rnd(g, self.SLOTS, geo.conv, K.SL)
                ssm = _rnd(
                    g, self.SLOTS, geo.nv, K.HD, K.HD, scale=0.05, dtype=torch.float32
                )
                conv_ref, ssm_ref = conv.clone(), ssm.clone()
                core_ref, res_ref = _stock_gdn(
                    K, geo, w, h, residual, conv_ref, ssm_ref, idx
                )
                core.fill_(7.0)
                epoch.add_(1)
                if normed:
                    x, _ = _gemma_norm(h, residual, w["ln_w"])
                    front = dict(hidden=x)
                else:
                    front = dict(
                        hidden=h, residual=residual, res_out=res_out, ln_w=w["ln_w"]
                    )
                K.dense_gdn_pre(
                    key,
                    **front,
                    w_qkvz=w_qkvz,
                    w_qkvzs=w["w_qkvzs"],
                    w_ba=w_ba,
                    w_bas=w["w_bas"],
                    conv_w=w["conv_w"],
                    conv_state=conv,
                    a_log=w["a_log"],
                    dt_bias=w["dt_bias"],
                    norm_w=w["norm_w"],
                    rstate=ssm,
                    st_idx=idx,
                    core=core,
                    scratch=scratch,
                    epoch=epoch,
                    layer=layer,
                    **fold,
                )
                exact = (
                    (normed or torch.equal(res_out, res_ref))
                    and torch.equal(core, core_ref)
                    and torch.equal(conv, conv_ref)
                    and torch.equal(ssm, ssm_ref)
                )
                if oproj:
                    o_ref = _linear(core_ref, w["w_o"], w["w_os"])
                    exact = exact and torch.equal(out, o_ref)
                if s > 1:
                    exact = exact and not core[1].any()
                if not exact:
                    bad.append(draw)
            with self.subTest(
                tp=tp, rank=rank, s=s, first=first, oproj=oproj, layer=layer
            ):
                self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
