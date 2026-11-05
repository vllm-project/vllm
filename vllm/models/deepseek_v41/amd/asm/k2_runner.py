# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2, the FFN half of a decoder layer (from the attention all-reduce's output to the
MoE all-reduce's input), as one callable.

    run = K2Runner()
    wt = run.weights(raw_from_layer(layer))
    token, r_new, post, comb, pre = run(wt, x, residual, post_mix, res_mix, attn_pre)
    out = comb_ar(*run.comb_in, m)  # ar_runner.CombAllReduce

Six launches: the seam (MXFP8 + bf16 y' = x1 · ffn_norm.w without rstd), router
logits + shared gate_up (one launch, kernels/multi.py), p3 (rstd, router finish,
shared SwiGLU, the seam's mixes on a 5th wave), MoE prep + shared down (one
launch), routed stage 1 (scales g / u by rstd) and stage 2 (y in bf16). The
combine with all-reduce #2 is ar_runner.CombAllReduce on ``comb_in`` = (y, ids,
the shared expert's output); the returned token is an unwritten [M, 5120] that
the FFN hook recognises.

Scratch is persistent per runner, sized for M = 256 and initialised once
(partial's unwritten columns 0, xs pad 0, stage 2's hs pad 0x7F), so a call
launches no fills. Layers run one after another on a stream, so they share it.
Outputs are fresh per call.
"""

import struct

import torch

from vllm.models.deepseek_v41.amd.asm.kernels import ffn_route as fr
from vllm.models.deepseek_v41.amd.asm.kernels import (
    ffn_seam,
    moe_down1,
    moe_prep,
    moe_ug1,
    multi,
)
from vllm.models.deepseek_v41.amd.asm.kernels.kmod import Kernel

H, MAX_M, TOPK, I_REAL = 5120, 256, 6, 576
BF = torch.bfloat16


def f32bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def raw_from_layer(layer) -> dict:
    """A vLLM DeepseekV4DecoderLayer's FFN-half tensors under the capture's
    names (capture/capture_moe.py)."""
    ts = {
        n: t
        for n, t in [*layer.ffn.named_parameters(), *layer.ffn.named_buffers()]
        if t is not None and "routed_experts" not in n
    }
    for n in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
        ts[f"seam.{n}"] = getattr(layer, n)
    ts["seam.ffn_norm.weight"] = layer.ffn_norm.weight
    for n, v in (
        ("rms_norm_eps", layer.rms_norm_eps),
        ("hc_eps", layer.hc_eps),
        ("hc_post_alpha", layer.hc_post_alpha),
        ("hc_sinkhorn_iters", layer.hc_sinkhorn_iters),
        ("norm_eps", layer.ffn_norm.variance_epsilon),
    ):
        ts[f"seam.{n}"] = torch.tensor(float(v), dtype=torch.float64)
    e = layer.ffn.experts.routed_experts
    for n in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        ts[f"routed.{n}"] = getattr(e, n)
    return ts


def _find(ws: dict, key: str) -> torch.Tensor:
    hits = [n for n in ws if key in n]
    assert len(hits) == 1, (key, hits)
    return ws[hits[0]]


class K2Runner:
    def __init__(self, device=None):
        self.dev = torch.device(device or "cuda")
        mk = lambda k: Kernel(k["name"], k["src"], k["ka"])
        self.k_seam = mk(ffn_seam.source_v2(yq="ug", name="ffn_seam2_ug"))
        d = multi.source(
            "ffn_rt_sgu",
            [
                ("rt", fr.source_router(), fr.R_GRID),
                ("sgu", fr.source_shared(), fr.S_GRID),
            ],
        )
        self.k_rt_sgu = (mk(d), d["grid"], d["block"])
        d = fr.source_p3()
        self.k_p3 = (mk(d), d["block"])
        d = multi.source(
            "ffn_prep_sdn",
            [
                ("prep", moe_prep.source(), 1),
                ("sdn", fr.source_shared_down(), fr.D_GRID),
            ],
        )
        self.k_prep_sdn = (mk(d), d["grid"], d["block"])
        self.k_ug = mk(moe_ug1.source())
        self.k_dn = mk(moe_down1.source())
        self._scratch()

    def _scratch(self):
        d, P = self.dev, MAX_M * TOPK
        f = lambda *s, dt=torch.float32: torch.empty(*s, dtype=dt, device=d)
        self.s = dict(
            partial=torch.zeros(MAX_M, ffn_seam.SLICES, 32, device=d),
            y=f(MAX_M, H, dt=BF),
            xq=f(MAX_M, H, dt=torch.uint8),
            xs=torch.zeros(MAX_M, 192, dtype=torch.uint8, device=d),
            rpart=f(MAX_M, fr.RS, fr.E),
            gpart=f(MAX_M, fr.SS, fr.NS),
            logits=f(MAX_M, fr.E),
            ids=f(MAX_M, TOPK, dt=torch.int32),
            wts=f(MAX_M, TOPK),
            hq=f(MAX_M, I_REAL, dt=torch.uint8),
            hs_sh=f(MAX_M, I_REAL // 32, dt=torch.uint8),
            shared=f(MAX_M, H, dt=BF),
            groups=f(P, 20, dt=torch.int32),
            counts=torch.zeros(moe_prep.COUNTS_DSPLIT, dtype=torch.int32, device=d),
            h=torch.zeros(P, I_REAL, dtype=torch.uint8, device=d),
            hs=torch.full((P, 32), 0x7F, dtype=torch.uint8, device=d),
            y2=f(P, H, dt=BF),
            rstd=torch.ones(MAX_M, device=d),
        )

    def weights(self, raw: dict) -> dict:
        """Kernel-ready weights of one layer from capture-named tensors."""
        g = lambda n: raw[f"seam.{n}"]
        assert int(g("hc_sinkhorn_iters").item()) == ffn_seam.SINKHORN
        u8 = lambda t: t.view(torch.uint8)
        return dict(
            fn_hl=ffn_seam.fn_hl(g("hc_ffn_fn").float()),
            scale=g("hc_ffn_scale").float().contiguous(),
            base=g("hc_ffn_base").float().contiguous(),
            norm_w=g("ffn_norm.weight").to(BF).contiguous(),
            rms_eps=f32bits(g("rms_norm_eps").item()),
            hc_eps=f32bits(g("hc_eps").item()),
            post_alpha=f32bits(g("hc_post_alpha").item()),
            norm_eps=f32bits(g("norm_eps").item()),
            gate=_find(raw, "gate.weight").to(BF).contiguous(),
            bias=_find(raw, "e_score_correction_bias").float().contiguous(),
            sw=raw["shared_experts.gate_up_proj.weight"].contiguous(),
            ss=raw["shared_experts.gate_up_proj.weight_scale"].contiguous(),
            dw=raw["shared_experts.down_proj.weight"].contiguous(),
            ds=raw["shared_experts.down_proj.weight_scale"].contiguous(),
            w13=u8(raw["routed.w13_weight"]),
            s13=raw["routed.w13_weight_scale"],
            w2=u8(raw["routed.w2_weight"]),
            s2=raw["routed.w2_weight_scale"],
        )

    def __call__(self, w: dict, x, residual, post_mix, res_mix, attn_pre):
        """X [M, 5120] bf16 (the attention's reduced output), residual [M, 4,
        5120] bf16, post_mix [M, 4, 1], res_mix [M, 4, 4], attn_pre [M, 4] f32
        → token, R_new, post_next, comb_next, ffn_pre; ``self.comb_in`` = (y,
        ids, add) for ar_runner.CombAllReduce."""
        m, d = x.shape[0], x.device
        assert 1 <= m <= MAX_M, m
        P = m * TOPK
        s = {
            n: t[:P] if n in ("groups", "h", "hs", "y2") else t[:m]
            for n, t in self.s.items()
            if n != "counts"
        }
        counts = self.s["counts"]
        ids, wts = s["ids"].reshape(-1), s["wts"].reshape(-1)
        r_out = torch.empty_like(residual)
        post = torch.empty(m, 4, 1, device=d)
        comb = torch.empty(m, 4, 4, device=d)
        pre = torch.empty(m, 4, device=d)
        tg = ffn_seam.tg_for(m)
        self.k_seam(
            40 * tg,
            256,
            residual=residual,
            x=x,
            post=post_mix,
            comb=res_mix,
            pre=attn_pre,
            r_out=r_out,
            x1=s["y"],
            partial=s["partial"],
            m=m,
            fn_hl=w["fn_hl"],
            tg=tg,
            norm_w=w["norm_w"],
            xq=s["xq"],
            xs=s["xs"],
        )
        rt = dict(gate=w["gate"], y=s["y"], part=s["rpart"], m=m)
        sgu = dict(w=w["sw"], ws=w["ss"], xq=s["xq"], xs=s["xs"], part=s["gpart"], m=m)
        kern, grid, block = self.k_rt_sgu
        kern(grid, block, **_prefixed(rt=rt, sgu=sgu))
        kern, block = self.k_p3
        kern(
            m,
            block,
            rpart=s["rpart"],
            bias=w["bias"],
            logits=s["logits"],
            ids=s["ids"],
            wts=s["wts"],
            gpart=s["gpart"],
            hq=s["hq"],
            hs=s["hs_sh"],
            m=m,
            mx_partial=s["partial"],
            mx_scale=w["scale"],
            mx_base=w["base"],
            mx_post_out=post,
            mx_comb_out=comb,
            mx_pre_out=pre,
            mx_rstd=s["rstd"],
            mx_m=m,
            mx_rms_eps=w["rms_eps"],
            mx_hc_eps=w["hc_eps"],
            mx_post_alpha=w["post_alpha"],
            mx_norm_eps=w["norm_eps"],
            mx_dbg=s["rstd"],
        )
        prep = dict(
            x=s["y"],
            ids=ids,
            xq=s["xq"],
            xs=s["xs"],
            groups=s["groups"],
            counts=counts,
            m=m,
            x_bytes=m * H * 2,
            ids_bytes=P * 4,
            xq_bytes=m * H,
            xs_bytes=m * 192,
            groups_bytes=P * 80,
        )
        sdn = dict(
            w=w["dw"], ws=w["ds"], hq=s["hq"], hs=s["hs_sh"], out=s["shared"], m=m
        )
        kern, grid, block = self.k_prep_sdn
        kern(grid, block, **_prefixed(prep=prep, sdn=sdn))
        self.k_ug(
            256,
            256,
            w13=w["w13"],
            s13=w["s13"],
            xq=s["xq"],
            xs=s["xs"],
            groups=s["groups"],
            h=s["h"],
            hs=s["hs"],
            counts=counts,
            xq_bytes=m * H,
            xs_bytes=m * 192,
            h_bytes=P * I_REAL,
            hs_bytes=P * 32,
            rstd=s["rstd"],
        )
        self.k_dn(
            256,
            256,
            w2=w["w2"],
            s2=w["s2"],
            h=s["h"],
            hs=s["hs"],
            wts=wts,
            groups=s["groups"],
            y=s["y2"],
            counts=counts,
            h_bytes=P * I_REAL,
            hs_bytes=P * 32,
            wts_bytes=P * 4,
            y_bytes=P * H * 2,
        )
        self.comb_in = (s["y2"], ids, s["shared"])
        token = torch.empty(m, H, dtype=BF, device=d)
        return token, r_out, post, comb, pre


def _prefixed(**bodies: dict) -> dict:
    """A merged launch's kernargs (kernels/multi.py): body tag + "." + name."""
    return {f"{t}.{n}": v for t, a in bodies.items() for n, v in a.items()}
