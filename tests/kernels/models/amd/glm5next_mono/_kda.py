# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash whole-layer KDA MonoKernel on CDNA4, one process a TP rank.

Every stage is measured against the fp64 reference of `reference.py` on the
kernel's own input to it, and so are vLLM's ops on the same input: the
attention-side mHC, the merged projection, the f_b / g_b gates, the short conv,
the gated delta rule (and both state updates), the gated RMSNorm, then the FFN
seam on the TP sum of the o_proj partials and the MoE."""

import torch

from ._ffn import (
    POLL_TIMEOUT_US,
    REPLAYS,
    _Layer,
    _vllm_seam,
    add_ffn_half,
)
from .reference import F64, Bound, Numerics, kda_decode, mhc_post, mhc_pre

SLOTS = 12


class _Kda:
    """One KDA layer's random attention weights (every rank builds the same) and
    this rank's shard, in vLLM's post-load layouts."""

    def __init__(self, rank: int, tp: int, device) -> None:
        from vllm.models.glm5next.amd.specialized_decode.mono.config import (
            HC_MULT,
            HIDDEN,
            KDA_DIM,
            KDA_HEADS,
            KDA_PROJ,
        )

        g = torch.Generator(device=device).manual_seed(1)
        D, PF, HF = KDA_DIM, KDA_PROJ, KDA_HEADS
        P, H = PF // tp, HF // tp
        self.P, self.H, self.tp, self.rank = P, H, tp, rank

        def rnd(*shape, scale=1.0):
            return torch.randn(*shape, generator=g, device=device) * scale

        nmix = HC_MULT * (2 + HC_MULT)
        self.hc_fn = rnd(nmix, HC_MULT * HIDDEN, scale=0.01)
        self.hc_scale = torch.rand(3, generator=g, device=device) + 0.5
        self.hc_base = rnd(nmix, scale=0.1)
        self.g_in = (1 + 0.1 * rnd(HIDDEN)).bfloat16()
        wq, wk, wv = (rnd(PF, HIDDEN, scale=HIDDEN**-0.5) for _ in range(3))
        wb = rnd(HF, HIDDEN, scale=HIDDEN**-0.5)
        wfa, wga = (rnd(D, HIDDEN, scale=HIDDEN**-0.5) for _ in range(2))
        self.w_fb_full = rnd(PF, D, scale=D**-0.5).bfloat16()
        self.w_gb_full = rnd(PF, D, scale=D**-0.5).bfloat16()
        self.w_o_full = rnd(HIDDEN, PF, scale=PF**-0.5).bfloat16()
        conv = rnd(3, PF, 4, scale=0.5)
        self.a_log_full = torch.log(torch.rand(HF, generator=g, device=device) * 8 + 1)
        self.dt_full = rnd(PF, scale=0.5)
        self.w_on = (1 + 0.1 * rnd(D)).float()

        def sh(t, n):
            return t[rank * n : (rank + 1) * n]

        self.w_in = torch.cat(
            [sh(wq, P), sh(wk, P), sh(wv, P), sh(wb, H), wfa, wga]
        ).bfloat16()
        self.w_fb = sh(self.w_fb_full, P).contiguous()
        self.w_gb = sh(self.w_gb_full, P).contiguous()
        self.w_o = self.w_o_full[:, rank * P : (rank + 1) * P].contiguous()
        self.conv_w = torch.cat([sh(conv[i], P) for i in range(3)]).contiguous()
        self.a_log = sh(self.a_log_full, H).contiguous()
        self.dt_bias = sh(self.dt_full, P).contiguous()
        self.conv0 = rnd(SLOTS, 3 * P, 3).bfloat16()
        self.rec0 = rnd(SLOTS, H, D, D, scale=0.05)

    def weights(self):
        from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import (
            FlashKdaWeights,
        )

        return FlashKdaWeights(
            hc_fn=self.hc_fn,
            hc_scale=self.hc_scale,
            hc_base=self.hc_base,
            g_norm=self.g_in,
            w_in=self.w_in,
            w_fb=self.w_fb,
            w_gb=self.w_gb,
            conv_w=self.conv_w,
            a_log=self.a_log,
            dt_bias=self.dt_bias,
            w_on=self.w_on,
            w_o=self.w_o,
        )


def _seam(k: _Kda, x, res, post, comb):
    from types import SimpleNamespace

    lay = SimpleNamespace(fn=k.hc_fn, scale=k.hc_scale, base=k.hc_base, gamma=k.g_in)
    return _vllm_seam(lay, x, res, post, comb)


def _kda_vllm(k: _Kda, proj, gates, conv, rec, idx, dim_first):
    """The decode ops of vLLM (causal_conv1d_update, fused_recurrent_kda,
    FusedRMSNormGated) on the given projection and gate rows; update `conv` and
    `rec` in place and return the gated-RMSNorm output [S, P]."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.glm5next.amd.ops.third_party.kda import fused_recurrent_kda
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        KDA_DIM,
        KDA_LOWER_BOUND,
    )
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    S, P, H, D = proj.shape[0], k.P, k.H, KDA_DIM
    qkv = causal_conv1d_update(
        proj[:, : 3 * P].contiguous(),
        conv if dim_first else conv.transpose(-1, -2),
        k.conv_w,
        None,
        "silu",
        conv_state_indices=idx,
    )
    q, kk, v = (t.reshape(1, S, H, D) for t in qkv.split(P, dim=-1))
    beta = proj[:, 3 * P : 3 * P + H].reshape(1, S, H).contiguous()
    g1 = gates[:, :P].reshape(1, S, H, D).contiguous()
    cu = torch.arange(S + 1, dtype=torch.int32, device=proj.device)
    o, _ = fused_recurrent_kda(
        q=q.contiguous(),
        k=kk.contiguous(),
        v=v.contiguous(),
        g=g1,
        beta=beta,
        initial_state=rec,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        ssm_state_indices=idx,
        sigmoid_beta=True,
        a_log=k.a_log.view(1, 1, H, 1),
        g_bias=k.dt_bias,
        compute_gate=True,
        lower_bound=KDA_LOWER_BOUND,
    )
    with set_current_vllm_config(VllmConfig()):
        norm = FusedRMSNormGated(D, activation="sigmoid").to(proj.device)
        norm.weight.data.copy_(k.w_on)
        out = norm(o, gates[:, P:].reshape(S, H, D).contiguous())
    return out.reshape(S, P)


def _box(rt, name, n, packed=False):
    """The first `n` values of a mailbox: f32 pairs, or packed bf16 pairs."""
    words = rt.scratch[rt.sc[name] :].view(torch.int32)
    if packed:
        return words[:n:2].contiguous().view(torch.bfloat16)
    return words[: 2 * n : 2].contiguous().view(torch.float32)


def _check(
    rank: int,
    tp: int,
    device,
    dim_first: bool,
    saturate: bool,
    widths: tuple[int, ...],
    bounds: dict[str, Bound],
) -> None:
    """`saturate` drives half the gate channels far below the sigmoid's range
    (amp * g < -88.7), where exp(-amp * g) is inf and the gate divides by it."""
    import torch.distributed as dist

    from vllm.distributed import get_tp_group
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        HC_MULT,
        HIDDEN,
        FlashShard,
    )
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import (
        FlashFfnOp,
        FlashFfnRuntime,
        FlashKdaState,
    )

    lay, kda = _Layer(rank, tp, device), _Kda(rank, tp, device)
    if not dim_first:
        kda.conv0 = kda.conv0.transpose(1, 2).contiguous()
    if saturate:
        kda.dt_bias[::2] = -1e3
    n_in = FlashShard(tp).kda_in_rows
    P, H = kda.P, kda.H
    group = get_tp_group().cpu_group
    layout = "dim-first" if dim_first else "width-first"
    title = f"KDA layer, TP{tp}, conv state {layout}" + (", saturated gates" * saturate)
    num = Numerics(title, bounds, rank)
    for S in widths:
        rt = FlashFfnRuntime(
            S,
            rank,
            tp,
            group,
            2048 // tp,
            1,
            POLL_TIMEOUT_US,
            False,
            kda=True,
            conv_dim_first=dim_first,
        )
        op = FlashFfnOp(rt, lay.weights(), 0, kda.weights())
        g = lay.g
        x = torch.randn(S, HIDDEN, generator=g, device=device).bfloat16()
        res = torch.randn(S, HC_MULT, HIDDEN, generator=g, device=device).bfloat16()
        post = torch.rand(S, HC_MULT, 1, generator=g, device=device) * 2
        comb = torch.softmax(
            torch.randn(S, HC_MULT, HC_MULT, generator=g, device=device), -1
        )
        # the last row is padding (state index -1); the rest decode one token each
        idx = torch.arange(1, S + 1, dtype=torch.int32, device=device)
        idx[-1] = -1
        live = S - 1
        slots = idx[:live].long()
        state = FlashKdaState(kda.conv0.clone(), kda.rec0.clone())
        dist.barrier(group)
        outs = [t.clone() for t in op.launch(x, res, post, comb, state, idx)]
        torch.accelerator.synchronize()
        assert not rt.words.any(), f"S={S}: expired waits {rt.words.tolist()}"
        assert state.rec_state.isfinite().all(), f"S={S}: non-finite state"
        y, r2, p2, c2 = outs

        gathered = [torch.empty_like(y).cpu() for _ in range(tp)]
        dist.all_gather(gathered, y.cpu(), group=group)
        assert all(torch.equal(gathered[0], o) for o in gathered), f"S={S}: ranks"

        # attention-side mHC (post of the previous layer, pre of this one)
        ra, pa, ca, a_vllm = _seam(kda, x, res, post, comb)
        ra64 = mhc_post(x, res, post, comb)
        pa64, ca64, a64 = mhc_pre(ra64, kda.hc_fn, kda.hc_scale, kda.hc_base, kda.g_in)
        a_in = _box(rt, "a_in", S * HIDDEN, packed=True).view(S, HIDDEN)
        num.add("attention input", S, a_in[:live], a64[:live], a_vllm[:live])

        # merged projection and the f_b / g_b gates on the kernel's own inputs;
        # vLLM's GEMMs are bf16 torch matmuls on the same inputs
        proj = _box(rt, "proj", S * n_in).view(S, n_in)
        num.add(
            "projection",
            S,
            proj[:live],
            (a_in.to(F64) @ kda.w_in.to(F64).T)[:live],
            (a_in @ kda.w_in.T)[:live],
        )
        gates = _box(rt, "gates", S * 2 * P).view(S, 2 * P)
        fa = proj[:, 3 * P + H :].bfloat16()
        gates64 = torch.cat(
            [
                fa[:, :128].to(F64) @ kda.w_fb.to(F64).T,
                fa[:, 128:].to(F64) @ kda.w_gb.to(F64).T,
            ],
            1,
        )
        gates_vllm = torch.cat([fa[:, :128] @ kda.w_fb.T, fa[:, 128:] @ kda.w_gb.T], 1)
        num.add("gates", S, gates[:live], gates64[:live], gates_vllm[:live])

        # short conv + gated delta rule + gated RMSNorm and both state updates:
        # fp64 on the kernel's own fp32 rows, vLLM on them rounded to bf16 as its
        # projection GEMM gives them
        conv64, rec64 = kda.conv0.to(F64), kda.rec0.to(F64)
        kw = (kda.conv_w, kda.a_log, kda.dt_bias, kda.w_on)
        o64 = kda_decode(proj, gates, conv64, rec64, idx, kw, dim_first)
        conv_v, rec_v = kda.conv0.clone(), kda.rec0.clone()
        o_v = _kda_vllm(
            kda, proj.bfloat16(), gates.bfloat16(), conv_v, rec_v, idx, dim_first
        )
        o_in = _box(rt, "o_in", S * P, packed=True).view(S, P)
        num.add("KDA output", S, o_in[:live], o64[:live], o_v[:live])
        num.add(
            "conv state",
            S,
            state.conv_state[slots],
            conv64[slots],
            conv_v[slots],
        )
        num.add(
            "recurrent state", S, state.rec_state[slots], rec64[slots], rec_v[slots]
        )
        assert torch.equal(state.rec_state[S:], kda.rec0[S:]), "untouched slots"
        assert torch.equal(state.conv_state[S:], kda.conv0[S:]), "untouched slots"

        # R'' on the TP sum of every rank's o_proj partial: fp64 over the fp64
        # attention-side streams, vLLM's sum of bf16 partials (its all-reduce) over
        # its own; then the FFN half on the kernel's own R''
        o_all = [torch.empty_like(o_in) for _ in range(tp)]
        dist.all_gather(o_all, o_in.contiguous())
        x_attn = torch.zeros(S, HIDDEN, device=device)
        x_attn64 = torch.zeros(S, HIDDEN, device=device, dtype=F64)
        for r in range(tp):
            w_o = kda.w_o_full[:, r * P : (r + 1) * P]
            x_attn += (o_all[r].float() @ w_o.float().T).bfloat16().float()
            x_attn64 += o_all[r].to(F64) @ w_o.to(F64).T
        v_r = _vllm_seam(lay, x_attn.bfloat16(), ra, pa, ca)[0]
        r64 = mhc_post(x_attn64, ra64, pa64, ca64)
        num.add("residual", S, r2[:live], r64[:live], v_r[:live])
        add_ffn_half(num, S, lay, r2, p2, c2, y, rt, live)

        # a non-finite padding row stays in its own row
        bad = x.clone()
        bad[-1] = float("nan")
        state_bad = FlashKdaState(kda.conv0.clone(), kda.rec0.clone())
        dist.barrier(group)
        rt.advance_step()
        z = [t.clone() for t in op.launch(bad, res, post, comb, state_bad, idx)]
        torch.accelerator.synchronize()
        for got, want in zip(z, outs):
            assert torch.equal(got[:-1], want[:-1]), f"S={S}: a NaN row leaked"
        assert torch.equal(state_bad.rec_state, state.rec_state)

        # graph replays from the same state, every one bit for bit the first
        dist.barrier(group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state.conv_state.copy_(kda.conv0)
            state.rec_state.copy_(kda.rec0)
            rt.advance_step()
            res_g = op.launch(x, res, post, comb, state, idx)
        for i in range(REPLAYS):
            graph.replay()
            if i % 100 == 99 or i == REPLAYS - 1:
                torch.accelerator.synchronize()
                assert all(torch.equal(o, w) for o, w in zip(res_g, outs)), (
                    f"S={S}: replay {i} differs"
                )
        assert not rt.words.any()
        dist.barrier(group)
        rt.close()
    num.check()
