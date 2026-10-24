# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash whole-layer KDA MonoKernel on CDNA4, one process a TP rank.

Every stage is checked against vLLM's own ops on the kernel's own inputs to that
stage: the attention-side mHC, the merged projection, the short conv, the gated
delta rule (and both state updates), the gated RMSNorm, then the FFN seam on the
TP sum of the o_proj partials and the MoE."""

import pytest
import torch

from .test_flash_ffn_kernel import (
    REPLAYS,
    _Layer,
    _moe_reference,
    _on_cdna4,
    _rel,
    _vllm_seam,
    _worker,
)

ROWS = (2, 4, 8)
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


def _kda_reference(k: _Kda, proj, gates, conv, rec, idx, dim_first):
    """The decode ops of vLLM on the kernel's own projection and gate rows; update
    ``conv`` and ``rec`` in place and return the gated-RMSNorm output [S, P]."""
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
    """The first ``n`` values of a mailbox: f32 pairs, or packed bf16 pairs."""
    words = rt.scratch[rt.sc[name] :].view(torch.int32)
    if packed:
        return words[:n:2].contiguous().view(torch.bfloat16)
    return words[: 2 * n : 2].contiguous().view(torch.float32)


def _check(rank: int, tp: int, device, dim_first: bool) -> None:
    import torch.distributed as dist

    from vllm.distributed import get_tp_group
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        HC_MULT,
        HIDDEN,
        MOE_SLOTS,
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
    n_in = FlashShard(tp).kda_in_rows
    group = get_tp_group().cpu_group
    for S in ROWS:
        rt = FlashFfnRuntime(
            S,
            rank,
            tp,
            group,
            2048 // tp,
            1,
            20_000_000,
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
        idx = torch.arange(1, S + 1, dtype=torch.int32, device=device)
        idx[-1] = -1
        live = S - 1
        state = FlashKdaState(kda.conv0.clone(), kda.rec0.clone())
        dist.barrier(group)
        outs = [t.clone() for t in op.launch(x, res, post, comb, state, idx)]
        torch.accelerator.synchronize()
        assert not rt.words.any(), f"S={S}: expired waits {rt.words.tolist()}"
        y, r2, p2, c2 = outs

        gathered = [torch.empty_like(y).cpu() for _ in range(tp)]
        dist.all_gather(gathered, y.cpu(), group=group)
        assert all(torch.equal(gathered[0], o) for o in gathered), f"S={S}: ranks"

        # attention-side mHC
        ra, pa, ca, a_ref = _seam(kda, x, res, post, comb)
        a_in = _box(rt, "a_in", S * HIDDEN, packed=True).view(S, HIDDEN)
        assert _rel(a_in, a_ref) < 1e-2, f"S={S}: attention input {_rel(a_in, a_ref)}"

        # merged projection and the f_b / g_b gates on the kernel's own inputs
        proj = _box(rt, "proj", S * n_in).view(S, n_in)
        want = (a_in.float() @ kda.w_in.float().T).bfloat16().float()
        assert _rel(proj, want) < 1e-2, f"S={S}: projection {_rel(proj, want)}"
        gates = _box(rt, "gates", S * 2 * kda.P).view(S, 2 * kda.P)
        fa = proj[:, 3 * kda.P + kda.H :].bfloat16().float()
        want = torch.cat(
            [fa[:, :128] @ kda.w_fb.float().T, fa[:, 128:] @ kda.w_gb.float().T], 1
        )
        assert _rel(gates, want) < 1e-2, f"S={S}: gates {_rel(gates, want)}"

        # short conv + gated delta rule + gated RMSNorm, and both state updates
        conv, rec = kda.conv0.clone(), kda.rec0.clone()
        o_ref = _kda_reference(
            kda, proj.bfloat16(), gates.bfloat16(), conv, rec, idx, dim_first
        )
        o_in = _box(rt, "o_in", S * kda.P, packed=True).view(S, kda.P)
        assert _rel(o_in[:live], o_ref[:live]) < 2e-2, (
            f"S={S}: KDA output {_rel(o_in[:live], o_ref[:live])}"
        )
        assert _rel(state.conv_state, conv) < 1e-2, f"S={S}: conv state"
        assert _rel(state.rec_state, rec) < 1e-3, (
            f"S={S}: recurrent state {_rel(state.rec_state, rec)}"
        )
        assert torch.equal(state.rec_state[S:], kda.rec0[S:]), "untouched slots"

        # FFN seam on the TP sum of every rank's o_proj partial
        o_all = [torch.empty_like(o_in) for _ in range(tp)]
        dist.all_gather(o_all, o_in.contiguous())
        x_attn = torch.zeros(S, HIDDEN, device=device)
        for r in range(tp):
            w_o = kda.w_o_full[:, r * kda.P : (r + 1) * kda.P].float()
            x_attn += (o_all[r].float() @ w_o.T).bfloat16().float()
        v_r, v_p, v_c, v_x = _vllm_seam(lay, x_attn.bfloat16(), ra, pa, ca)
        assert _rel(r2[:live], v_r[:live]) < 1e-3, f"S={S}: R'' {_rel(r2, v_r)}"
        assert _rel(p2[:live], v_p[:live]) < 1e-4 and _rel(c2[:live], v_c[:live]) < 1e-4

        a = rt.scratch[rt.sc["a"] :].view(torch.int32)[: S * HIDDEN : 2]
        k_in = a.contiguous().view(torch.bfloat16).view(S, HIDDEN)
        ref, order = _moe_reference(k_in, lay.w_r, lay.bias, lay.w13, lay.w2, 2048)
        sel = rt.scratch[rt.sc["sel"] :].view(torch.int32)[: 2 * S * MOE_SLOTS : 2]
        routed = sel.view(S, MOE_SLOTS)[:, 1:].sort(-1).values
        assert torch.equal(routed[:live], order[:live].int().sort(-1).values)
        assert _rel(y[:live], ref[:live]) < 5e-3, f"S={S}: MoE {_rel(y, ref)}"

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


@pytest.mark.skipif(not _on_cdna4(), reason="the Flash MonoKernel targets CDNA4")
@pytest.mark.parametrize(
    "tp, dim_first", [(2, False), (4, False), (4, True), (8, False)]
)
def test_kda_layer_matches_vllm_ops(
    monkeypatch: pytest.MonkeyPatch, tp: int, dim_first: bool
):
    from tests.utils import multi_process_parallel

    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    pytest.importorskip("flydsl")
    multi_process_parallel(monkeypatch, tp, 1, _worker(_check, dim_first))
