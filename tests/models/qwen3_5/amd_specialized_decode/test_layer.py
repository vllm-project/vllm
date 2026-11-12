# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The wide decode's layer kernels at TP8 on 8 MI355X against the float64
references (``reference``) with the stock path's rounding points: three
launches on synthetic per-rank weights, a GDN layer that is the step's first
(the embedding's input norm, the next attention layer's qkv_proj), an
attention layer (o_proj and the MLP half) and a GDN layer closing with the
final norm, each launch's reference fed the kernels' previous outputs.

Tolerances are relative Frobenius errors. The MoE is checked on the kernels'
own MXFP4 input rows (``x4``): the references' inputs differ from the kernels'
in a few bf16 ulps, which flip MXFP4 codes; routes are compared as top-k sets
within a tie margin (``reference.route``), as such inputs flip near-tied
experts."""

import os
import time

import pytest
import torch

from vllm.platforms import current_platform

TP = 8
WIDTHS = [1, 2, 8, 16, 17, 32, 64, 128]
# GDN projection formats: "mxfp4t" the MXFP4 weights unshuffled (AITER's
# Triton GEMM layout)
VARIANTS = ["bf16", "mxfp4", "mxfp4t", "h32"]


def _on_8_gfx950() -> bool:
    if not current_platform.is_rocm() or torch.accelerator.device_count() < TP:
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(
    not _on_8_gfx950(), reason="the wide kernels run TP8 on 8 gfx950 (MI355X)"
)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _worker(rank, rows, variant, port, out_q):
    import torch.distributed as dist

    from vllm.models.qwen3_5.amd.specialized_decode.runtime.peer import PeerBuffer
    from vllm.models.qwen3_5.amd.specialized_decode.wide import layer as K
    from vllm.models.qwen3_5.amd.specialized_decode.wide import layout as L
    from vllm.models.qwen3_5.amd.specialized_decode.wide.abi import (
        FAILED,
        PEER_HEAD,
        health_words,
    )
    from vllm.models.qwen3_5.amd.specialized_decode.wide.common.grid import (
        FOLD_BASE,
        SLOT_WORDS,
        sync_words,
    )
    from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import exchange

    from . import reference as R

    S = rows
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP
    )
    fmt = {"bf16": "bf16", "mxfp4": "mxfp4", "mxfp4t": "mxfp4", "h32": "mxfp4_h32"}[
        variant
    ]
    shuffled = variant != "mxfp4t"
    q4 = K.q4_of(fmt, shuffled) or "none"
    rip = L.RI
    eps = 1e-6
    log = []

    def note(name, v, tol):
        log.append((name, v, tol))

    shared = torch.Generator(device=dev).manual_seed(1234)
    mine = torch.Generator(device=dev).manual_seed(1000 + rank)

    def rn(*shape, scale=1.0, g=mine, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, device=dev) * scale).to(dtype)

    def dense_or_fp4(n, k):
        """(kernel weight, kernel scales or 0, reference weight)."""
        if fmt == "bf16":
            w = rn(n, k, scale=0.02)
            return w, 0, w
        codes, sc = R.mxfp4(n, k, mine, lo=119, hi=123, device=dev)
        ref = R.dequant_mx(codes, sc)
        if shuffled:
            return R.shuffle_data(codes), R.shuffle_scale(sc), ref
        return codes.contiguous(), sc.T.contiguous(), ref

    def gdn_weights():
        qkvz, qkvz_s, qkvz_r = dense_or_fp4(L.QKVZ, L.HIDDEN)
        out, out_s, out_r = dense_or_fp4(L.HIDDEN, L.VALUE_DIM)
        w = dict(
            ba=rn(L.BA, L.HIDDEN, scale=0.02),
            conv=rn(L.CONV_DIM, 1, 4, scale=0.5),
            a_log=torch.randn(L.NV, generator=mine, device=dev) * 0.5,
            dt_bias=rn(L.NV, scale=0.5),
            norm=rn(L.GDN_DV, scale=0.2, g=shared) + 1,
        )
        return dict(w, qkvz=qkvz, qkvz_s=qkvz_s, out=out, out_s=out_s), dict(
            w, qkvz=qkvz_r, out=out_r
        )

    def moe_weights():
        e = L.E_ALL
        w13, w13s = R.mxfp4(e * 2 * rip, L.HIDDEN, mine, lo=119, hi=123, device=dev)
        w2, w2s = R.mxfp4(e * L.HIDDEN, rip, mine, lo=119, hi=123, device=dev)
        return dict(
            w_gate=rn(e, L.HIDDEN, scale=0.02, g=shared),
            w13=R.shuffle_data(w13).view(e, 2 * rip, L.HIDDEN // 2),
            w13s=R.shuffle_scale(w13s).view(e, 2 * rip, -1),
            w2=R.shuffle_data(w2).view(e, L.HIDDEN, rip // 2),
            w2s=R.shuffle_scale(w2s).view(e, L.HIDDEN, -1),
            r13=w13.view(e, 2 * rip, -1),
            r13s=w13s.view(e, 2 * rip, -1),
            r2=w2.view(e, L.HIDDEN, -1),
            r2s=w2s.view(e, L.HIDDEN, -1),
        )

    norms = [rn(L.HIDDEN, scale=0.1, g=shared) for _ in range(7)]
    g0k, g0r = gdn_weights()
    g2k, g2r = gdn_weights()
    moes = [moe_weights() for _ in range(3)]
    w_qkv = rn(L.QKV, L.HIDDEN, scale=0.02)
    w_o = rn(L.HIDDEN, L.ATTN_CORE, scale=0.02)
    emb = rn(S, L.HIDDEN, g=shared)
    attn = rn(S, L.ATTN_CORE)
    gate = rn(S, L.ATTN_CORE)
    perm = torch.randperm(S, generator=shared, device=dev).to(torch.int32) + 1
    if S > 2:
        perm[-1] = 0  # a graph pad row
    st_idx = perm.contiguous()
    convs = [rn(S + 2, 3, L.CONV_DIM, scale=0.5) for _ in range(2)]
    ssms = [
        rn(S + 2, L.NV, L.GDN_DV, L.GDN_DK, scale=0.1, dtype=torch.float32)
        for _ in range(2)
    ]

    peers = PeerBuffer(
        PEER_HEAD + exchange.mailbox_bytes(TP), dist.group.WORLD, rank, TP, dev
    )
    peers.bytes.zero_()
    table = (peers.addresses + PEER_HEAD).clone()
    alarm = int(peers.addresses[0].item()) + 4 * rank
    sync = torch.zeros(sync_words(), dtype=torch.int32, device=dev)
    health = torch.zeros(health_words(L.BLOCKS), dtype=torch.int32, device=dev)
    epoch = torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=dev)
    scratch = torch.zeros(K.scratch_bytes(S, rip), dtype=torch.uint8, device=dev)
    lay = K.scratch_layout(S, rip)
    base = sync.data_ptr()
    slots = (base, base + 4 * SLOT_WORDS)

    def sview(name, cols, dtype=torch.bfloat16):
        off = lay[name][0]
        el = torch.tensor([], dtype=dtype).element_size()
        return scratch[off : off + S * cols * el].view(dtype).view(S, cols)

    def args(i, seq, **kw):
        a = dict.fromkeys(K.ABI.names, 0)
        m = moes[i]
        a |= dict(
            scratch=scratch,
            peers=table,
            rank=rank,
            epoch=epoch,
            layer=i,
            folds=base + 4 * FOLD_BASE,
            health=health,
            alarm=alarm,
            trace=0,
            sync=slots[seq % 2],
            sync_next=slots[(seq + 1) % 2],
            w_gate=m["w_gate"],
            w13=m["w13"],
            w13s=m["w13s"],
            w2=m["w2"],
            w2s=m["w2s"],
        )
        return a | kw

    def gdn_args(gk, conv, ssm):
        return dict(
            w_qkvz=gk["qkvz"],
            w_qkvz_s=gk["qkvz_s"],
            w_ba=gk["ba"],
            w_conv=gk["conv"],
            a_log=gk["a_log"],
            dt_bias=gk["dt_bias"],
            gdn_norm=gk["norm"],
            w_out=gk["out"],
            w_out_s=gk["out_s"],
            conv_state=conv,
            conv_stride=conv.stride(0),
            ssm=ssm,
            ssm_stride=ssm.stride(0),
            st_idx=st_idx,
        )

    def launch(key, a):
        K.run(key, a)
        torch.cuda.synchronize()
        if int(health[FAILED]) != 0:
            raise RuntimeError(f"rank {rank}: an in-kernel wait gave up")

    def allsum(part):
        """Every rank's bf16 partial, summed in rank order -> bf16."""
        parts = [torch.empty_like(part.cpu()) for _ in range(TP)]
        dist.all_gather(parts, part.cpu().contiguous())
        tot = parts[0].double()
        for p in parts[1:]:
            tot = tot + p.double()
        return R.bf(tot).to(dev)

    # the first layer's reference rows come from the embedding, not the
    # kernel's x: a rare row a bf16 ulp apart under the Hadamard rotation moves
    # its 32-groups' MXFP4 codes (the rotation itself is checked bit for bit in
    # test_split.py), so an h32 first layer allows that one row
    slack = {"L0": 1 if fmt == "mxfp4_h32" else 0}

    def mlp_ref(res, xn, i, n_next, tag, x4_live):
        m = moes[i]
        route = scratch[lay["route"][0] :].view(torch.int32)
        kr = route[: S * L.SLOTS].view(S, L.SLOTS)
        kxd = None
        if x4_live:
            kxd = R.dequant_mx(
                sview("x4", L.HIDDEN // 2, torch.uint8),
                sview("x4s", L.HIDDEN // 32, torch.int32),
            )
            note(
                f"{tag} moe input (MXFP4)",
                rel(kxd, R.quant_act(xn)),
                2e-2 * (1 + slack.get(tag, 0)),
            )
        y, ids, ws = R.moe(
            xn,
            m["w_gate"],
            m["r13"],
            m["r13s"],
            m["r2"],
            m["r2s"],
            L.EXPERTS,
            L.TOPK,
            L.RI,
            rip,
            prefer=kr[:, : L.TOPK],
            xd=kxd,
        )
        if "y" in lay:  # else (expert-major) the residual after the exchange
            note(
                f"{tag} moe y",
                rel(sview("y", L.HIDDEN, torch.float32), y),
                1e-2 * (1 + slack.get(tag, 0)),
            )
        same = sum(set(kr[r].tolist()) == set(ids[r].tolist()) for r in range(S))
        note(f"{tag} routes", 1 - same / S, slack.get(tag, 0) / S)
        kws = route.view(torch.float32)[S * L.SLOTS : 2 * S * L.SLOTS]
        note(f"{tag} route weights", rel(kws.view(S, L.SLOTS), ws), 1e-2)
        res2 = R.bf(res + allsum(R.bf(y)))
        return res2, R.gemma_norm(res2, n_next, eps)

    t0 = time.time()
    # GDN layer 0, the step's first; the next layer is attention
    k0 = K.LayerBuild(
        kind="gdn",
        tokens=S,
        tp=TP,
        first=True,
        qkvz=fmt,
        qkvz_shuffled=shuffled,
        out=fmt,
        out_shuffled=shuffled,
        first_q4=q4,
        qkv_tail=True,
    )
    conv0, ssm0 = convs[0].clone(), ssms[0].clone()
    launch(
        k0,
        args(
            0,
            0,
            emb=emb,
            n_in=norms[0],
            n_post=norms[1],
            n_next=norms[2],
            w_qkv=w_qkv,
            **gdn_args(g0k, conv0, ssm0),
        ),
    )
    # in f32 as the kernel and stock: under a Hadamard rotation one ulp of x
    # moves a whole 32-group's MXFP4 codes
    z0 = emb.float()
    x0 = R.bf(
        z0
        * torch.rsqrt((z0 * z0).mean(-1, keepdim=True) + eps)
        * (1 + norms[0].float())
    )
    _, _, rconv, rssm, _, outp = R.gdn(
        x0, g0r, st_idx, convs[0], ssms[0], eps, (fmt, fmt), shuffled
    )
    note("L0 conv state", rel(conv0, rconv), 1e-2)
    note("L0 ssm state", rel(ssm0, rssm), 1e-2)
    res1 = R.bf(emb.double() + allsum(outp))
    res2, x2 = mlp_ref(res1, R.gemma_norm(res1, norms[1], eps), 0, norms[2], "L0", True)
    kres, kx = sview("res", L.HIDDEN).clone(), sview("x", L.HIDDEN).clone()
    note("L0 residual", rel(kres, res2), 1e-2 * (1 + slack["L0"]))
    note("L0 x", rel(kx, x2), 1e-2 * (1 + slack["L0"]))
    tail = R.bf(kx.double() @ w_qkv.double().T)
    note("L0 next qkv_proj", rel(sview("qkv_out", L.QKV), tail), 1e-2)

    # attention layer 1: o_proj and the MLP half; the next layer is GDN
    k1 = K.LayerBuild(kind="attn", tokens=S, tp=TP, next_q4=q4)
    launch(
        k1,
        args(1, 1, n_post=norms[3], n_next=norms[4], w_out=w_o, attn=attn, gate=gate),
    )
    core1 = R.bf(attn.double() * R.bf(torch.sigmoid(gate.double())))
    res3 = R.bf(kres.double() + allsum(R.bf(core1 @ w_o.double().T)))
    res4, x4 = mlp_ref(
        res3, R.gemma_norm(res3, norms[3], eps), 1, norms[4], "L1", q4 == "none"
    )
    kres, kx = sview("res", L.HIDDEN).clone(), sview("x", L.HIDDEN).clone()
    note("L1 residual", rel(kres, res4), 1e-2)
    note("L1 x", rel(kx, x4), 1e-2)

    # GDN layer 2, closing with the final norm
    k2 = K.LayerBuild(
        kind="gdn",
        tokens=S,
        tp=TP,
        qkvz=fmt,
        qkvz_shuffled=shuffled,
        out=fmt,
        out_shuffled=shuffled,
    )
    conv2, ssm2 = convs[1].clone(), ssms[1].clone()
    launch(
        k2, args(2, 2, n_post=norms[5], n_next=norms[6], **gdn_args(g2k, conv2, ssm2))
    )
    _, _, rconv, rssm, _, outp = R.gdn(
        kx.double(), g2r, st_idx, convs[1], ssms[1], eps, (fmt, fmt), shuffled
    )
    note("L2 conv state", rel(conv2, rconv), 1e-2)
    note("L2 ssm state", rel(ssm2, rssm), 1e-2)
    res5 = R.bf(kres.double() + allsum(outp))
    res6, x6 = mlp_ref(res5, R.gemma_norm(res5, norms[5], eps), 2, norms[6], "L2", True)
    note("L2 residual", rel(sview("res", L.HIDDEN), res6), 1e-2)
    note("L2 x (final norm)", rel(sview("x", L.HIDDEN), x6), 1e-2)
    out_q.put((rank, log, time.time() - t0))
    dist.barrier()
    peers.close()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("rows", WIDTHS)
def test_layers_tp8(rows, variant):
    import torch.multiprocessing as mp

    port = 29500 + (os.getpid() + 37 * rows + VARIANTS.index(variant)) % 2000
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, rows, variant, port, q)) for r in range(TP)
    ]
    for p in procs:
        p.start()
    logs = {}
    try:
        for _ in range(TP):
            rank, log, _ = q.get(timeout=1800)
            logs[rank] = log
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.kill()
    assert sorted(logs) == list(range(TP)), f"ranks reported: {sorted(logs)}"
    bad = [
        f"rank {r} {name}: {v:.2e} > {tol:.0e}"
        for r, log in logs.items()
        for name, v, tol in log
        if not v <= tol
    ]
    assert not bad, "\n".join(bad)
