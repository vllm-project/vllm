# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The split widths (`layout.SPLIT_MAX` rows and fewer) at TP8 on 8
MI355X: a layer is the wide GDN (or attention) launch up to its core (or
qkv_proj), then `mono.blocks.layer_block`. GDN layer 0 (the step's first)
to its core, then layer_block; attention layer 1's input combine and
qkv_proj, then layer_block on synthetic attention rows; GDN layer 2 on layer
1's output and residual. Each reference is fed the kernels' previous outputs.

layer_block's MoE is judged against the exact one: its error to the
exact MoE (float64 on the dequantized weights, activations unquantized) at
most `MOE_ERR_RATIO` times the stock `fused_moe`'s on the same input,
its cosine to the exact one above `1 - 5e-2`.

`test_split_gdn_quant` (one gfx950 GPU) checks the split GDN launch's MXFP4
input rows bit for bit against AITER's quantizers: `per_1x32_f4_quant_hip`
for the ASM GEMM layout, `dynamic_mxfp4_quant` for the Triton one."""

import os

import pytest
import torch

from vllm.platforms import current_platform

TP = 8
WIDTHS = [1, 2, 4, 8, 16]
VARIANTS = ["bf16", "mxfp4", "mxfp4t", "h32"]
MOE_ERR_RATIO = 1.1


def _gfx950(n: int) -> bool:
    if not current_platform.is_rocm() or torch.accelerator.device_count() < n:
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _worker(rank, S, variant, port, out_q):
    import torch.distributed as dist

    from vllm.models.qwen3_5.amd.specialized_decode import layout as L
    from vllm.models.qwen3_5.amd.specialized_decode.abi import FAILED, health_words
    from vllm.models.qwen3_5.amd.specialized_decode.abi import PEER_HEAD as MONO_HEAD
    from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks
    from vllm.models.qwen3_5.amd.specialized_decode.mono.common import HEALTH_WORDS
    from vllm.models.qwen3_5.amd.specialized_decode.runtime.peer import PeerBuffer
    from vllm.models.qwen3_5.amd.specialized_decode.wide import layer as K
    from vllm.models.qwen3_5.amd.specialized_decode.wide.grid import (
        FOLD_BASE,
        SLOT_WORDS,
        launch_slots,
        sync_words,
    )

    from . import reference as R

    torch.accelerator.set_device_index(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP
    )
    group = dist.group.WORLD
    fmt = {"bf16": "bf16", "mxfp4": "mxfp4", "mxfp4t": "mxfp4", "h32": "mxfp4_h32"}[
        variant
    ]
    shuffled = variant != "mxfp4t"
    q4 = K.q4_of(fmt, shuffled) or "none"
    rip = L.RI
    log = []

    def note(name, v, tol):
        log.append((name, v, tol))

    shared = torch.Generator(device=dev).manual_seed(1234)
    mine = torch.Generator(device=dev).manual_seed(1000 + rank)

    def rn(*shape, scale=1.0, g=mine, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, device=dev) * scale).to(dtype)

    def dense_or_fp4(rows, k):
        if fmt == "bf16":
            w = rn(rows, k, scale=0.02)
            return w, 0, w
        codes, sc = R.mxfp4(rows, k, mine, lo=119, hi=123, device=dev)
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
        k = dict(w, qkvz=qkvz, qkvz_s=qkvz_s, out=out, out_s=out_s)
        r = dict(w, qkvz=qkvz_r, out=out_r)
        return k, r

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

    norms = [rn(L.HIDDEN, scale=0.1, g=shared) for _ in range(6)]
    g0k, g0r = gdn_weights()
    g2k, g2r = gdn_weights()
    moes = [moe_weights() for _ in range(3)]
    w_o = rn(L.HIDDEN, L.ATTN_CORE, scale=0.02)
    w_qkv = rn(L.QKV, L.HIDDEN, scale=0.02)
    emb = rn(S, L.HIDDEN, g=shared)
    attn = rn(S, L.ATTN_CORE)
    gate = rn(S, L.ATTN_CORE)
    slots = S + 2
    perm = torch.randperm(S, generator=shared, device=dev).to(torch.int32) + 1
    if S > 2:
        perm[-1] = 0  # a graph pad row
    st_idx = perm.contiguous()
    live = st_idx > 0
    convs = [rn(slots, 3, L.CONV_DIM, scale=0.5) for _ in range(2)]
    ssms = [
        rn(slots, L.NV, L.GDN_DV, L.GDN_DK, scale=0.1, dtype=torch.float32)
        for _ in range(2)
    ]

    # ---- the wide GDN launches' state
    sync = torch.zeros(sync_words(), dtype=torch.int32, device=dev)
    health = torch.zeros(health_words(L.BLOCKS), dtype=torch.int32, device=dev)
    epoch = torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=dev)
    scratch = torch.zeros(K.scratch_bytes(S, rip), dtype=torch.uint8, device=dev)
    lay = K.scratch_layout(S, rip)
    base = sync.data_ptr()
    slot_ptr = [base + 4 * SLOT_WORDS * i for i in range(3)]
    sched = launch_slots(3)

    def sview(name, cols, dtype=torch.bfloat16):
        off, n = lay[name]
        el = torch.tensor([], dtype=dtype).element_size()
        return scratch[off : off + S * cols * el].view(dtype).view(S, cols)

    # ---- layer_block's
    mpeers = PeerBuffer(
        MONO_HEAD + blocks.peer_bytes(L.SPLIT_MAX), group, rank, TP, dev
    )
    mpeers.bytes.zero_()
    mtable = (mpeers.addresses + MONO_HEAD).clone()
    malarm = int(mpeers.addresses[0].item()) + 4 * rank
    mhealth = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=dev)
    mlay = blocks.layer_scratch_layout(blocks.LayerBuild(tokens=S))
    mscratch = torch.zeros(
        (blocks.scratch_bytes(mlay) + 255) // 256 * 256, dtype=torch.uint8, device=dev
    )
    h = torch.empty(S, L.HIDDEN, dtype=torch.bfloat16, device=dev)
    res = torch.empty(S, L.HIDDEN, dtype=torch.bfloat16, device=dev)
    core = torch.empty(S, L.VALUE_DIM, dtype=torch.bfloat16, device=dev)

    def check_health():
        torch.accelerator.synchronize()
        if int(health[FAILED]) or int(mhealth[0]):
            raise RuntimeError(
                f"rank {rank}: in-kernel wait gave up: wide {health[:16].tolist()} "
                f"mono {mhealth[:16].tolist()}"
            )

    def gdn_launch(seq, i, gk, conv, ssm, n_in, first, h_in, res_in):
        key = K.LayerBuild(
            kind="gdn",
            tokens=S,
            tp=TP,
            first=first,
            qkvz=fmt,
            qkvz_shuffled=shuffled,
            out=fmt,
            out_shuffled=shuffled,
            first_q4=q4,
        )
        a = dict.fromkeys(K.ABI.names, 0)
        a |= dict(
            scratch=scratch,
            emb=h_in,
            res_in=res_in,
            core_out=core,
            n_in=n_in,
            w_qkvz=gk["qkvz"],
            w_qkvz_s=gk["qkvz_s"],
            w_ba=gk["ba"],
            w_conv=gk["conv"],
            a_log=gk["a_log"],
            dt_bias=gk["dt_bias"],
            gdn_norm=gk["norm"],
            conv_state=conv,
            conv_stride=conv.stride(0),
            ssm=ssm,
            ssm_stride=ssm.stride(0),
            st_idx=st_idx,
            rank=rank,
            epoch=epoch,
            layer=i,
            folds=base + 4 * FOLD_BASE,
            health=health,
            alarm=0,
            trace=0,
            sync=slot_ptr[sched[seq][0]],
            sync_next=slot_ptr[sched[seq][1]],
        )
        K.run(key, a)
        check_health()

    def attn_launch(seq, i, n_in, h_in, res_in):
        key = K.LayerBuild(kind="attn", tokens=S, tp=TP)
        a = dict.fromkeys(K.ABI.names, 0)
        a |= dict(
            scratch=scratch,
            emb=h_in,
            res_in=res_in,
            n_in=n_in,
            w_qkv=w_qkv,
            rank=rank,
            epoch=epoch,
            layer=i,
            folds=base + 4 * FOLD_BASE,
            health=health,
            alarm=0,
            trace=0,
            sync=slot_ptr[sched[seq][0]],
            sync_next=slot_ptr[sched[seq][1]],
        )
        K.run(key, a)
        check_health()

    def mono_launch(i, core_in, gate_in, residual, w, w_s, n_post, og):
        m = moes[i]
        oq = fmt != "bf16" and not og
        key = blocks.LayerBuild(
            tokens=S, oq=oq, og=og, shuffled=shuffled if oq else True
        )
        blocks.layer_block(
            key,
            core=core_in,
            gate=gate_in,
            residual=residual,
            w_o=w,
            w_os=w_s if oq else None,
            ln_w=n_post,
            w_gate=m["w_gate"][: L.EXPERTS].contiguous(),
            w_sg=m["w_gate"][L.EXPERTS].contiguous(),
            w13=m["w13"],
            w13s=m["w13s"],
            w2=m["w2"],
            w2s=m["w2s"],
            out=h,
            res_out=res,
            scratch=mscratch,
            peers=mtable,
            rank=rank,
            epoch=epoch[:1],
            health=mhealth,
            alarm=malarm,
            layer=blocks.LAYER_SLOT + i,
        )
        check_health()

    def allsum(part):
        parts = [torch.empty_like(part.cpu()) for _ in range(TP)]
        dist.all_gather(parts, part.cpu().contiguous())
        tot = parts[0].double()
        for p in parts[1:]:
            tot = tot + p.double()
        return R.bf(tot).to(dev)

    def check_core(tag, ref):
        want = R.rotate32(ref) if fmt == "mxfp4_h32" else ref
        note(f"{tag} core (live rows)", rel(core[live], want[live]), 1e-2)

    def moe_exact(xn, m, ids, ws):
        """The experts in f64 on the dequantized weights, x and the
        intermediate unquantized."""
        y = torch.zeros_like(xn)
        for e in ids.unique().tolist():
            w13d = R.dequant_mx(m["r13"][e], m["r13s"][e])
            w2d = R.dequant_mx(m["r2"][e], m["r2s"][e])[:, : L.RI]
            for b_, j in (ids == e).nonzero().tolist():
                gu = w13d @ xn[b_]
                act = torch.nn.functional.silu(gu[: L.RI]) * gu[rip : rip + L.RI]
                y[b_] += ws[b_, j] * (w2d @ act)
        return y

    def moe_stock(xn, m, ids, ws):
        """AITER's fused_moe (the stock MXFP4 decode path) on the same routes."""
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe

        return fused_moe(
            xn.to(torch.bfloat16),
            m["w13"].view(torch.float4_e2m1fn_x2),
            m["w2"].view(torch.float4_e2m1fn_x2),
            ws.float(),
            ids.to(torch.int32),
            None,
            ActivationType.Silu,
            QuantType.per_1x32,
            False,
            m["w13s"].view(torch.float8_e8m0fnu),
            m["w2s"].view(torch.float8_e8m0fnu),
            None,
            None,
            dtype=torch.bfloat16,
        )

    def check_mlp(tag, res_ref, n_post, i):
        """layer_block's residual, and its MoE output: its error to the exact
        MoE (of the kernel's residual's norm) at most `MOE_ERR_RATIO` times
        the stock fused_moe's, cosine to the exact above 0.95."""
        note(f"{tag} residual", rel(res, res_ref), 1e-2)
        xn = R.gemma_norm(res.double(), n_post, eps)
        m = moes[i]
        ids, ws, _ = R.route(xn, m["w_gate"], L.EXPERTS, L.TOPK)
        exact = allsum(R.bf(moe_exact(xn, m, ids, ws)))
        stock = allsum(moe_stock(xn, m, ids, ws))
        err, err_stock = rel(h, exact), rel(stock, exact)
        cos = float(
            torch.nn.functional.cosine_similarity(
                h.double().flatten(), exact.flatten(), dim=0
            )
        )
        note(f"{tag} moe err / stock err", err / err_stock, MOE_ERR_RATIO)
        note(f"{tag} moe 1 - cos(exact)", 1 - cos, 5e-2)

    eps = 1e-6
    # ---- layer 0: GDN (first) -> core -> layer_block
    conv0, ssm0 = convs[0].clone(), ssms[0].clone()
    gdn_launch(0, 0, g0k, conv0, ssm0, norms[0], True, emb, 0)
    z0 = emb.float()
    x0 = R.bf(
        z0
        * torch.rsqrt((z0 * z0).mean(-1, keepdim=True) + eps)
        * (1 + norms[0].float())
    )
    _, _, rconv, rssm, rcore, outp = R.gdn(
        x0, g0r, st_idx, convs[0], ssms[0], eps, (fmt, fmt), shuffled
    )
    note("L0 conv state", rel(conv0, rconv), 1e-2)
    note("L0 ssm state", rel(ssm0, rssm), 1e-2)
    check_core("L0", rcore)
    note("L0 input residual", rel(sview("res", L.HIDDEN), emb), 1e-9)
    kres = sview("res", L.HIDDEN).clone()
    mono_launch(
        0, core, None, sview("res", L.HIDDEN), g0k["out"], g0k["out_s"], norms[1], False
    )
    # an h32 core is rotated already: the rotated weight's plain MXFP4 GEMM
    kout = R.linear(
        core.double(), g0r["out"], "mxfp4" if fmt != "bf16" else "bf16", shuffled
    )
    check_mlp("L0", R.bf(kres.double() + allsum(kout)), norms[1], 0)

    # ---- layer 1: attention: input combine + qkv_proj, then layer_block on
    # synthetic attention rows
    hres = (h.clone(), res.clone())
    attn_launch(1, 1, norms[2], hres[0], hres[1])
    z1 = R.bf(hres[1].double() + hres[0].double())
    note("L1 input residual", rel(sview("res", L.HIDDEN), z1), 1e-6)
    x1 = R.gemma_norm(z1, norms[2], eps)
    note("L1 x", rel(sview("x", L.HIDDEN), x1), 1e-2)
    kx1 = sview("x", L.HIDDEN).double()
    note(
        "L1 qkv_proj", rel(sview("qkv_out", L.QKV), R.bf(kx1 @ w_qkv.double().T)), 1e-2
    )
    core1 = R.bf(attn.double() * R.bf(torch.sigmoid(gate.double())))
    mono_launch(1, attn, gate, sview("res", L.HIDDEN), w_o, 0, norms[3], True)
    check_mlp("L1", R.bf(z1 + allsum(R.bf(core1 @ w_o.double().T))), norms[3], 1)

    # ---- layer 2: GDN (h + res) -> core -> layer_block
    h1, r1 = h.clone(), res.clone()
    conv2, ssm2 = convs[1].clone(), ssms[1].clone()
    gdn_launch(2, 2, g2k, conv2, ssm2, norms[4], False, h1, r1)
    z2 = R.bf(r1.double() + h1.double())
    note("L2 input residual", rel(sview("res", L.HIDDEN), z2), 1e-6)
    x2 = R.gemma_norm(z2, norms[4], eps)
    note("L2 x", rel(sview("x", L.HIDDEN), x2), 1e-2)
    _, _, rconv, rssm, rcore, _ = R.gdn(
        sview("x", L.HIDDEN).double(),
        g2r,
        st_idx,
        convs[1],
        ssms[1],
        eps,
        (fmt, fmt),
        shuffled,
    )
    note("L2 conv state", rel(conv2, rconv), 1e-2)
    note("L2 ssm state", rel(ssm2, rssm), 1e-2)
    check_core("L2", rcore)
    kres = sview("res", L.HIDDEN).clone()
    mono_launch(
        2, core, None, sview("res", L.HIDDEN), g2k["out"], g2k["out_s"], norms[5], False
    )
    kout = R.linear(
        core.double(), g2r["out"], "mxfp4" if fmt != "bf16" else "bf16", shuffled
    )
    check_mlp("L2", R.bf(kres.double() + allsum(kout)), norms[5], 2)

    out_q.put((rank, log))
    dist.barrier()
    mpeers.close()


@pytest.mark.skipif(not _gfx950(TP), reason="layer_block runs TP8 on 8 gfx950")
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("rows", WIDTHS)
def test_split_layers_tp8(rows, variant):
    import torch.multiprocessing as mp

    port = 31500 + (os.getpid() + 37 * rows + VARIANTS.index(variant)) % 2000
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
            rank, log = q.get(timeout=1800)
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


@pytest.mark.skipif(not _gfx950(1), reason="the wide kernels run on gfx950")
@pytest.mark.parametrize("variant", ["mxfp4", "mxfp4t", "h32"])
@pytest.mark.parametrize("rows", [1, 16])
def test_split_gdn_quant(rows, variant):
    from aiter import per_1x32_f4_quant_hip
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.models.qwen3_5.amd.specialized_decode import layout as L
    from vllm.models.qwen3_5.amd.specialized_decode.abi import FAILED, health_words
    from vllm.models.qwen3_5.amd.specialized_decode.wide import layer as K
    from vllm.models.qwen3_5.amd.specialized_decode.wide.grid import (
        FOLD_BASE,
        SLOT_WORDS,
        launch_slots,
        sync_words,
    )

    from . import reference as R

    S = rows
    fmt = {"bf16": "bf16", "mxfp4": "mxfp4", "mxfp4t": "mxfp4", "h32": "mxfp4_h32"}[
        variant
    ]
    shuffled = variant != "mxfp4t"
    q4 = K.q4_of(fmt, shuffled) or "none"
    dev = torch.device("cuda", 0)
    g = torch.Generator(device=dev).manual_seed(7)

    def rn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, device=dev) * scale).to(dtype)

    def dense_or_fp4(rows, k):
        if fmt == "bf16":
            w = rn(rows, k, scale=0.02)
            return w, 0, w
        codes, sc = R.mxfp4(rows, k, g, lo=119, hi=123, device=dev)
        ref = R.dequant_mx(codes, sc)
        if shuffled:
            return R.shuffle_data(codes), R.shuffle_scale(sc), ref
        return codes.contiguous(), sc.T.contiguous(), ref

    qkvz, qkvz_s, qkvz_r = dense_or_fp4(L.QKVZ, L.HIDDEN)
    w = dict(
        ba=rn(L.BA, L.HIDDEN, scale=0.02),
        conv=rn(L.CONV_DIM, 1, 4, scale=0.5),
        a_log=torch.randn(L.NV, generator=g, device=dev) * 0.5,
        dt_bias=rn(L.NV, scale=0.5),
        norm=rn(L.GDN_DV, scale=0.2) + 1,
    )
    wr = dict(w, qkvz=qkvz_r, out=torch.zeros(1, L.VALUE_DIM, device=dev))
    n_in = rn(L.HIDDEN, scale=0.1)
    emb = rn(S, L.HIDDEN, scale=3.0)
    res_in = rn(S, L.HIDDEN, scale=3.0)
    slots = S + 2
    st_idx = (
        torch.randperm(S, generator=g, device=dev).to(torch.int32) + 1
    ).contiguous()
    conv = rn(slots, 3, L.CONV_DIM, scale=0.5)
    ssm = rn(slots, L.NV, L.GDN_DV, L.GDN_DK, scale=0.1, dtype=torch.float32)
    conv_k, ssm_k = conv.clone(), ssm.clone()

    sync = torch.zeros(sync_words(), dtype=torch.int32, device=dev)
    health = torch.zeros(health_words(L.BLOCKS), dtype=torch.int32, device=dev)
    epoch = torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=dev)
    rip = L.RI
    scratch = torch.zeros(K.scratch_bytes(S, rip), dtype=torch.uint8, device=dev)
    lay = K.scratch_layout(S, rip)
    base = sync.data_ptr()
    sched = launch_slots(2)
    core = torch.empty(S, L.VALUE_DIM, dtype=torch.bfloat16, device=dev)

    def sview(name, cols, dtype=torch.bfloat16):
        off, n = lay[name]
        el = torch.tensor([], dtype=dtype).element_size()
        return scratch[off : off + S * cols * el].view(dtype).view(S, cols)

    key = K.LayerBuild(
        kind="gdn",
        tokens=S,
        tp=8,
        first=False,
        qkvz=fmt,
        qkvz_shuffled=shuffled,
        out=fmt,
        out_shuffled=shuffled,
        first_q4=q4,
    )
    a = dict.fromkeys(K.ABI.names, 0)
    a |= dict(
        scratch=scratch,
        emb=emb,
        res_in=res_in,
        core_out=core,
        n_in=n_in,
        w_qkvz=qkvz,
        w_qkvz_s=qkvz_s,
        w_ba=w["ba"],
        w_conv=w["conv"],
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        gdn_norm=w["norm"],
        conv_state=conv_k,
        conv_stride=conv_k.stride(0),
        ssm=ssm_k,
        ssm_stride=ssm_k.stride(0),
        st_idx=st_idx,
        rank=0,
        epoch=epoch,
        layer=0,
        folds=base + 4 * FOLD_BASE,
        health=health,
        alarm=0,
        trace=0,
        sync=base + 4 * SLOT_WORDS * sched[0][0],
        sync_next=base + 4 * SLOT_WORDS * sched[0][1],
    )
    K.run(key, a)
    torch.accelerator.synchronize()
    if int(health[FAILED]):
        raise RuntimeError(f"in-kernel wait gave up: {health[:16].tolist()}")

    eps = 1e-6
    log = []

    def note(name, v, tol):
        log.append((name, v, tol))

    z = R.bf(res_in.double() + emb.double())
    note("residual", rel(sview("res", L.HIDDEN), z), 0)
    kx = sview("x", L.HIDDEN)
    note("x", rel(kx, R.gemma_norm(z, n_in, eps)), 1e-2)
    if q4 != "none":
        xq = R.rotate32(kx.double()).to(torch.bfloat16) if q4.startswith("h32") else kx
        if q4.endswith("_t"):
            qq, qs = dynamic_mxfp4_quant(xq.contiguous())
        else:
            qq, qs = per_1x32_f4_quant_hip(xq.contiguous(), shuffle=False)
        kq = sview("x4", L.HIDDEN // 2, torch.uint8)
        ks = sview("x4s", L.HIDDEN // 32, torch.int32)
        qs = qs.view(torch.uint8)[:S, : L.HIDDEN // 32].int()
        note(
            "x4 scale codes differing (frac)",
            float((ks & 0xFF != qs).float().mean()),
            0,
        )
        flips = float((kq != qq.view(torch.uint8)[:S]).float().mean())
        note("x4 fp4 bytes differing (frac)", flips, 0)
    x_ref = kx.double()
    _, _, rconv, rssm, rcore, _ = R.gdn(
        x_ref, wr, st_idx, conv, ssm, eps, (fmt, "bf16"), shuffled
    )
    note("conv state", rel(conv_k, rconv), 1e-2)
    note("ssm state", rel(ssm_k, rssm), 1e-2)
    want = R.rotate32(rcore) if fmt == "mxfp4_h32" else rcore
    note("core", rel(core, want), 1e-2)
    bad = [f"{name}: {v:.2e} > {tol:.0e}" for name, v, tol in log if not v <= tol]
    assert not bad, "\n".join(bad)
