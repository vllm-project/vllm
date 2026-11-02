# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The wide decode's stages on MI355X against float64 references with the
stock path's rounding points, at the shapes of Qwen3.8-Flash-Next at TP2, one
GPU (the TP exchange with one rank): the batch-tiled GEMM on each weight
format, the gated residual (stats, mix, combine), the gated delta net (input
projection + conv, recurrence + gated norm, out_proj into the exchange) and
the MoE (router, routing, AITER-layout experts in place, combine).

Tolerances: an output rounded to bf16 is within one bf16 ulp (2^-8 relative)
of the reference rounded at the same points, two where it follows another
bf16-rounded value; a max-relative bound of 1e-2 covers that. Routing is
compared as sets: bf16 logits tie at the top-k boundary, so a few rows may
pick different experts; their outputs are excluded from the MoE bound."""

import pytest
import torch

from vllm.platforms import current_platform

from . import reference as ref
from .reference import mxfp4, shuffle_data, shuffle_scale


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the wide kernels need gfx950 (MI355X)"
)

ROWS = 128


def rel(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).abs().max() / b.abs().max()).item()


class Harness:
    """A one-GPU launch environment: sync slots, health, a one-rank mailbox."""

    def __init__(self):
        from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

        from vllm.models.qwen4_exp.amd.specialized_decode.runtime.peer import (
            _DeviceBytes,
        )
        from vllm.models.qwen4_exp.amd.specialized_decode.wide.common import grid
        from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange

        self.dev = torch.device("cuda")
        self.sync = torch.zeros(grid.sync_words(), dtype=torch.int32, device=self.dev)
        self.health = torch.zeros(8192, dtype=torch.int32, device=self.dev)
        self.epoch = torch.zeros(4, dtype=torch.int32, device=self.dev)
        nbytes = exchange.mailbox_bytes(1)
        self.mbox = UncachedIpcHeap.alloc_uncached(nbytes)
        self.mbox_t = torch.as_tensor(
            _DeviceBytes(self.mbox, nbytes), device=self.dev
        ).view(torch.int32)
        self.mbox_t.zero_()
        self.seq = 0

    def sync_args(self):
        from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
            FOLD_BASE,
            SLOT_WORDS,
        )

        base = self.sync.data_ptr()
        k = self.seq % 2
        self.seq += 1
        return [
            base + 4 * k * SLOT_WORDS,
            base + 4 * (1 - k) * SLOT_WORDS,
            base + 4 * FOLD_BASE,
            self.health.data_ptr(),
        ]

    def exchange_y(self, rows, slot, codec="bf16"):
        """The rank's partial in exchange slot ``slot`` as [rows, HIDDEN]."""
        from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import HIDDEN
        from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import (
            exchange as ex,
        )

        words, lpg = ex.row_words(), ex.lines_per_group(rows)
        row = self.mbox_t[slot * words : slot * words + ex.payload_words(rows, codec)]
        if codec == "int4":
            return ref.int4_lines(row, rows, ex.GROUPS, lpg, HIDDEN)
        y = (
            row.contiguous()
            .view(torch.bfloat16)
            .view(ex.GROUPS, lpg, ex.TOKENS, ex.COLS)
        )
        return y.permute(1, 2, 0, 3).reshape(lpg * ex.TOKENS, HIDDEN)[:rows]


@pytest.fixture(scope="module")
def harness():
    return Harness()


# --------------------------------------------------------------------------
# kernels: one stage family a launch, built from the package's stages
# --------------------------------------------------------------------------


def _gemm_kernel(rows, n, k, splits, fmt, fold):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.typing import Int64

    default_stream = fx.Stream(None)

    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
        Grid,
        grid_sync,
        prologue,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import CM_DEV, st
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import BLOCKS, THREADS
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import tiled

    plan = tiled.SplitK(n, k, splits)

    @fx.struct
    class Smem:
        stage: fx.Array[fx.Int32, tiled.stage_words(rows), 16]
        word: fx.Array[fx.Int32, 4, 16]

    def weights(w, ws):
        if fmt == "bf16":
            return tiled.Bf16Weights(w, n, k)
        return tiled.Fp4Weights(w, ws, n, k, shuffled=fmt == "fp4_asm")

    @flyc.kernel(
        name=f"t_gemm_{rows}_{n}_{k}_{splits}_{fmt}_{int(fold)}",
        known_block_size=[THREADS, 1, 1],
    )
    def kern(
        x: Int64,
        w: Int64,
        ws: Int64,
        out: Int64,
        part: Int64,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        sm = fx.SharedAllocator().allocate(Smem).peek()
        g = Grid(tid, bid, s0, s1, folds, health, fx.Int64(0), sm.word.ptr)
        prologue(g)
        gemm = tiled.Gemm(sm.stage.ptr, rows, tid, k_total=k)
        if fx.const_expr(not fold):
            tiled.gemm_units(
                bid,
                BLOCKS,
                gemm,
                tiled.Source(x, k),
                weights(w, ws),
                plan,
                tiled.partial_epilogue(gemm, plan, part),
            )
            grid_sync(g)
            n4 = plan.padded // 4
            for i in range(bid * THREADS + tid, rows * n4, BLOCKS * THREADS):
                tot = tiled.sum_partials4(part, plan, rows, i // n4, i % n4 * 4)
                st(out, i * 4, fx.Vector.from_elements(tot, fx.Float32), cm=CM_DEV)
        else:

            def finish(group, first, lane, acc):
                for t in fx.range_constexpr(gemm.tiles):
                    token = t * 16 + lane % 16
                    if token < rows:
                        st(out, token * plan.padded + first + lane // 16 * 4, acc[t])

            epi, after = tiled.fold(g, gemm, plan, part, 0, finish)
            tiled.gemm_units(
                bid, BLOCKS, gemm, tiled.Source(x, k), weights(w, ws), plan, epi, after
            )

    @flyc.jit
    def launch(
        x: Int64,
        w: Int64,
        ws: Int64,
        out: Int64,
        part: Int64,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
        stream: fx.Stream = default_stream,
    ):
        kern(x, w, ws, out, part, s0, s1, folds, health).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch, plan


@pytest.mark.parametrize("fmt", ["bf16", "fp4_triton", "fp4_asm"])
@pytest.mark.parametrize("fold", [False, True])
def test_tiled_gemm(harness, fmt, fold):
    """``x @ W.T`` of the gated residual's down shape (split K, summed in order
    or folded by the last arriver), bf16 or MXFP4 weights in either AITER
    layout: f32 accumulation of exact bf16 products."""
    rows, n, k, splits = ROWS, 352, 2560, 4
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(rows, k, generator=gen).to(torch.bfloat16)
    if fmt == "bf16":
        w = (torch.randn(n, k, generator=gen) * 0.05).to(torch.bfloat16)
        wref, wd, wsd = w.double(), w.cuda(), torch.zeros(1, device="cuda")
    else:
        packed, sc, wref = mxfp4(n, k, gen)
        if fmt == "fp4_triton":
            wd, wsd = packed.cuda(), sc.T.contiguous().cuda()
        else:
            wd, wsd = shuffle_data(packed).cuda(), shuffle_scale(sc).cuda()
    launch, plan = _gemm_kernel(rows, n, k, splits, fmt, fold)
    out = torch.zeros(rows, plan.padded, device="cuda")
    part = torch.zeros(plan.partial_words(rows), device="cuda")
    xd = x.cuda()
    launch(
        xd.data_ptr(),
        wd.data_ptr(),
        wsd.data_ptr(),
        out.data_ptr(),
        part.data_ptr(),
        *harness.sync_args(),
        torch.cuda.current_stream(),
    )
    torch.accelerator.synchronize()
    assert harness.health[0].item() == 0
    assert rel(out[:, :n], x.double() @ wref.double().T) < 1e-5


def test_fp4_shuffle_matches_aiter():
    """The shuffled layout the kernels read is AITER's ``shuffle_weight``."""
    from aiter.ops.shuffle import shuffle_weight

    gen = torch.Generator().manual_seed(1)
    packed = torch.randint(0, 256, (2, 64, 128), generator=gen, dtype=torch.uint8)
    assert torch.equal(
        shuffle_weight(packed.cuda(), layout=(16, 16)).cpu().view(torch.uint8),
        shuffle_data(packed),
    )


# --------------------------------------------------------------------------
# the gated residual: stats, mix (down, reduce, up), exchange push, combine
# --------------------------------------------------------------------------

EPS = 1e-6


def _hc_kernel(rows, codec="bf16"):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.typing import Int64

    default_stream = fx.Stream(None)

    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
        Grid,
        grid_sync,
        prologue,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
        CM_DEV,
        ld_i32x4,
        lds_st,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        BLOCKS,
        HIDDEN,
        LOWRANK,
        THREADS,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import hc, tiled

    @fx.struct
    class Smem:
        stage: fx.Array[fx.Int32, tiled.stage_words(rows), 16]
        red: fx.Array[fx.Float32, hc.up_lds_words(rows), 16]
        tile: fx.Array[fx.Int32, ex.GroupTile.words(rows), 16]
        word: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=f"t_hc_{rows}_{codec}", known_block_size=[THREADS, 1, 1])
    def kern(
        state: Int64,
        norm_w: Int64,
        w_di: Int64,
        w_up: Int64,
        x_out: Int64,
        inj: Int64,
        rrms: Int64,
        part: Int64,
        down: Int64,
        y: Int64,
        mbox: Int64,
        rrms2: Int64,
        epoch: Int64,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        sm = fx.SharedAllocator().allocate(Smem).peek()
        g = Grid(tid, bid, s0, s1, folds, health, fx.Int64(0), sm.word.ptr)
        prologue(g)
        hc.stats(g, state, rrms, rows, EPS)
        grid_sync(g)
        hc.mix_down(
            g, tiled.Gemm(sm.stage.ptr, rows, tid), state, rrms, norm_w, w_di, part
        )
        grid_sync(g)
        hc.mix_reduce(g, part, down, inj, rows, True)
        grid_sync(g)
        up = tiled.Gemm(sm.stage.ptr, rows, tid, k_total=LOWRANK)
        hc.mix_up(g, up, sm.red.ptr, down, w_up, state, rrms, norm_w, x_out, rows)
        grid_sync(g)
        # y through a one-rank exchange (a 64-column group a CTA), then the combine
        tag = fx.Int32(ld_i32x4(epoch, 0)[0]) * 128 + 5
        xch = ex.Exchange([fx.Int64(mbox)], mbox, 0, 1, 1, tag, rows, codec)
        tile = ex.GroupTile(sm.tile.ptr, rows)
        for grp in range(bid, ex.GROUPS, BLOCKS):
            for idx in range(tid, rows * 8, THREADS):
                token, piece = idx // 8, idx % 8
                w = ld_i32x4(y, (token * HIDDEN + grp * 64 + piece * 8) // 2, CM_DEV)
                lds_st(tile.ptr + token * ex.GroupTile.STRIDE + piece * 4, w)
            fx.barrier()
            ex.push_tile(g, xch, tile, grp)
            fx.barrier()
        hc.combine(g, xch, state, inj, rrms2, rows, EPS)

    @flyc.jit
    def launch(
        state: Int64,
        norm_w: Int64,
        w_di: Int64,
        w_up: Int64,
        x_out: Int64,
        inj: Int64,
        rrms: Int64,
        part: Int64,
        down: Int64,
        y: Int64,
        mbox: Int64,
        rrms2: Int64,
        epoch: Int64,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
        stream: fx.Stream = default_stream,
    ):
        kern(
            state,
            norm_w,
            w_di,
            w_up,
            x_out,
            inj,
            rrms,
            part,
            down,
            y,
            mbox,
            rrms2,
            epoch,
            s0,
            s1,
            folds,
            health,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


@pytest.mark.parametrize("rows, codec", [(32, "bf16"), (128, "bf16"), (128, "int4")])
def test_gated_residual(harness, rows, codec):
    """The mix's normalized gated mean and injection scales, the combine's new
    state and its rrms; with ``int4`` exchange lines the combine adds the
    codec's round trip of the block output."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        DOWN_ROWS,
        HC,
        HC_HIDDEN,
        HIDDEN,
        LOWRANK,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import hc

    gen = torch.Generator().manual_seed(1)
    state = (torch.randn(rows, HC_HIDDEN, generator=gen) * 2).to(torch.bfloat16)
    norm_w = (torch.randn(HC_HIDDEN, generator=gen) * 0.1).to(torch.bfloat16)
    w_di = (torch.randn(DOWN_ROWS, HC_HIDDEN, generator=gen) * 0.02).to(torch.bfloat16)
    w_up = (torch.randn(HC_HIDDEN, LOWRANK, generator=gen) * 0.1).to(torch.bfloat16)
    y = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16)
    y_lines = ref.int4_roundtrip(y) if codec == "int4" else y
    mixed, inj_r, new, rrms_r, rrms2_r = ref.gated_residual(
        state, norm_w, w_di, w_up, y_lines, HC, LOWRANK, EPS
    )

    d = {
        n: v.cuda()
        for n, v in dict(state=state, norm_w=norm_w, w_di=w_di, w_up=w_up, y=y).items()
    }
    x_out = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    inj, rrms, rrms2 = (torch.zeros(rows, HC, device="cuda") for _ in range(3))
    part = torch.zeros(hc.down_plan().partial_words(rows), device="cuda")
    down = torch.zeros(rows, LOWRANK, dtype=torch.bfloat16, device="cuda")
    harness.mbox_t.zero_()
    _hc_kernel(rows, codec)(
        d["state"].data_ptr(),
        d["norm_w"].data_ptr(),
        d["w_di"].data_ptr(),
        d["w_up"].data_ptr(),
        x_out.data_ptr(),
        inj.data_ptr(),
        rrms.data_ptr(),
        part.data_ptr(),
        down.data_ptr(),
        d["y"].data_ptr(),
        harness.mbox,
        rrms2.data_ptr(),
        harness.epoch.data_ptr(),
        *harness.sync_args(),
        torch.cuda.current_stream(),
    )
    torch.accelerator.synchronize()
    assert harness.health[0].item() == 0
    assert rel(rrms, rrms_r) < 1e-5
    assert rel(inj, inj_r) < 1e-3
    assert rel(x_out, mixed) < 1e-2
    assert rel(d["state"], new) < 1e-2
    assert rel(rrms2, rrms2_r) < 1e-3


# --------------------------------------------------------------------------
# the gated delta net: in_proj + conv, recurrence + gated norm, out_proj
# --------------------------------------------------------------------------


def _gdn_kernel(rows):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.typing import Int64

    default_stream = fx.Stream(None)

    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
        Grid,
        grid_sync,
        prologue,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import ld_i32x4
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        BLOCKS,
        CONV_DIM,
        HIDDEN,
        NV,
        THREADS,
        VALUE_DIM,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import gdn, tiled

    @fx.struct
    class Smem:
        stage: fx.Array[fx.Int32, tiled.stage_words(rows), 16]
        tile: fx.Array[fx.Int32, ex.GroupTile.words(rows), 16]
        red: fx.Array[fx.Float32, gdn.recur_lds_words(), 16]
        word: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=f"t_gdn_{rows}", known_block_size=[THREADS, 1, 1])
    def kern(
        x: Int64,
        wq: Int64,
        wq_scale: Int64,
        wba: Int64,
        wconv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        normw: Int64,
        wo: Int64,
        wo_scale: Int64,
        qkv: Int64,
        z: Int64,
        ba: Int64,
        core: Int64,
        part: Int64,
        st_idx: Int64,
        conv_state: Int64,
        ssm: Int64,
        mbox: Int64,
        epoch: Int64,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        sm = fx.SharedAllocator().allocate(Smem).peek()
        g = Grid(tid, bid, s0, s1, folds, health, fx.Int64(0), sm.word.ptr)
        prologue(g)
        w = gdn.Bufs(
            qkvz=wq,
            qkvz_s=wq_scale,
            shuffled=False,
            ba=wba,
            conv=wconv,
            a_log=a_log,
            dt_bias=dt_bias,
            norm=normw,
        )
        b = gdn.Bufs(
            x=x,
            qkv=qkv,
            z=z,
            ba=ba,
            core=core,
            part=part,
            st_idx=st_idx,
            conv_state=conv_state,
            ssm=ssm,
            conv_stride=fx.Int64(3 * CONV_DIM),
            ssm_stride=fx.Int64(NV * 128 * 128),
        )
        gemm = tiled.Gemm(sm.stage.ptr, rows, tid)
        gdn.proj(g, gemm, w, b, rows, 0)
        grid_sync(g)
        gdn.recur(g, w, b, sm.red.ptr, rows, EPS)
        grid_sync(g)
        tag = fx.Int32(ld_i32x4(epoch, 0)[0]) * 128 + 3
        xch = ex.Exchange([fx.Int64(mbox)], mbox, 0, 1, 0, tag, rows)
        weights = tiled.Fp4Weights(wo, wo_scale, HIDDEN, VALUE_DIM, False)
        gdn.out(
            g,
            gemm,
            weights,
            tiled.Source(core, VALUE_DIM),
            part,
            ex.GroupTile(sm.tile.ptr, rows),
            xch,
            VALUE_DIM,
            200,
        )

    @flyc.jit
    def launch(
        x: Int64,
        wq: Int64,
        wq_scale: Int64,
        wba: Int64,
        wconv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        normw: Int64,
        wo: Int64,
        wo_scale: Int64,
        qkv: Int64,
        z: Int64,
        ba: Int64,
        core: Int64,
        part: Int64,
        st_idx: Int64,
        conv_state: Int64,
        ssm: Int64,
        mbox: Int64,
        epoch: Int64,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
        stream: fx.Stream = default_stream,
    ):
        kern(
            x,
            wq,
            wq_scale,
            wba,
            wconv,
            a_log,
            dt_bias,
            normw,
            wo,
            wo_scale,
            qkv,
            z,
            ba,
            core,
            part,
            st_idx,
            conv_state,
            ssm,
            mbox,
            epoch,
            s0,
            s1,
            folds,
            health,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


@pytest.mark.parametrize("rows", [32, 128])
def test_gated_delta_net(harness, rows):
    """Conv outputs and states, z, the recurrent states, the gated-norm core
    and out_proj through the exchange; two padding rows (slot 0) leave the
    states alone and give a zero core."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        BA,
        CONV_DIM,
        HIDDEN,
        NV,
        QKVZ,
        VALUE_DIM,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import gdn

    gen = torch.Generator().manual_seed(2)
    slots = rows + 9
    x = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16)
    wq_data, wq_scale, wq = mxfp4(QKVZ, HIDDEN, gen, 115, 121)
    wba = (torch.randn(BA, HIDDEN, generator=gen) * 0.02).to(torch.bfloat16)
    wconv = (torch.randn(CONV_DIM, 1, 4, generator=gen) * 0.5).to(torch.bfloat16)
    a_log = torch.randn(NV, generator=gen) * 0.5
    dt_bias = (torch.randn(NV, generator=gen) * 0.5).to(torch.bfloat16)
    normw = (1 + torch.randn(128, generator=gen) * 0.1).to(torch.bfloat16)
    wo_data, wo_scale, wo = mxfp4(HIDDEN, VALUE_DIM, gen, 116, 121)
    st_idx = (torch.randperm(slots - 1, generator=gen)[:rows] + 1).to(torch.int32)
    st_idx[3] = 0
    st_idx[rows - 1] = 0
    conv_state = torch.randn(slots, 3, CONV_DIM, generator=gen).to(torch.bfloat16)
    ssm = torch.randn(slots, NV, 128, 128, generator=gen) * 0.05
    conv_r, z_r, cs_r, ssm_r, core_r, y_r = ref.gated_delta_net(
        x, wq, wba, wconv, a_log, dt_bias, normw, wo, st_idx, conv_state, ssm, EPS
    )

    d = {
        n: v.cuda()
        for n, v in dict(
            x=x,
            wq_data=wq_data,
            wq_scale=wq_scale.T.contiguous(),
            wba=wba,
            wconv=wconv,
            a_log=a_log,
            dt_bias=dt_bias,
            normw=normw,
            wo_data=wo_data,
            wo_scale=wo_scale.T.contiguous(),
            st_idx=st_idx,
            conv_state=conv_state,
            ssm=ssm,
        ).items()
    }
    qkv = torch.zeros(rows, CONV_DIM, dtype=torch.bfloat16, device="cuda")
    z = torch.zeros(rows, VALUE_DIM, dtype=torch.bfloat16, device="cuda")
    ba = torch.zeros(gdn.ba_plan().partial_words(rows), device="cuda")
    core = torch.zeros(rows, VALUE_DIM, dtype=torch.bfloat16, device="cuda")
    part = torch.zeros(
        max(gdn.proj_plan().partial_words(rows), gdn.out_plan().partial_words(rows)),
        device="cuda",
    )
    harness.mbox_t.zero_()
    _gdn_kernel(rows)(
        *[
            d[n].data_ptr()
            for n in (
                "x",
                "wq_data",
                "wq_scale",
                "wba",
                "wconv",
                "a_log",
                "dt_bias",
                "normw",
                "wo_data",
                "wo_scale",
            )
        ],
        qkv.data_ptr(),
        z.data_ptr(),
        ba.data_ptr(),
        core.data_ptr(),
        part.data_ptr(),
        d["st_idx"].data_ptr(),
        d["conv_state"].data_ptr(),
        d["ssm"].data_ptr(),
        harness.mbox,
        harness.epoch.data_ptr(),
        *harness.sync_args(),
        torch.cuda.current_stream(),
    )
    torch.accelerator.synchronize()
    assert harness.health[0].item() == 0
    assert rel(qkv, conv_r) < 1e-2
    assert rel(z, z_r) < 1e-2
    assert rel(d["conv_state"], cs_r) < 1e-2
    assert rel(d["ssm"], ssm_r) < 1e-2
    assert rel(core, core_r) < 1e-2
    assert core[3].abs().max().item() == 0 and core[rows - 1].abs().max().item() == 0
    assert rel(harness.exchange_y(rows, 0), y_r) < 2e-2


# --------------------------------------------------------------------------
# the MoE: router, routing + activation quant, experts in place, combine
# --------------------------------------------------------------------------

RIP = 384  # the padded intermediate of a TP2 rank (320 real)


def _moe_kernel(rows, codec="bf16"):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.typing import Int64

    default_stream = fx.Stream(None)

    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
        Grid,
        grid_sync,
        prologue,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import ld_i32x4
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import BLOCKS, THREADS
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import moe, tiled

    shape = moe.Shape(rows, RIP)
    words = moe.lds_words(shape)

    @fx.struct
    class Smem:
        stage: fx.Array[fx.Int32, tiled.stage_words(rows), 16]
        index: fx.Array[fx.Int32, words.index, 16]
        units: fx.Array[fx.Int32, words.units, 16]
        scan: fx.Array[fx.Int32, words.scan, 16]
        x: fx.Array[fx.Int32, words.x, 16]
        xs: fx.Array[fx.Int32, words.xs, 16]
        act: fx.Array[fx.Int32, words.act, 16]
        acts: fx.Array[fx.Int32, words.acts, 16]
        mstage: fx.Array[fx.Int32, words.stage, 16]
        word: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=f"t_moe_{rows}_{codec}", known_block_size=[THREADS, 1, 1])
    def kern(
        x: Int64,
        w_gate: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        part: Int64,
        route: Int64,
        x4: Int64,
        x4s: Int64,
        mpart: Int64,
        mpart2: Int64,
        flags: Int64,
        mbox: Int64,
        epoch: Int64,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        sm = fx.SharedAllocator().allocate(Smem).peek()
        g = Grid(tid, bid, s0, s1, folds, health, fx.Int64(0), sm.word.ptr)
        prologue(g)
        moe.router(g, tiled.Gemm(sm.stage.ptr, rows, tid), w_gate, x, part)
        grid_sync(g)
        moe.quantize_rows(g, x, x4, x4s, rows)
        moe.route(g, part, route, rows)
        grid_sync(g)
        lds = moe.Lds(
            index=sm.index.ptr,
            units=sm.units.ptr,
            scan=sm.scan.ptr,
            x=sm.x.ptr,
            xs=sm.xs.ptr,
            act=sm.act.ptr,
            acts=sm.acts.ptr,
            stage=sm.mstage.ptr,
        )
        tag = fx.Int32(ld_i32x4(epoch, 0)[0]) * 128 + 9
        moe.experts(
            g, shape, lds, route, x4, x4s, w13, w13s, w2, w2s, mpart, mpart2, flags, tag
        )
        grid_sync(g)
        moe.combine(
            g,
            ex.Exchange([fx.Int64(mbox)], mbox, 0, 1, 2, tag, rows, codec),
            route,
            mpart,
            mpart2,
            flags,
            rows,
        )

    @flyc.jit
    def launch(
        x: Int64,
        w_gate: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        part: Int64,
        route: Int64,
        x4: Int64,
        x4s: Int64,
        mpart: Int64,
        mpart2: Int64,
        flags: Int64,
        mbox: Int64,
        epoch: Int64,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
        stream: fx.Stream = default_stream,
    ):
        kern(
            x,
            w_gate,
            w13,
            w13s,
            w2,
            w2s,
            part,
            route,
            x4,
            x4s,
            mpart,
            mpart2,
            flags,
            mbox,
            epoch,
            s0,
            s1,
            folds,
            health,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


def _expert_weights(gen, experts, rows, k, rows_real, k_real):
    """MXFP4 experts [experts, rows, k] in AITER's shuffled layouts, zero past
    ``rows_real`` rows of each half (``rows`` covers two) and past ``k_real``:
    (data, scales [experts, rows, pad8(k / 32)], dequantized f32)."""
    codes = torch.randint(0, 16, (experts, rows, k), generator=gen, dtype=torch.uint8)
    codes[:, :, k_real:] = 0
    cols = (k // 32 + 7) // 8 * 8
    sc = torch.full((experts, rows, cols), 127, dtype=torch.uint8)
    sc[:, :, : k // 32] = torch.randint(
        118, 124, (experts, rows, k // 32), generator=gen, dtype=torch.uint8
    )
    half = rows // 2 if rows_real < rows else rows
    for h in range(rows // half):
        codes[:, h * half + rows_real : (h + 1) * half] = 0
    deq = ref.E2M1[codes.long()] * torch.pow(
        2.0, sc[:, :, : k // 32].float() - 127
    ).repeat_interleave(32, -1)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return (
        shuffle_data(packed),
        shuffle_scale(sc.view(experts * rows, cols)).view(experts, rows, cols),
        deq,
    )


def _routed_inputs(gen, rows, picks):
    """``x`` and a router gate whose top-k of row ``b`` is ``picks[b]``: each
    expert has a near-orthogonal direction ``u_e``; a row is the sum of its
    experts' directions (slightly unequal) plus noise, the gate rows ``2
    u_e``."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        E_ALL,
        EXPERTS,
        HIDDEN,
    )

    u = torch.randn(EXPERTS, HIDDEN, generator=gen)
    u = u / u.norm(dim=1, keepdim=True)
    x = 0.05 * torch.randn(rows, HIDDEN, generator=gen)
    for b, pick in enumerate(picks):
        w = 1 + 0.1 * torch.rand(len(pick), 1, generator=gen)
        x[b] += (u[list(pick)] * w).sum(0) * 4
    gate = torch.randn(E_ALL, HIDDEN, generator=gen) * 0.05
    gate[:EXPERTS] = u * 2
    return x.to(torch.bfloat16), gate.to(torch.bfloat16)


def _skewed_picks(rows, gen):
    """A real-checkpoint-like B128 routing: 20 hot experts of 24 rows and 280
    of about 3 (328 units with the shared expert's 8: 72 run as halves)."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import TOPK

    left = torch.zeros(512, dtype=torch.long)
    left[:20] = 24
    rest = rows * TOPK - 480
    left[20:300] = rest // 280
    left[20 : 20 + rest % 280] += 1
    picks = []
    for _ in range(rows):
        pick = torch.topk(left.float() + 1e-3 * torch.rand(512, generator=gen), TOPK)
        left[pick.indices] -= 1
        picks.append(pick.indices.tolist())
    return picks


@pytest.mark.parametrize(
    "rows, routing, codec",
    [
        (32, "random", "bf16"),
        (128, "random", "bf16"),
        (32, "hot", "bf16"),
        (128, "skewed", "bf16"),
        (128, "random", "int4"),
    ],
)
def test_moe(harness, rows, routing, codec):
    """Routing (as sets: bf16 logits tie at the top-k boundary) and weights,
    then the rows whose routing agrees against the reference's MXFP4
    activations; at most a few rows may route differently. ``hot`` (every
    row on the same experts) and ``skewed`` routings run split units as
    intermediate halves; their marks must be set."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        E_ALL,
        EXPERTS,
        HIDDEN,
        RI,
        SLOTS,
        TOPK,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import moe

    gen = torch.Generator().manual_seed(3)
    if routing == "random":
        x = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16)
        w_gate = (torch.randn(E_ALL, HIDDEN, generator=gen) * 0.05).to(torch.bfloat16)
    elif routing == "hot":
        x, w_gate = _routed_inputs(gen, rows, [list(range(TOPK))] * rows)
    else:
        x, w_gate = _routed_inputs(gen, rows, _skewed_picks(rows, gen))
    w13, w13s, w13d = _expert_weights(gen, E_ALL, 2 * RIP, HIDDEN, RI, HIDDEN)
    w2, w2s, w2d = _expert_weights(gen, E_ALL, HIDDEN, RIP, HIDDEN, RI)
    y_r, ids_r, ws_r = ref.moe(x, w_gate, w13d, w2d, EXPERTS, TOPK, RI, RIP)

    d = {
        n: v.cuda()
        for n, v in dict(x=x, w_gate=w_gate, w13=w13, w13s=w13s, w2=w2, w2s=w2s).items()
    }
    part = torch.zeros(moe.router_plan().partial_words(rows), device="cuda")
    route = torch.zeros(2 * rows * SLOTS, dtype=torch.int32, device="cuda")
    x4 = torch.zeros(rows, HIDDEN // 8, dtype=torch.int32, device="cuda")
    x4s = torch.zeros(rows, HIDDEN // 32, dtype=torch.int32, device="cuda")
    mpart = torch.zeros(rows * SLOTS, HIDDEN, device="cuda")
    mpart2 = torch.zeros(rows * SLOTS, HIDDEN, device="cuda")
    flags = torch.zeros(rows * SLOTS, dtype=torch.int32, device="cuda")
    harness.mbox_t.zero_()
    _moe_kernel(rows, codec)(
        *[d[n].data_ptr() for n in ("x", "w_gate", "w13", "w13s", "w2", "w2s")],
        part.data_ptr(),
        route.data_ptr(),
        x4.data_ptr(),
        x4s.data_ptr(),
        mpart.data_ptr(),
        mpart2.data_ptr(),
        flags.data_ptr(),
        harness.mbox,
        harness.epoch.data_ptr(),
        *harness.sync_args(),
        torch.cuda.current_stream(),
    )
    torch.accelerator.synchronize()
    assert harness.health[0].item() == 0
    ids = route[: rows * SLOTS].view(rows, SLOTS).cpu()
    ws = route[rows * SLOTS :].view(torch.float32).view(rows, SLOTS).cpu()
    agree = (torch.sort(ids, 1).values == torch.sort(ids_r.int(), 1).values).all(1)
    assert agree.float().mean().item() >= 0.85
    ws_err = (torch.sort(ws, 1).values - torch.sort(ws_r.float(), 1).values).abs()[
        agree
    ]
    assert ws_err.max().item() < 1e-5
    y = harness.exchange_y(rows, 2, codec).double().cpu()
    err = ((y - y_r).norm(dim=1) / y_r.norm(dim=1))[agree]
    if codec == "int4":
        # the codec's own error: a 16-column group's step is its extremum / 8
        q_err = (ref.int4_roundtrip(y_r) - y_r).norm(dim=1) / y_r.norm(dim=1)
        assert err.median().item() < 1.5 * q_err.median().item() + 5e-3
    else:
        assert err.median().item() < 5e-3
        assert err.max().item() < 5e-2
    tag = harness.epoch[0].item() * 128 + 9
    marked = (flags == tag).sum().item()
    if routing == "random":
        assert marked == 0
    else:
        assert marked > 0
    if routing == "hot":
        assert marked == rows * SLOTS


@pytest.mark.parametrize(
    "level, codec",
    [
        ("INT4", "int4"),
        ("INT3", "int4"),
        ("INT6", "bf16"),
        ("FP", "bf16"),
        ("NONE", "bf16"),
    ],
)
def test_allreduce_codec_follows_quickreduce(monkeypatch, level, codec):
    """The TP exchange quantizes as the deployment's QuickReduce setting asks
    (INT4 lines for INT4 and INT3), and is exact otherwise."""
    from vllm.models.qwen4_exp.amd.specialized_decode.runtime.layers import (
        allreduce_codec,
    )

    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", level)
    assert allreduce_codec() == codec
