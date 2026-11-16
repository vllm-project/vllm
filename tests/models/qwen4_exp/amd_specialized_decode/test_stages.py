# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The wide decode's stages on MI355X against float64 references with the
stock path's rounding points, at the shapes of Qwen3.8-Flash-Next at TP2 and
TP1, one GPU (the TP exchange with one rank): the batch-tiled GEMM on each weight
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
# the step widths each stage is checked at: every width 1-5, the partial
# exchange lines of 12 and 129, and every new capture size through 192
WIDTHS = [1, 2, 3, 4, 5, 8, 12, 16, 32, 128, 129, *range(136, 193, 8)]
SMALL = [w for w in WIDTHS if w <= 16]
CODECS = ["bf16", "int4"]


def tp_cases(tp2, tp1):
    """`pytest.param`s `(tp, *case)`: TP2's cases keep the ids they have
    without `tp`, TP1's are `tp1-<case>`."""
    params = []
    for tp, cases, prefix in ((2, tp2, ""), (1, tp1, "tp1-")):
        for case in cases:
            case = case if isinstance(case, tuple) else (case,)
            params.append(pytest.param(tp, *case, id=prefix + "-".join(map(str, case))))
    return params


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
        """The rank's partial in exchange slot `slot` as [rows, HIDDEN]."""
        return exchange_lines(self.mbox_t, 1, slot, 0, rows, codec)


def exchange_lines(mbox, tp, slot, src, rows, codec):
    """Source rank `src`'s lines in exchange slot `slot` of a mailbox (i32
    words) as [rows, HIDDEN]."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import HIDDEN
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import (
        exchange as ex,
    )

    words, lpg = ex.row_words(), ex.lines_per_group(rows)
    at = (slot * tp + src) * words
    row = mbox[at : at + ex.payload_words(rows, codec)]
    if codec == "int4":
        return ref.int4_lines(row, rows, ex.GROUPS, lpg, HIDDEN)
    y = row.contiguous().view(torch.bfloat16).view(ex.GROUPS, lpg, ex.TOKENS, ex.COLS)
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
        stage: fx.Array[fx.Int32, tiled.stage_words(rows, plan.per), 16]
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
        gemm = tiled.Gemm(sm.stage.ptr, rows, tid, k_total=k, chunks=plan.per)
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


def _check_gemm(harness, rows, n, k, splits, fmt, fold):
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


@pytest.mark.parametrize(
    "rows, fmt",
    [(ROWS, "fp4_triton"), (ROWS, "fp4_asm"), (1, "fp4_asm"), (12, "fp4_asm")]
    + [(w, fmt) for w in (136, 160, 192) for fmt in ("fp4_triton", "fp4_asm")]
    + [(w, "bf16") for w in WIDTHS],
)
@pytest.mark.parametrize("fold", [False, True])
def test_tiled_gemm(harness, rows, fmt, fold):
    """`x @ W.T` of the gated residual's down shape (split K, summed in order
    or folded by the last arriver), bf16 or MXFP4 weights in either AITER
    layout: f32 accumulation of exact bf16 products."""
    _check_gemm(harness, rows, 352, 2560, 4, fmt, fold)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    "n, k, splits, fmt, fold",
    [
        (352, 2560, 2, "bf16", False),
        (256, 2560, 4, "fp4_asm", True),
        (192, 320, 1, "bf16", False),
    ],
)
def test_tiled_gemm_resident(harness, rows, n, k, splits, fmt, fold):
    """The resident stage of widths up to `tiled.RESIDENT_MAX`: a unit's
    whole K slice (10 chunks, 5 chunks, a ragged K of 320) staged once."""
    _check_gemm(harness, rows, n, k, splits, fmt, fold)


def test_fp4_shuffle_matches_aiter():
    """The shuffled layout the kernels read is AITER's `shuffle_weight`."""
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


def _hc_kernel(rows, codec="bf16", combine=True, cold=False):
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

    chunks = max(hc.down_plan(rows).per, hc.UP_CHUNKS)
    aux_words = max(hc.up_lds_words(rows), ex.GroupTile.words(rows))

    @fx.struct
    class Smem:
        stage: fx.Array[fx.Int32, tiled.stage_words(rows, chunks), 16]
        aux: fx.Array[fx.Int32, aux_words, 16]
        word: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(
        name=f"t_hc_{rows}_{codec}_{int(combine)}_{int(cold)}",
        known_block_size=[THREADS, 1, 1],
    )
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
        # past the resident widths the down weights issued during the stats, as
        # the layer kernels do up to ``tiled.PRELOAD_ROWS``
        pre = None
        if fx.const_expr(not tiled.resident(rows)):
            down_w = hc.down_weights(w_di)
            early = tiled.Gemm(sm.stage.ptr, rows, tid, chunks=chunks)
            pre = tiled.unit_preload(early, down_w, hc.down_plan(rows), g.bid)
        hc.stats(g, state, rrms, rows, EPS)
        gemm = tiled.Gemm(sm.stage.ptr, rows, tid, chunks=chunks)
        up = tiled.Gemm(sm.stage.ptr, rows, tid, k_total=LOWRANK, chunks=chunks)
        if fx.const_expr(tiled.resident(rows)):
            # one phase with tagged hand-offs, its weights issued inside the
            # barrier before it (or, ``cold``, in it), as the layer kernels do
            # (``part``, ``down``: the tagged partials and low-rank rows)
            issue = None if cold else (lambda: hc.mix_issue(g, w_di, w_up, rows))
            pending = grid_sync(g, issue)
            tag = fx.Int32(ld_i32x4(epoch, 0)[0]) * 128 + 7
            hc.mix_resident(
                g,
                gemm,
                up,
                sm.aux.ptr,
                state,
                rrms,
                norm_w,
                w_di,
                w_up,
                x_out,
                inj,
                (part, down),
                tag,
                rows,
                pending,
            )
        else:
            grid_sync(g)
            pre_up = hc.UpLoads(g, up, w_up, state, rrms, norm_w, rows)
            hc.mix_down(g, gemm, state, rrms, norm_w, w_di, part, pre)
            grid_sync(g)
            hc.mix_reduce(g, part, down, inj, rows, True)
            grid_sync(g)
            hc.mix_up(g, up, sm.aux.ptr, down, w_up, x_out, rows, pre_up)
        if fx.const_expr(combine):
            grid_sync(g)
            # y through a one-rank exchange (a 64-column group a CTA), then the
            # combine
            tag = fx.Int32(ld_i32x4(epoch, 0)[0]) * 128 + 5
            xch = ex.Exchange([fx.Int64(mbox)], mbox, 0, 1, 1, tag, rows, codec)
            tile = ex.GroupTile(sm.aux.ptr, rows)
            for grp in range(bid, ex.GROUPS, BLOCKS):
                for idx in range(tid, rows * 8, THREADS):
                    token, piece = idx // 8, idx % 8
                    at = (token * HIDDEN + grp * 64 + piece * 8) // 2
                    w = ld_i32x4(y, at, CM_DEV)
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


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("rows", WIDTHS)
def test_gated_residual(harness, rows, codec):
    """The mix's normalized gated mean and injection scales, the combine's new
    state and its rrms; with `int4` exchange lines the combine adds the
    codec's round trip of the block output."""
    _check_gated_residual(harness, rows, codec, True)


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
def test_gated_residual_resident(harness, rows, cold):
    """The one-phase mix of the resident widths, its weights issued ahead or
    (`cold`) in the phase, then the combine."""
    _check_gated_residual(harness, rows, "bf16", True, cold)


def _check_gated_residual(harness, rows, codec, combine, cold=False):
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
    words = max(hc.down_plan(rows).partial_words(rows), hc.mix_words(rows)[0])
    part = torch.zeros(words, device="cuda")
    down = torch.zeros(rows, 2 * LOWRANK, dtype=torch.bfloat16, device="cuda")
    harness.mbox_t.zero_()
    _hc_kernel(rows, codec, combine, cold)(
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
    if combine:
        assert rel(d["state"], new) < 1e-2
        assert rel(rrms2, rrms2_r) < 1e-3


# --------------------------------------------------------------------------
# the TP exchange between two ranks, both mailboxes on this GPU
# --------------------------------------------------------------------------

SENTINEL = 0x5A5A5A5A  # a mailbox word no launch wrote


@pytest.fixture(scope="module")
def mailboxes():
    """Two ranks' mailboxes (uncached, as the peer-mapped ones) on one GPU."""
    from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

    from vllm.models.qwen4_exp.amd.specialized_decode.runtime.peer import (
        _DeviceBytes,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange

    nbytes = exchange.mailbox_bytes(2)
    ptrs = [UncachedIpcHeap.alloc_uncached(nbytes) for _ in range(2)]
    views = [
        torch.as_tensor(_DeviceBytes(p, nbytes), device="cuda").view(torch.int32)
        for p in ptrs
    ]
    return ptrs, views


def _exchange_kernel(rows, codec):
    """Rank `rank` of a two-rank exchange in slot 1: with `push`, its
    partial `y` [rows, HIDDEN] into both mailboxes (a 64-column group a
    CTA, as `gdn.out`'s last arrivers push); with `combine`, the combine
    of both ranks' lines from its own mailbox into `state`."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.typing import Int32, Int64

    default_stream = fx.Stream(None)

    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
        Grid,
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
        THREADS,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import hc

    @fx.struct
    class Smem:
        tile: fx.Array[fx.Int32, ex.GroupTile.words(rows), 16]
        word: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=f"t_xch_{rows}_{codec}", known_block_size=[THREADS, 1, 1])
    def kern(
        y: Int64,
        mbox0: Int64,
        mbox1: Int64,
        state: Int64,
        inj: Int64,
        rrms: Int64,
        epoch: Int64,
        rank: Int32,
        push: Int32,
        combine: Int32,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        sm = fx.SharedAllocator().allocate(Smem).peek()
        g = Grid(tid, bid, s0, s1, folds, health, fx.Int64(0), sm.word.ptr)
        prologue(g)
        bases = [fx.Int64(mbox0), fx.Int64(mbox1)]
        own = (rank == 1).select(bases[1], bases[0])
        tag = fx.Int32(ld_i32x4(epoch, 0)[0]) * 128 + 7
        xch = ex.Exchange(bases, own, rank, 2, 1, tag, rows, codec)
        tile = ex.GroupTile(sm.tile.ptr, rows)
        if push != 0:
            for grp in range(bid, ex.GROUPS, BLOCKS):
                for idx in range(tid, rows * 8, THREADS):
                    token, piece = idx // 8, idx % 8
                    w = ld_i32x4(
                        y, (token * HIDDEN + grp * 64 + piece * 8) // 2, CM_DEV
                    )
                    lds_st(tile.ptr + token * ex.GroupTile.STRIDE + piece * 4, w)
                fx.barrier()
                ex.push_tile(g, xch, tile, grp)
                fx.barrier()
        if combine != 0:
            hc.combine(g, xch, state, inj, rrms, rows, EPS)

    @flyc.jit
    def launch(
        y: Int64,
        mbox0: Int64,
        mbox1: Int64,
        state: Int64,
        inj: Int64,
        rrms: Int64,
        epoch: Int64,
        rank: Int32,
        push: Int32,
        combine: Int32,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
        stream: fx.Stream = default_stream,
    ):
        kern(
            y,
            mbox0,
            mbox1,
            state,
            inj,
            rrms,
            epoch,
            rank,
            push,
            combine,
            s0,
            s1,
            folds,
            health,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("rows", WIDTHS)
def test_exchange_two_ranks(harness, mailboxes, rows, codec):
    """Rank 1 pushes, then rank 0 pushes and combines, then rank 1 combines:
    both ranks' new states are the same bits (each decodes both copies, its
    own included, and adds them in rank order) and match the reference; each
    mailbox holds both ranks' lines and tagged flags, and nothing past a
    partial line's rows, past the step's payload or past its flags."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        HC,
        HC_HIDDEN,
        HIDDEN,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex

    ptrs, views = mailboxes
    gen = torch.Generator().manual_seed(4)
    ys = [torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16) for _ in "01"]
    state = (torch.randn(rows, HC_HIDDEN, generator=gen) * 2).to(torch.bfloat16)
    inj = 0.5 + torch.rand(rows, HC, generator=gen)
    lines = [ref.int4_roundtrip(y) if codec == "int4" else y.float() for y in ys]
    y = ref.bf((lines[0] + lines[1]).double())
    x = state.double().view(rows, HC, HIDDEN)
    new = ref.bf(x + y[:, None, :] * inj[..., None]).view(rows, HC_HIDDEN)

    for v in views:
        v.fill_(SENTINEL)
    yd = [y.cuda() for y in ys]
    states = [state.cuda() for _ in "01"]
    injd = inj.cuda()
    rrms = [torch.zeros(rows, HC, device="cuda") for _ in "01"]
    launch = _exchange_kernel(rows, codec)
    for rank, push, combine in ((1, 1, 0), (0, 1, 1), (1, 0, 1)):
        launch(
            yd[rank].data_ptr(),
            *ptrs,
            states[rank].data_ptr(),
            injd.data_ptr(),
            rrms[rank].data_ptr(),
            harness.epoch.data_ptr(),
            rank,
            push,
            combine,
            *harness.sync_args(),
            torch.cuda.current_stream(),
        )
    torch.accelerator.synchronize()
    assert harness.health[0].item() == 0
    assert torch.equal(states[0], states[1])
    assert torch.equal(rrms[0], rrms[1])
    assert rel(states[0], new) < 1e-2

    tag = harness.epoch[0].item() * 128 + 7
    words, lpg, lw = ex.row_words(), ex.lines_per_group(rows), ex.line_words(codec)
    live = torch.zeros(lpg * ex.TOKENS, dtype=torch.bool)
    live[:rows] = True
    for mbox in views:
        for src in range(2):
            got = exchange_lines(mbox, 2, 1, src, rows, codec).double().cpu()
            if codec == "int4":
                assert rel(got, lines[src]) < 1e-2
            else:
                assert torch.equal(got, ys[src].double())
            row = mbox[(2 + src) * words : (3 + src) * words].cpu()
            payload = row[: ex.payload_words(rows, codec)].view(ex.GROUPS, lpg, lw)
            if codec == "bf16":
                dead = payload.view(ex.GROUPS, lpg * ex.TOKENS, 32)[:, ~live]
            else:
                codes = payload[..., :64].reshape(ex.GROUPS, lpg * ex.TOKENS, 8)
                scales = payload[..., 64:].reshape(ex.GROUPS, lpg * ex.TOKENS, 2)
                dead = torch.cat([codes[:, ~live], scales[:, ~live]], -1)
            assert (dead == SENTINEL).all()
            rest = row[ex.payload_words(rows, codec) : ex.FLAGS_AT]
            assert (rest == SENTINEL).all()
            flags = row[ex.FLAGS_AT :]
            assert (flags[: ex.flag_words(rows)] == tag).all()
            assert (flags[ex.flag_words(rows) :] == SENTINEL).all()


# --------------------------------------------------------------------------
# the gated delta net: in_proj + conv, recurrence + gated norm, out_proj
# --------------------------------------------------------------------------


def _gdn_kernel(sh, rows, out=True):
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
        HIDDEN,
        THREADS,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import (
        gdn,
        tagged,
        tiled,
    )

    chunks = max(gdn.proj_plan(sh).per, gdn.out_plan(sh.VALUE_DIM).per)

    @fx.struct
    class Smem:
        stage: fx.Array[fx.Int32, tiled.stage_words(rows, chunks), 16]
        tile: fx.Array[fx.Int32, ex.GroupTile.words(rows), 16]
        red: fx.Array[fx.Float32, gdn.recur_lds_words(), 16]
        word: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(
        name=f"t_gdn_tp{sh.TP}_{rows}_{int(out)}", known_block_size=[THREADS, 1, 1]
    )
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
            **sh._asdict(),
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
            **sh._asdict(),
            x=x,
            qkv=qkv,
            z=z,
            ba=ba,
            core=core,
            part=part,
            st_idx=st_idx,
            conv_state=conv_state,
            ssm=ssm,
            conv_stride=fx.Int64(3 * sh.CONV_DIM),
            ssm_stride=fx.Int64(sh.NV * 128 * 128),
        )
        # at the resident widths each GEMM's first weights are issued ahead,
        # the out_proj's inside the barrier before it, and the recurrence
        # polls the projection's tagged outputs, as the layer kernels do
        pre = tiled.resident(rows)
        gemm = tiled.Gemm(sm.stage.ptr, rows, tid, chunks=chunks)
        weights = tiled.Fp4Weights(wo, wo_scale, HIDDEN, sh.VALUE_DIM, False)
        if fx.const_expr(pre):
            gtag = fx.Int32(ld_i32x4(epoch, 0)[0]) * 128 + 9
            gdn.proj(g, gemm, w, b, rows, 0, gdn.proj_issue(g, w), gtag)
            gdn.recur(g, w, b, sm.red.ptr, rows, EPS, None, gtag)
            out_w = tiled.issue_unit(bid, tid, weights, gdn.out_plan(sh.VALUE_DIM))
            source = tagged.Pairs(g, core, sh.VALUE_DIM, gtag)
        else:
            # each GEMM's first weights preloaded, out_proj's during the
            # recurrence, as the layer kernels do up to ``tiled.PRELOAD_ROWS``
            proj_w = tiled.unit_preload(
                gemm, gdn.proj_weights(w), gdn.proj_plan(sh), g.bid
            )
            gdn.proj(g, gemm, w, b, rows, 0, proj_w)
            grid_sync(g)
            out_w = None
            if fx.const_expr(out):
                out_w = tiled.unit_preload(
                    gemm, weights, gdn.out_plan(sh.VALUE_DIM), g.bid
                )
            gdn.recur(g, w, b, sm.red.ptr, rows, EPS)
            source = tiled.Source(core, sh.VALUE_DIM)
            if fx.const_expr(out):
                grid_sync(g)
        if fx.const_expr(out):
            tag = fx.Int32(ld_i32x4(epoch, 0)[0]) * 128 + 3
            xch = ex.Exchange([fx.Int64(mbox)], mbox, 0, 1, 0, tag, rows)
            gdn.out(
                g,
                gemm,
                weights,
                source,
                part,
                ex.GroupTile(sm.tile.ptr, rows),
                xch,
                sh.VALUE_DIM,
                max(200, gdn.proj_plan(sh).groups),
                out_w,
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


@pytest.mark.parametrize("tp, rows", tp_cases(WIDTHS, (32, 64)))
def test_gated_delta_net(harness, tp, rows):
    """Conv outputs and states, z, the recurrent states, the gated-norm core
    and out_proj through the exchange; padding rows (slot 0: rows 3 and the
    last, the last alone below five rows, none at one row) leave the states
    alone and give a zero core."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import shard

    pads = sorted({min(3, rows - 1), rows - 1}) if rows > 1 else []
    _check_gated_delta_net(harness, shard(tp), rows, pads, True)


@pytest.mark.parametrize("tp, rows", tp_cases([1, 2, 4, 8, 16], (1, 16)))
def test_gated_delta_net_resident(harness, tp, rows):
    """The gated delta net at the resident widths, out_proj through the
    exchange; the last row of two or more pads."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import shard

    _check_gated_delta_net(
        harness, shard(tp), rows, [rows - 1] if rows > 1 else [], True
    )


def _check_gated_delta_net(harness, sh, rows, pads, out):
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import HIDDEN
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import gdn

    gen = torch.Generator().manual_seed(2)
    slots = rows + 9
    x = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16)
    wq_data, wq_scale, wq = mxfp4(sh.QKVZ, HIDDEN, gen, 115, 121)
    wba = (torch.randn(sh.BA, HIDDEN, generator=gen) * 0.02).to(torch.bfloat16)
    wconv = (torch.randn(sh.CONV_DIM, 1, 4, generator=gen) * 0.5).to(torch.bfloat16)
    a_log = torch.randn(sh.NV, generator=gen) * 0.5
    dt_bias = (torch.randn(sh.NV, generator=gen) * 0.5).to(torch.bfloat16)
    normw = (1 + torch.randn(128, generator=gen) * 0.1).to(torch.bfloat16)
    wo_data, wo_scale, wo = mxfp4(HIDDEN, sh.VALUE_DIM, gen, 116, 121)
    st_idx = (torch.randperm(slots - 1, generator=gen)[:rows] + 1).to(torch.int32)
    st_idx[pads] = 0
    conv_state = torch.randn(slots, 3, sh.CONV_DIM, generator=gen).to(torch.bfloat16)
    ssm = torch.randn(slots, sh.NV, 128, 128, generator=gen) * 0.05
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
    qkv_words, ba_words, core_words = gdn.handoff_words(sh, rows)
    qkv = torch.zeros(qkv_words, dtype=torch.int32, device="cuda")
    z = torch.zeros(rows, sh.VALUE_DIM, dtype=torch.bfloat16, device="cuda")
    ba = torch.zeros(ba_words, device="cuda")
    core = torch.zeros(core_words, dtype=torch.int32, device="cuda")
    part = torch.zeros(
        max(
            gdn.proj_plan(sh).partial_words(rows),
            gdn.out_plan(sh.VALUE_DIM).partial_words(rows),
        ),
        device="cuda",
    )
    harness.mbox_t.zero_()
    _gdn_kernel(sh, rows, out)(
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
    if qkv_words == rows * sh.QKVZ:  # tagged (bf16 pair, tag) pairs, z included
        qkvz = qkv.view(rows, sh.QKVZ // 2, 2)[:, :, 0].contiguous()
        qkvz = qkvz.view(torch.bfloat16).view(rows, sh.QKVZ)
        qkv, z = qkvz[:, : sh.CONV_DIM], qkvz[:, sh.CONV_DIM :]
    else:
        qkv = qkv.view(torch.bfloat16).view(rows, sh.CONV_DIM)
    if core_words == rows * sh.VALUE_DIM:  # tagged (bf16 pair, tag) pairs
        core = core.view(rows, sh.VALUE_DIM // 2, 2)[:, :, 0].contiguous()
    core = core.view(torch.bfloat16).view(rows, sh.VALUE_DIM)
    assert rel(qkv, conv_r) < 1e-2
    assert rel(z, z_r) < 1e-2
    assert rel(d["conv_state"], cs_r) < 1e-2
    assert rel(d["ssm"], ssm_r) < 1e-2
    assert rel(core, core_r) < 1e-2
    for p in pads:
        assert core[p].abs().max().item() == 0
    if out:
        assert rel(harness.exchange_y(rows, 0), y_r) < 2e-2


# --------------------------------------------------------------------------
# the MoE: router, routing + activation quant, experts in place, combine
# --------------------------------------------------------------------------

# the experts' padded intermediate a rank, by TP (real: 320 at TP2, 640 at TP1)
RIP = {1: 640, 2: 384}


def _moe_kernel(sh, rows, codec="bf16"):
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
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.plan import (
        source_digest,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import BLOCKS, THREADS
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import moe, tiled

    digest = source_digest()  # FlyDSL does not hash the stages' sources
    rip = RIP[sh.TP]
    shape = moe.Shape(rows, rip, sh.RI)
    MoeLds = moe.lds_struct(shape)

    @fx.struct
    class GemmLds:
        stage: fx.Array[fx.Int32, tiled.stage_words(rows), 16]

    @fx.union
    class PhaseLds:
        gemm: GemmLds
        moe: MoeLds  # type: ignore[valid-type]

    @fx.struct
    class Smem:
        word: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(
        name=f"t_moe_tp{sh.TP}_{rows}_{codec}", known_block_size=[THREADS, 1, 1]
    )
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
        _ = (digest, sh, rip)
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        alloc = fx.SharedAllocator()
        sm = alloc.allocate(Smem).peek()
        ph = alloc.allocate(PhaseLds)
        g = Grid(tid, bid, s0, s1, folds, health, fx.Int64(0), sm.word.ptr)
        prologue(g)
        tag = fx.Int32(ld_i32x4(epoch, 0)[0]) * 128 + 9
        gemm = tiled.Gemm(ph.gemm.peek().stage.ptr, rows, tid)
        # past the resident widths the router's weights issued ahead, as the
        # layer kernels do up to ``tiled.PRELOAD_ROWS``
        pre = None
        if fx.const_expr(not tiled.resident(rows)):
            router_w = moe.router_weights(w_gate)
            pre = tiled.unit_preload(gemm, router_w, moe.router_plan(), g.bid)
        moe.mlp(
            g,
            shape,
            gemm,
            ph.moe.peek(),
            x,
            moe.Weights(w_gate, w13, w13s, w2, w2s),
            moe.Bufs(part, route, x4, x4s, mpart, mpart2, flags),
            ex.Exchange([fx.Int64(mbox)], mbox, 0, 1, 2, tag, rows, codec),
            lambda name, between=None: grid_sync(g, between),
            pre,
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
        _ = (digest, sh, rip)
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


def _routed_inputs(gen, rows, picks):
    """`x` and a router gate whose top-k of row `b` is `picks[b]`: each
    expert has a near-orthogonal direction `u_e`; a row is the sum of its
    experts' directions (slightly unequal) plus noise, the gate rows `2
    u_e`."""
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


def _split_assignments(ids):
    """The (row, slot) assignments of a step's split units (`moe.pieces`):
    units are (expert, <= 16 rows) in expert order, and a last round at most
    half full (every unit below 129 units, so every row of a narrow step)
    runs as two halves a unit."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        BLOCKS,
        E_ALL,
    )

    counts = torch.bincount(ids.flatten().long(), minlength=E_ALL).tolist()
    sizes = [min(16, c - k) for c in counts for k in range(0, c, 16)]
    r = len(sizes) % BLOCKS
    split = r if 0 < r <= BLOCKS // 2 else 0
    return sum(sizes[len(sizes) - split :])


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
    "tp, rows, routing, codec",
    tp_cases(
        [
            (32, "random", "bf16"),
            (128, "random", "bf16"),
            (32, "hot", "bf16"),
            (128, "skewed", "bf16"),
            (128, "random", "int4"),
            (192, "hot", "bf16"),
            (192, "skewed", "bf16"),
        ]
        + [(w, "random", c) for w in (129, 136, 160, 192) for c in CODECS]
        + [(w, "random", c) for w in SMALL for c in CODECS]
        + [(w, "hot", "bf16") for w in (1, 4, 12)],
        [(32, "hot", "bf16"), (64, "skewed", "bf16")],
    ),
)
def test_moe(harness, tp, rows, routing, codec):
    """Routing (as sets: bf16 logits tie at the top-k boundary) and weights,
    then the rows whose routing agrees against the reference's MXFP4
    activations; at most a few rows may route differently. Narrow `hot`
    (every row on the same experts) and `skewed` (from 128 rows) routings run
    split units as intermediate halves; 192 hot rows fill over half a grid, so
    do not."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        E_ALL,
        EXPERTS,
        HIDDEN,
        SLOTS,
        TOPK,
        shard,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import moe

    sh = shard(tp)
    gen = torch.Generator().manual_seed(3)
    if routing == "random":
        x = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16)
        w_gate = (torch.randn(E_ALL, HIDDEN, generator=gen) * 0.05).to(torch.bfloat16)
    elif routing == "hot":
        x, w_gate = _routed_inputs(gen, rows, [list(range(TOPK))] * rows)
    else:
        x, w_gate = _routed_inputs(gen, rows, _skewed_picks(rows, gen))
    rip = RIP[sh.TP]
    w13, w13s, w13d = _lazy_expert_weights(gen, E_ALL, 2 * rip, HIDDEN, sh.RI, HIDDEN)
    w2, w2s, w2d = _lazy_expert_weights(gen, E_ALL, HIDDEN, rip, HIDDEN, sh.RI)
    y_r, ids_r, ws_r = ref.moe(x, w_gate, w13d, w2d, EXPERTS, TOPK, sh.RI, rip)

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
    _moe_kernel(sh, rows, codec)(
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
    if moe.small(rows):  # the small-batch path splits no units (test_moe_small)
        assert marked == 0
        return
    assert marked == _split_assignments(ids)
    if routing == "random" and rows >= 32:
        assert marked == 0
    if routing == "skewed" and rows >= 128:  # below, its units are whole
        assert marked > 0
    if routing == "hot":
        assert marked == (0 if rows == 192 else rows * SLOTS)


class _LazyDequant:
    """`w[e]`: expert `e` of e2m1 `codes` / E8M0 `scales` dequantized to
    f32 on first use (a test routes to few of the 513 experts)."""

    def __init__(self, codes, scales, k):
        self.codes, self.scales, self.k, self.cache = codes, scales, k, {}

    def __getitem__(self, e):
        if e not in self.cache:
            sc = self.scales[e][:, : self.k // 32].float()
            self.cache[e] = ref.E2M1[self.codes[e].long()] * torch.pow(
                2.0, sc - 127
            ).repeat_interleave(32, -1)
        return self.cache[e]


def _lazy_expert_weights(gen, experts, rows, k, rows_real, k_real):
    """MXFP4 experts [experts, rows, k] in AITER's shuffled layouts, zero past
    `rows_real` rows of each half (`rows` covers two) and past `k_real`:
    (data, scales [experts, rows, pad8(k / 32)], dequantized f32 per expert on
    use)."""
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
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return (
        shuffle_data(packed),
        shuffle_scale(sc.view(experts * rows, cols)).view(experts, rows, cols),
        _LazyDequant(codes, sc, k),
    )


@pytest.fixture(scope="module")
def small_moe_weights():
    """`weights(tp)`: the experts at `tp` on the device, with their
    dequantized references, built on first use."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        E_ALL,
        HIDDEN,
        shard,
    )

    built = {}

    def weights(tp):
        if tp not in built:
            sh, rip = shard(tp), RIP[tp]
            gen = torch.Generator().manual_seed(5)
            w13, w13s, w13d = _lazy_expert_weights(
                gen, E_ALL, 2 * rip, HIDDEN, sh.RI, HIDDEN
            )
            w2, w2s, w2d = _lazy_expert_weights(gen, E_ALL, HIDDEN, rip, HIDDEN, sh.RI)
            dev = {
                n: v.cuda() for n, v in dict(w13=w13, w13s=w13s, w2=w2, w2s=w2s).items()
            }
            built[tp] = dev, w13d, w2d
        return built[tp]

    return weights


@pytest.mark.parametrize(
    "tp, rows, routing, codec",
    tp_cases(
        [
            (1, "picked", "bf16"),
            (2, "picked", "bf16"),
            (3, "picked", "bf16"),
            (4, "picked", "bf16"),
            (5, "picked", "bf16"),
            (7, "shared", "bf16"),
            (8, "picked", "bf16"),
            (8, "random", "bf16"),
            (8, "hot", "bf16"),
            (8, "shared", "bf16"),
            (8, "inactive", "bf16"),
            (8, "picked", "int4"),
            (12, "picked", "bf16"),
            (12, "inactive", "bf16"),
        ],
        [(1, "picked", "bf16"), (12, "picked", "bf16")],
    ),
)
def test_moe_small(harness, small_moe_weights, tp, rows, routing, codec):
    """The small-batch MoE path (rows <= `moe.SMALL_MAX`: GEMV router, local
    routing, gate/up pieces, down by output owners into the exchange) against
    the reference, as `test_moe`. `picked`: every row its own 10 experts
    with clear logit margins; `hot`: every row the same 10; `shared`: rows
    drawing 10 of 16 experts; `inactive`: the last three rows zero (padding),
    which route to experts 0..9 (ties go to the lower expert) with equal
    weights and give exactly zero. `int4`: the exchange's INT4 lines."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
        E_ALL,
        EXPERTS,
        HIDDEN,
        SLOTS,
        TOPK,
        shard,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import moe

    sh = shard(tp)
    assert moe.small(rows)
    dev, w13d, w2d = small_moe_weights(tp)
    gen = torch.Generator().manual_seed(rows)
    idle = 3 if routing == "inactive" else 0
    if routing == "random":
        x = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16)
        w_gate = (torch.randn(E_ALL, HIDDEN, generator=gen) * 0.05).to(torch.bfloat16)
    else:
        pool = torch.randperm(EXPERTS, generator=gen)
        if routing == "hot":
            picks = [pool[:TOPK].tolist()] * rows
        elif routing == "shared":
            picks = [
                pool[:16][torch.randperm(16, generator=gen)[:TOPK]].tolist()
                for _ in range(rows)
            ]
        else:
            picks = [
                torch.randperm(EXPERTS, generator=gen)[:TOPK].tolist()
                for _ in range(rows)
            ]
        x, w_gate = _routed_inputs(gen, rows, picks)
        if idle:
            x[rows - idle :] = 0
    y_r, ids_r, ws_r = ref.moe(x, w_gate, w13d, w2d, EXPERTS, TOPK, sh.RI, RIP[sh.TP])

    part = torch.zeros(moe.router_plan().partial_words(rows), device="cuda")
    route = torch.zeros(2 * rows * SLOTS, dtype=torch.int32, device="cuda")
    x4 = torch.zeros(rows, HIDDEN // 8, dtype=torch.int32, device="cuda")
    x4s = torch.zeros(rows, HIDDEN // 32, dtype=torch.int32, device="cuda")
    mpart = torch.zeros(rows * SLOTS, HIDDEN, device="cuda")
    mpart2 = torch.zeros(rows * SLOTS, HIDDEN, device="cuda")
    flags = torch.zeros(rows * SLOTS, dtype=torch.int32, device="cuda")
    harness.mbox_t.zero_()
    _moe_kernel(sh, rows, codec)(
        x.cuda().data_ptr(),
        w_gate.cuda().data_ptr(),
        *[dev[n].data_ptr() for n in ("w13", "w13s", "w2", "w2s")],
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
    y = harness.exchange_y(rows, 2, codec).double().cpu()
    live = rows - idle
    if idle:
        expect = list(range(TOPK)) + [EXPERTS]
        assert (ids[live:] == torch.tensor(expect, dtype=torch.int32)).all()
        assert (ws[live:, :TOPK] == ws[live:, :1]).all()
        assert (ws[live:, :TOPK].sum(1) - 1).abs().max().item() < 1e-6
        assert (ws[live:, TOPK] == 0.5).all()
        assert (y[live:] == 0).all()
    ids, ws, y, ids_r, ws_r, y_r = (t[:live] for t in (ids, ws, y, ids_r, ws_r, y_r))
    agree = (torch.sort(ids, 1).values == torch.sort(ids_r.int(), 1).values).all(1)
    if routing == "random":
        assert agree.float().mean().item() >= 0.85
    else:
        assert agree.all()
    ws_err = (torch.sort(ws, 1).values - torch.sort(ws_r.float(), 1).values).abs()[
        agree
    ]
    assert ws_err.max().item() < 1e-5
    err = ((y - y_r).norm(dim=1) / y_r.norm(dim=1))[agree]
    if codec == "int4":
        # the codec's own error, as test_moe
        q_err = (ref.int4_roundtrip(y_r) - y_r).norm(dim=1) / y_r.norm(dim=1)
        assert err.median().item() < 1.5 * q_err.median().item() + 5e-3
    else:
        assert err.median().item() < 5e-3
        assert err.max().item() < 5e-2
    assert (flags != 0).sum().item() == 0  # no split marks on this path


def test_decode_widths_take_every_capture_size():
    """Every CUDA graph capture size in 1-192 at TP2, 1-64 at TP1, gets a
    build (vLLM pads a step to the smallest captured width that holds it),
    and a build of any width in that range is valid; a TP the kernels are not
    built for has none."""
    from vllm.config import CompilationConfig, VllmConfig
    from vllm.models.qwen4_exp.amd.specialized_decode.runtime import decode_widths
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layer import LayerBuild

    widths = (1, 2, 3, 4, 8, 12, 16, 24, 32, 64, 128, 129, 136, 160, 184, 192)
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        cudagraph_capture_sizes=[*widths, 193, 200, 256]
    )
    for tp, cap in ((2, 192), (1, 64)):
        config.parallel_config.tensor_parallel_size = tp
        assert decode_widths(config) == tuple(w for w in widths if w <= cap)
        for rows in range(1, cap + 1):
            LayerBuild(kind="gdn", tokens=rows, tp=tp)
        for rows in (0, cap + 1):
            with pytest.raises(ValueError):
                LayerBuild(kind="gdn", tokens=rows, tp=tp)
    config.parallel_config.tensor_parallel_size = 4
    assert decode_widths(config) == ()
    with pytest.raises(ValueError):
        LayerBuild(kind="gdn", tokens=1, tp=3)


def test_unsupported_tp_is_refused_with_every_reason():
    """TP4 under `--monokernel` gets the aggregated config-time refusal (the
    TP and the other reasons, here expert parallelism), not `shard`'s
    ValueError nor a capture-size reason (it has no cap)."""
    from types import SimpleNamespace

    from vllm.config import VllmConfig
    from vllm.models.qwen4_exp.amd.specialized_decode.runtime import (
        specialized_decode_config,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide import layout as L

    config = VllmConfig()
    config.kernel_config.monokernel = True
    config.parallel_config.tensor_parallel_size = 4
    config.parallel_config.enable_expert_parallel = True
    config.model_config = SimpleNamespace(
        enable_sleep_mode=False,
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(**dict(zip(L.SHAPE_NAMES, L.SHAPES))),
    )
    with pytest.raises(ValueError) as refusal:
        specialized_decode_config(config)
    message = str(refusal.value)
    assert "TP 4 (TP1 or TP2 only)" in message
    assert "expert parallelism" in message
    assert "no CUDA graph capture size" not in message


@pytest.mark.parametrize(
    "tp, level, codec",
    tp_cases(
        [
            ("INT4", "int4"),
            ("INT3", "int4"),
            ("INT6", "bf16"),
            ("FP", "bf16"),
            ("NONE", "bf16"),
        ],
        [("INT4", "bf16")],
    ),
)
def test_allreduce_codec_follows_quickreduce(monkeypatch, tp, level, codec):
    """The TP exchange quantizes as the deployment's QuickReduce setting asks
    (INT4 lines for INT4 and INT3), and is exact otherwise; at TP1 nothing is
    reduced, so it is exact whatever the setting."""
    from vllm.models.qwen4_exp.amd.specialized_decode.runtime.layers import (
        allreduce_codec,
    )

    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", level)
    assert allreduce_codec(tp) == codec


@pytest.mark.parametrize("rows", WIDTHS)
def test_qsa_sparse_attention(rows):
    """`wide.attention` over a strided paged K | V pool against float64
    softmax attention of each row's tokens: the selected compressed blocks
    (all of a row's complete blocks in any order up to 512, else 512 of them)
    and the open tail, from no context to past the 2048-token budget, with a
    hole (-1) in a list and (from two rows) a padded row (request -1, zero
    output)."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide import attention

    gen = torch.Generator().manual_seed(rows)
    page, heads, hd, top, ratio = 96, 12, 256, 512, 4
    contexts = [[5001, 0, 1, 37, 2048, 2051, 3003][r % 7] for r in range(rows)]
    pages_per_row = -(-max(contexts) // page)
    blocks = rows * pages_per_row + 1
    stride = page * 2 * hd + 4096  # pages apart in the pool, as vLLM's hybrid pool
    pool = torch.randn(blocks * stride, generator=gen).to(torch.bfloat16)
    kv = pool.as_strided((blocks, 1, page, 2 * hd), (stride, page * 2 * hd, 2 * hd, 1))
    table = (
        torch.randperm(blocks - 1, generator=gen)[: rows * pages_per_row] + 1
    ).view(rows, -1)
    t2r = torch.arange(rows, dtype=torch.int32)
    pad = 1 if rows > 1 else None
    if pad is not None:
        t2r[pad] = -1
    sel = torch.full((rows, top), -1, dtype=torch.int32)
    for r, c in enumerate(contexts):
        sel[r, : min(c // ratio, top)] = torch.randperm(c // ratio, generator=gen)[
            :top
        ].int()
    sel[4 if rows > 4 else 0, 3] = -1  # a hole
    q = (torch.randn(rows, heads, hd, generator=gen) * 0.5).to(torch.bfloat16)

    expect = torch.zeros(rows, heads, hd, dtype=torch.float64)
    for r, c in enumerate(contexts):
        chosen = sel[r][sel[r] >= 0].long()
        t = torch.cat(
            [
                (chosen[:, None] * ratio + torch.arange(ratio)).flatten(),
                torch.arange(c // ratio * ratio, c),
            ]
        )
        if t2r[r] < 0 or not t.numel():
            continue
        rec = kv[table[r][t // page], 0, t % page].double()  # [tokens, 512]
        k, v = rec[:, :hd], rec[:, hd:]
        p = torch.softmax(q[r].double() @ k.T / hd**0.5, dim=-1)
        expect[r] = p @ v

    dev = torch.device("cuda")
    out = torch.empty(rows, heads, hd, dtype=torch.bfloat16, device=dev)
    attention.sparse_attention(
        q.to(dev),
        pool.to(dev).as_strided(kv.shape, kv.stride()),
        sel.to(dev),
        table.int().to(dev).contiguous(),
        t2r.to(dev),
        torch.tensor(contexts, dtype=torch.int32, device=dev),
        ratio,
        out,
    )
    if pad is not None:
        assert out[pad].abs().max().item() == 0
    assert rel(out.float(), expect) < 1e-2


@pytest.mark.parametrize("rows", WIDTHS)
def test_qsa_indexer_scores(rows):
    """`wide.indexer.score` over a strided paged compressed-key pool
    against float64 `sum_h relu(q_h . k_c) / sqrt(128)` on each row's
    visible columns, and the visible counts exactly (a padded row, from three
    rows, sees none)."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide import indexer

    gen = torch.Generator().manual_seed(rows)
    page, width, heads, dim, ratio = 64, 40, 4, 128, 4
    contexts = [[9001, 0, 3, 37, 2048, 4095, 6000][r % 7] for r in range(rows)]
    stride = page * dim + 1024
    pool_pages = rows * width + 1
    pool = torch.randn(pool_pages * stride, generator=gen).to(torch.bfloat16)
    kc = pool.as_strided((pool_pages, page, 1, dim), (stride, dim, dim, 1))
    table = (
        (torch.randperm(pool_pages - 1, generator=gen) + 1)[: rows * width]
        .view(rows, width)
        .int()
    )
    t2r = torch.arange(rows, dtype=torch.int32)
    if rows > 2:
        t2r[2] = -1
    q = torch.randn(rows, heads, dim, generator=gen).to(torch.bfloat16)
    seq = torch.tensor(contexts, dtype=torch.int32)
    positions = (seq - 1).clamp(min=0).long()

    dev = torch.device("cuda")
    columns = width * page
    logits = torch.full((rows, columns), float("nan"), device=dev)
    visible = torch.full((rows,), -7, dtype=torch.int32, device=dev)
    indexer.score(
        q.to(dev),
        pool.to(dev).as_strided(kc.shape, kc.stride()),
        table.to(dev),
        t2r.to(dev),
        positions.to(dev),
        seq.to(dev),
        ratio,
        logits,
        visible,
    )
    visible, logits = visible.cpu(), logits.cpu()
    for r, c in enumerate(contexts):
        want = 0 if t2r[r] < 0 else min(c // ratio, columns)
        assert visible[r].item() == want
        if not want:
            continue
        cols = torch.arange(want)
        keys = kc[table[r][cols // page].long(), cols % page, 0].double()  # [want, dim]
        expect = (q[r].double() @ keys.T).clamp(min=0).sum(0) / dim**0.5
        assert rel(logits[r, :want], expect) < 1e-5


@pytest.mark.parametrize("rows", [1, 4, 8, 16, 128, 136, 192])
def test_qsa_select(rows):
    """`wide.indexer.score` with the in-kernel top-k (`selected`) against
    vLLM's `top_k_per_row_decode` on the same scores, compared as sets: a
    row within the budget lists all of its blocks in order (-1 after) without
    scoring; a longer row picks stock's 512 but for ties at the boundary
    (stock's pick among ties is arbitrary; ours takes the lowest blocks), in
    ascending order. Rows from no context to the most keys the select stages
    in LDS (`select.MAX_KEYS`) and past them, and a padded row."""
    from vllm import _custom_ops as ops
    from vllm.models.qwen4_exp.amd.specialized_decode.wide import indexer, select

    gen = torch.Generator().manual_seed(rows)
    page, heads, dim, ratio, top = 128, 4, 128, 4, 512
    base = [9001, 0, 3, 2048, 2051, 2120, 37, 6000, 33000, 2600]
    contexts = [base[r % len(base)] for r in range(rows)]
    if rows >= 4:
        # the most LDS-staged keys (page 128: the page table leaves the LDS
        # room for them), then past them: the select re-reads the scores
        contexts[rows // 2] = select.MAX_KEYS * ratio
        contexts[rows // 2 + 1] = (select.MAX_KEYS + 300) * ratio
    width = -(-max(contexts) // page)
    need = [-(-c // page) for c in contexts]
    stride = page * dim + 1024
    pool_pages = sum(need) + 1
    pool = torch.randn(pool_pages * stride, generator=gen).to(torch.bfloat16)
    kc = pool.as_strided((pool_pages, page, 1, dim), (stride, dim, dim, 1))
    order = (torch.randperm(pool_pages - 1, generator=gen) + 1).int()
    table = torch.full((rows, width), -1, dtype=torch.int32)
    at = 0
    for r, n in enumerate(need):
        table[r, :n] = order[at : at + n]
        at += n
    t2r = torch.arange(rows, dtype=torch.int32)
    if rows > 1:
        t2r[1] = -1
    q = torch.randn(rows, heads, dim, generator=gen).to(torch.bfloat16)
    seq = torch.tensor(contexts, dtype=torch.int32)
    positions = (seq - 1).clamp(min=0).long()

    dev = torch.device("cuda")
    args = (
        q.to(dev),
        pool.to(dev).as_strided(kc.shape, kc.stride()),
        table.to(dev),
        t2r.to(dev),
        positions.to(dev),
        seq.to(dev),
        ratio,
    )
    columns = width * page
    logits = torch.zeros(rows, columns, device=dev)
    visible = torch.empty(rows, dtype=torch.int32, device=dev)
    indexer.score(*args, logits, visible)
    stock = torch.empty(rows, top, dtype=torch.int32, device=dev)
    ops.top_k_per_row_decode(
        logits, 1, visible, stock, rows, logits.stride(0), logits.stride(1), top
    )
    mine_logits = torch.zeros(rows, columns, device=dev)
    mine_visible = torch.empty(rows, dtype=torch.int32, device=dev)
    mine = torch.full((rows, top), -7, dtype=torch.int32, device=dev)
    for _ in range(2):  # the arrival counters reset themselves
        indexer.score(*args, mine_logits, mine_visible, selected=mine)
    assert torch.equal(mine_visible, visible)
    logits, stock, mine, visible = logits.cpu(), stock.cpu(), mine.cpu(), visible.cpu()
    for r in range(rows):
        v = visible[r].item()
        got = mine[r][mine[r] >= 0]
        assert torch.equal(got, got.sort().values), r
        assert (mine[r] >= 0).sum().item() == min(v, top), r
        assert set(stock[r][stock[r] >= 0].tolist()) <= set(range(v)), r
        if v <= top:
            assert torch.equal(mine[r][:v], torch.arange(v, dtype=torch.int32)), r
            assert (mine[r][v:] == -1).all(), r
            assert set(stock[r][stock[r] >= 0].tolist()) == set(range(v)), r
            continue
        scores = logits[r, :v]
        ours, theirs = set(got.tolist()), set(stock[r].tolist())
        boundary = scores[got.long()].min().item()
        assert scores[stock[r].long()].min().item() == boundary, r
        # every score above the boundary is in both; they differ only in ties
        above = set(torch.nonzero(scores > boundary).flatten().tolist())
        assert above <= ours and above <= theirs, r
        assert all(scores[i].item() == boundary for i in ours ^ theirs), r
        ties = torch.nonzero(scores == boundary).flatten()
        assert torch.equal(
            got[scores[got.long()] == boundary], ties[: top - len(above)].int()
        ), r


@pytest.mark.parametrize("spread", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 16, 32, 128, 192])
def test_qsa_sparse_attention_dense(rows, spread):
    """`wide.attention` with `dense`: a row within the budget attends all
    of its complete blocks and the tail whatever its list holds; a longer row
    reads its list as before. `spread`: every split merges a slice of the
    output. Twice (the arrival counters reset themselves), against float64
    softmax attention."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide import attention

    gen = torch.Generator().manual_seed(100 + rows)
    page, heads, hd, top, ratio = 96, 12, 256, 512, 4
    contexts = [[1100, 0, 1, 37, 2048, 2051, 3003, 5001][r % 8] for r in range(rows)]
    pages_per_row = -(-max(contexts) // page)
    blocks = rows * pages_per_row + 1
    stride = page * 2 * hd + 4096
    pool = torch.randn(blocks * stride, generator=gen).to(torch.bfloat16)
    kv = pool.as_strided((blocks, 1, page, 2 * hd), (stride, page * 2 * hd, 2 * hd, 1))
    table = (
        torch.randperm(blocks - 1, generator=gen)[: rows * pages_per_row] + 1
    ).view(rows, -1)
    t2r = torch.arange(rows, dtype=torch.int32)
    if rows > 1:
        t2r[1] = -1
    sel = torch.full((rows, top), -3, dtype=torch.int32)  # unread for short rows
    for r, c in enumerate(contexts):
        if c // ratio > top:
            sel[r] = torch.randperm(c // ratio, generator=gen)[:top].int()
    if rows > 7:
        sel[7, 3] = -1  # a hole in a long row's list
    q = (torch.randn(rows, heads, hd, generator=gen) * 0.5).to(torch.bfloat16)

    expect = torch.zeros(rows, heads, hd, dtype=torch.float64)
    for r, c in enumerate(contexts):
        if c // ratio > top:
            chosen = sel[r][sel[r] >= 0].long()
        else:
            chosen = torch.arange(c // ratio)
        t = torch.cat(
            [
                (chosen[:, None] * ratio + torch.arange(ratio)).flatten(),
                torch.arange(c // ratio * ratio, c),
            ]
        )
        if t2r[r] < 0 or not t.numel():
            continue
        rec = kv[table[r][t // page], 0, t % page].double()
        k, v = rec[:, :hd], rec[:, hd:]
        p = torch.softmax(q[r].double() @ k.T / hd**0.5, dim=-1)
        expect[r] = p @ v

    dev = torch.device("cuda")
    args = (
        q.to(dev),
        pool.to(dev).as_strided(kv.shape, kv.stride()),
        sel.to(dev),
        table.int().to(dev).contiguous(),
        t2r.to(dev),
        torch.tensor(contexts, dtype=torch.int32, device=dev),
        ratio,
    )
    for _ in range(2):
        out = torch.full((rows, heads, hd), 3.0, dtype=torch.bfloat16, device=dev)
        attention.sparse_attention(*args, out, dense=True, spread=spread)
        if rows > 1:
            assert out[1].abs().max().item() == 0
        assert rel(out.float(), expect) < 1e-2


def _qsa_proj_kernel(sh, rows):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.typing import Int64

    default_stream = fx.Stream(None)

    from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
        Grid,
        prologue,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import BLOCKS, THREADS
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import qsa, tiled

    chunks = max(qsa.qkv_plan(sh).per, qsa.ix_plan().per)  # a resident unit's K slice

    @fx.struct
    class Smem:
        stage: fx.Array[fx.Int32, tiled.stage_words(rows, chunks), 16]
        word: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=f"t_qsa_proj_tp{sh.TP}_{rows}", known_block_size=[THREADS, 1, 1])
    def kern(
        x: Int64,
        w_qkv: Int64,
        w_ix: Int64,
        qkv: Int64,
        ix: Int64,
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
        gemm = tiled.Gemm(sm.stage.ptr, rows, tid, chunks=chunks)
        qsa.proj(g, gemm, sh, w_qkv, w_ix, x, qkv, ix, part, rows)

    @flyc.jit
    def launch(
        x: Int64,
        w_qkv: Int64,
        w_ix: Int64,
        qkv: Int64,
        ix: Int64,
        part: Int64,
        s0: Int64,
        s1: Int64,
        folds: Int64,
        health: Int64,
        stream: fx.Stream = default_stream,
    ):
        kern(x, w_qkv, w_ix, qkv, ix, part, s0, s1, folds, health).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch


@pytest.mark.parametrize(
    "tp, rows", tp_cases([1, 4, 8, 16, 128, 136, 192], (1, 16, 64))
)
def test_qsa_proj(harness, tp, rows):
    """qsa_front's projection phase: `x @ W_qkv.T` and `x @ W_ix.T` (bf16
    weights in vLLM's row-major layout), f32 accumulation of exact bf16
    products rounded to bf16 once, against float64 and against torch's bf16
    GEMM (the stock `qkv_proj` / `index_qk_proj`); rows past the step
    untouched."""
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import HIDDEN, shard
    from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import qsa

    sh = shard(tp)
    gen = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16)
    w_qkv = (torch.randn(sh.QKV, HIDDEN, generator=gen) * 0.05).to(torch.bfloat16)
    w_ix = (torch.randn(qsa.IX, HIDDEN, generator=gen) * 0.05).to(torch.bfloat16)
    dev = torch.device("cuda")
    xd, wqd, wid = x.to(dev), w_qkv.to(dev), w_ix.to(dev)
    qkv = torch.full((rows + 1, sh.QKV), 7.0, dtype=torch.bfloat16, device=dev)
    ix = torch.full((rows + 1, qsa.IX), 7.0, dtype=torch.bfloat16, device=dev)
    part = torch.zeros(qsa.partial_words(sh, rows), device=dev)
    launch = _qsa_proj_kernel(sh, rows)
    for _ in range(2):  # the fold counters reset themselves
        launch(
            xd.data_ptr(),
            wqd.data_ptr(),
            wid.data_ptr(),
            qkv.data_ptr(),
            ix.data_ptr(),
            part.data_ptr(),
            *harness.sync_args(),
            torch.cuda.current_stream(),
        )
    torch.accelerator.synchronize()
    assert harness.health[0].item() == 0
    for out, w, wd in ((qkv, w_qkv, wqd), (ix, w_ix, wid)):
        assert (out[rows] == 7.0).all()
        assert rel(out[:rows].float(), ref.bf(x.double() @ w.double().T)) < 1e-2
        assert rel(out[:rows].float(), (xd @ wd.T).float()) < 1e-2


@pytest.mark.parametrize("rows", [1, 4, 8, 16, 32])
def test_qsa_score_prepare(rows):
    """`wide.indexer.score` with vLLM's fused prepare in the launch
    (`prep`) against `ops.qsa_prepare` and the plain score on its output:
    the main queries, gates, K | V cache rows, the raw-key ring (keys and
    MRoPE positions), the compressed keys of the rows that complete a group
    (rotated at MRoPE positions read from the ring) and the scores, from
    decode rows of the projections; then the selection of those scores. A
    padded row writes no cache."""
    from vllm.models.qwen4_exp.amd.ops.qsa_prepare import qsa_prepare
    from vllm.models.qwen4_exp.amd.specialized_decode.wide import indexer

    gen = torch.Generator().manual_seed(200 + rows)
    dev = torch.device("cuda")
    page, cpage, size, ratio, top = 64, 64, 4, 4, 512
    seqs = [4096, 2052, 1031, 8, 3001, 2048, 37, 5000, 6]
    seqs = [seqs[r % len(seqs)] for r in range(rows)]
    lp = torch.tensor([s - 1 for s in seqs])
    t2r = torch.arange(rows, dtype=torch.int32)
    if rows > 2:
        t2r[2] = -1
    live = t2r >= 0

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)

    qkv, ixqk = rnd(rows, indexer.QKV_ROW), rnd(rows, indexer.IX_ROW)
    qn, kn, iqn, ikn = rnd(256, scale=0.2), rnd(256, scale=0.2), rnd(128), rnd(128)
    cos_sin = (torch.rand(8192, 64, generator=gen) * 2 - 1).to(torch.bfloat16)
    rope = torch.zeros(3, rows + 5, dtype=torch.int64)
    rope[:, :rows] = lp  # a decode row's three axes agree
    # main K | V: a page a row
    kv = rnd(rows + 1, 1, page, 512)
    main_slots = torch.tensor(
        [(r + 1) * page + int(lp[r]) % page if live[r] else -1 for r in range(rows)]
    )
    # the raw-key rings: one a request, keys then (t, h, w) of an image-like
    # position, so the compressed key's MRoPE axes differ
    ring = rnd(rows + 1, size, 140)
    for r in range(rows):
        for j in range(size):
            p = int(lp[r]) - ((int(lp[r]) - j) % size)
            ring[r + 1, j, 128:].view(torch.int64)[:] = torch.tensor(
                [p % 8000, (p + 3) % 8000, (p + 11) % 8000]
            )
    ring_table = torch.arange(1, rows + 1, dtype=torch.int32).view(rows, 1)
    ring_slots = torch.tensor(
        [(r + 1) * size + int(lp[r]) % size if live[r] else -1 for r in range(rows)]
    )
    # compressed keys: pages a row, then a padding page 0
    width = -(-(max(seqs) // ratio) // cpage)
    kc = rnd(rows * width + 1, cpage, 1, 128)
    ctable = (torch.randperm(rows * width, generator=gen) + 1).int().view(rows, width)
    comp_slots = torch.full((rows,), -1, dtype=torch.int64)
    for r in range(rows):
        b = int(lp[r]) // ratio
        if live[r] and (int(lp[r]) + 1) % ratio == 0:
            comp_slots[r] = int(ctable[r, b // cpage]) * cpage + b % cpage
    seq = torch.tensor(seqs, dtype=torch.int32)
    eps, ix_eps = 1e-6, 1e-6

    # stock: the fused prepare, then the plain scores of its queries
    d = dict(dev=dev)
    s_kv, s_ring, s_kc = kv.to(dev), ring.to(dev), kc.to(dev)
    s_ixq = torch.empty(rows, 4, 128, dtype=torch.bfloat16, device=dev)
    s_gate = torch.empty(rows, 12, 256, dtype=torch.bfloat16, device=dev)
    work = torch.stack([t2r.long(), torch.zeros(rows, dtype=torch.long)], 1)
    ixqk_d, qkv_d = ixqk.to(dev), qkv.to(dev)
    s_q, _ = qsa_prepare(
        ixqk_d[:, :512],
        ixqk_d[:, 512:],
        rope.to(dev)[:, :rows],
        cos_sin.to(dev),
        iqn.to(dev),
        ikn.to(dev),
        ix_eps,
        s_ixq,
        s_ring,
        ring_slots.to(dev),
        ring_table.to(dev),
        torch.arange(rows + 1, dtype=torch.int32, device=dev),
        lp.to(dev),
        s_kc,
        comp_slots.to(dev),
        work.int().to(dev),
        compress_ratio=ratio,
        mrope_section=(11, 11, 10),
        rope_pos_offset=128,
        main_qkv=qkv_d,
        main_q_norm_weight=qn.to(dev),
        main_k_norm_weight=kn.to(dev),
        main_eps=eps,
        main_kv_cache=s_kv.transpose(1, 2),
        main_slot_mapping=main_slots.to(dev),
        main_k_scale=1.0,
        main_v_scale=1.0,
        main_gate_out=s_gate,
    )
    columns = width * cpage
    s_logits = torch.zeros(rows, columns, device=dev)
    s_vis = torch.empty(rows, dtype=torch.int32, device=dev)
    common = (ctable.to(dev), t2r.to(dev), lp.to(dev), seq.to(dev), ratio)
    indexer.score(s_ixq, s_kc, *common, s_logits, s_vis)

    # the score launch's own prepare
    m_kv, m_ring, m_kc = kv.to(dev), ring.to(dev), kc.to(dev)
    query = torch.empty(rows, 12, 256, dtype=torch.bfloat16, device=dev)
    gate = torch.empty_like(query)
    logits = torch.zeros(rows, columns, device=dev)
    vis = torch.empty(rows, dtype=torch.int32, device=dev)
    sel = torch.full((rows, top), -7, dtype=torch.int32, device=dev)
    prep = indexer.Prep(
        ixqk_d,
        qkv_d,
        query,
        gate,
        rope.to(dev),
        cos_sin.to(dev),
        qn.to(dev),
        kn.to(dev),
        iqn.to(dev),
        ikn.to(dev),
        eps,
        ix_eps,
        (11, 10),
        m_kv,
        main_slots.to(dev),
        m_ring,
        ring_slots.to(dev),
        ring_table.to(dev),
        comp_slots.to(dev),
    )
    indexer.score(None, m_kc, *common, logits, vis, selected=sel, prep=prep)
    torch.accelerator.synchronize()
    del d

    assert torch.equal(vis, s_vis)
    assert torch.equal(gate, s_gate)
    assert rel(query.float(), s_q.float()) < 1e-2
    assert torch.equal(m_ring, s_ring)
    for mine, stock, init in ((m_kv, s_kv, kv), (m_kc, s_kc, kc)):
        mine, stock = mine.cpu(), stock.cpu()
        changed = (stock != init).flatten(1).any(1) | (mine != init).flatten(1).any(1)
        assert torch.equal(mine[~changed], stock[~changed])
        assert rel(mine.float(), stock.float()) < 1e-2
    logits, s_logits, vis, sel = logits.cpu(), s_logits.cpu(), vis.cpu(), sel.cpu()
    for r in range(rows):
        v = int(vis[r])
        if v <= top:
            assert torch.equal(sel[r][:v], torch.arange(v, dtype=torch.int32)), r
            continue
        assert rel(logits[r, :v], s_logits[r, :v]) < 1e-2, r
        got = sel[r][sel[r] >= 0].long()
        assert got.numel() == top, r
        boundary = logits[r, got].min()
        rest = torch.ones(v, dtype=torch.bool)
        rest[got] = False
        assert (logits[r, :v][rest] <= boundary).all(), r
