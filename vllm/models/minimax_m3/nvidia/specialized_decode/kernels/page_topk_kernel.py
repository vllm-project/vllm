# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, SIM102
"""page_topk device code (CuTe DSL): exact indexer top-16 page selector for the sparse-attention decode.

256 threads, 4 witness lanes per warp; the selection is exact, so the table is the same for any launch shape, bit
for bit.

Per (decode row, index head) one CTA selects, from the row's `cap = min(n_valid, num_cols, 8192)` pages, the
forced pages (the first `init` and the last `local`) plus the `need` best other pages by (fp32 key desc, page
asc): every thread holds KPT order-preserving keys in registers; per warp the R-th largest lane maximum is a witness,
and the K-th largest witness (K = ceil(need / R)) bounds a pool that provably holds the top `need`. A pool of at
most 64 entries is ranked by warp 0 (certificate); otherwise an exact block-wide MSD radix select runs on the
register keys. Winners are written in ascending page order as physical page ids (`-1` tail when fewer than 16
pages are valid).

`split_mode` (launches whose rows may exceed `single_max` pages): grid (rows x heads, splits). A row of at most
`single_max` pages runs the one-CTA select in split 0 (the other splits exit before the grid wait); a longer row
is cut into `chunk`-page CTAs that write their exact local top-`need` as `key << 32 | ~page` to `part`; the
last chunk CTA to arrive (per-row counter in `cnt`, threadfence pattern, re-armed for graph replay) ranks the
union and writes the row. No CTA waits for another.

PDL: n_valid and the page rows are read before `griddepcontrol.wait` (they are written before the predecessor),
the scores after it; dependents are released right after the wait.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import dsl_user_op

NT = 256  # threads per CTA
W = NT // 32
R = 4  # witness lanes per warp
TOPK = 16
MAX_CAP = 8192  # pages a row selects over
ONE_CAP = 4096  # pages one chunk CTA selects over
MAX_PART = 256  # partial entries one row merges (splits x 16)
POOL = 64
INT_MAX = 0x7FFFFFFF
DIAG_FAST32, DIAG_FAST64, DIAG_FALLBACK, DIAG_COPY, DIAG_MERGED = 0, 1, 2, 3, 4


# ------------------------------------------------------------------------------------------------ PTX primitives
def _asm(ret, args, text, cons, side, loc, ip):
    from cutlass._mlir.dialects import llvm as _llvm

    return _llvm.inline_asm(ret, [a.ir_value(loc=loc, ip=ip) for a in args], text, cons, has_side_effects=side,
                            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)  # fmt: skip


def _i32():
    from cutlass._mlir.extras import types as _T

    return _T.i32()


def _i64():
    from cutlass._mlir.extras import types as _T

    return _T.i64()


@dsl_user_op
def ldcg_u32(addr, pred, *, loc=None, ip=None):
    """`pred ? ld.global.cg.b32 [addr] : 0` (a predicated load: no access when pred is 0)."""
    return cutlass.Int32(_asm(_i32(), [cutlass.Int64(addr), cutlass.Int32(pred)],
                              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 $0, 0;\n\t"
                              "@p ld.global.cg.b32 $0, [$1];\n\t}", "=r,l,r", True, loc, ip))  # fmt: skip


@dsl_user_op
def ldcg_u64(addr, pred, *, loc=None, ip=None):
    """`pred ? ld.global.cg.b64 [addr] : 0`."""
    return cutlass.Int64(_asm(_i64(), [cutlass.Int64(addr), cutlass.Int32(pred)],
                              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b64 $0, 0;\n\t"
                              "@p ld.global.cg.b64 $0, [$1];\n\t}", "=l,l,r", True, loc, ip))  # fmt: skip


@dsl_user_op
def st_s32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(v)],
        "st.global.b32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_u64(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int64(v)],
        "st.global.b64 [$0], $1;",
        "l,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def atom_add_global(addr, v, *, loc=None, ip=None):
    return cutlass.Int32(_asm(_i32(), [cutlass.Int64(addr), cutlass.Int32(v)], "atom.global.add.u32 $0, [$1], $2;",
                              "=r,l,r", True, loc, ip))  # fmt: skip


@dsl_user_op
def red_add_global(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(v)],
        "red.global.add.u32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def atom_add_shared(saddr, v, *, loc=None, ip=None):
    return cutlass.Int32(_asm(_i32(), [cutlass.Int32(saddr), cutlass.Int32(v)], "atom.shared.add.u32 $0, [$1], $2;",
                              "=r,r,r", True, loc, ip))  # fmt: skip


@dsl_user_op
def threadfence(*, loc=None, ip=None):
    _asm(None, [], "membar.gl;", "", True, loc, ip)


@dsl_user_op
def pdl_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", True, loc, ip)


@dsl_user_op
def pdl_trigger(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", True, loc, ip)


@dsl_user_op
def ballot(pred, *, loc=None, ip=None):
    return cutlass.Int32(_asm(_i32(), [cutlass.Int32(pred)],
                              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $1, 0;\n\tvote.sync.ballot.b32 $0, p, 0xffffffff;\n\t}",
                              "=r,r", True, loc, ip))  # fmt: skip


@dsl_user_op
def redux_max_u32(v, *, loc=None, ip=None):
    return cutlass.Int32(_asm(_i32(), [cutlass.Int32(v)], "redux.sync.max.u32 $0, $1, 0xffffffff;", "=r,r", True,
                              loc, ip))  # fmt: skip


@dsl_user_op
def shfl_idx(v, src, *, loc=None, ip=None):
    return cutlass.Int32(_asm(_i32(), [cutlass.Int32(v), cutlass.Int32(src)],
                              "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=r,r,r", True, loc, ip))  # fmt: skip


@dsl_user_op
def shfl_up(v, off, *, loc=None, ip=None):
    return cutlass.Int32(_asm(_i32(), [cutlass.Int32(v), cutlass.Int32(off)],
                              "shfl.sync.up.b32 $0, $1, $2, 0x0, 0xffffffff;", "=r,r,r", True, loc, ip))  # fmt: skip


@dsl_user_op
def popc(v, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(_i32(), [cutlass.Int32(v)], "popc.b32 $0, $1;", "=r,r", False, loc, ip)
    )


@dsl_user_op
def ctz(v, *, loc=None, ip=None):
    """Index of the lowest set bit (`__ffs(v) - 1` for v != 0)."""
    return cutlass.Int32(_asm(_i32(), [cutlass.Int32(v)], "{\n\t.reg .b32 t;\n\tbrev.b32 t, $1;\n\tclz.b32 $0, t;\n\t}",
                              "=r,r", False, loc, ip))  # fmt: skip


@dsl_user_op
def lanemask_lt(*, loc=None, ip=None):
    return cutlass.Int32(
        _asm(_i32(), [], "mov.u32 $0, %lanemask_lt;", "=r", False, loc, ip)
    )


@dsl_user_op
def shr_u32(v, s, *, loc=None, ip=None):
    """Logical right shift; shift amounts >= 32 give 0 (PTX clamps)."""
    return cutlass.Int32(_asm(_i32(), [cutlass.Int32(v), cutlass.Int32(s)], "shr.b32 $0, $1, $2;", "=r,r,r", False,
                              loc, ip))  # fmt: skip


@dsl_user_op
def pack64(hi, lo, *, loc=None, ip=None):
    """`hi << 32 | lo` of two 32-bit words."""
    return cutlass.Int64(_asm(_i64(), [cutlass.Int32(lo), cutlass.Int32(hi)], "mov.b64 $0, {$1, $2};", "=l,r,r",
                              False, loc, ip))  # fmt: skip


@dsl_user_op
def lo32(v, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(_i32(), [cutlass.Int64(v)], "cvt.u32.u64 $0, $1;", "=r,l", False, loc, ip)
    )


@dsl_user_op
def smem_addr(arr, idx, *, loc=None, ip=None):
    """shared::cta address of int32 element `idx` of `arr`."""
    return cutlass.Int32(arr.data_ptr().toint(loc=loc, ip=ip)) + cutlass.Int32(
        idx
    ) * cutlass.Int32(4)


# ------------------------------------------------------------------------------------------------ helpers
def _sel(c, a, b):
    return cutlass.select_(c, a, b)


def _min(a, b):
    return cutlass.select_(a < b, a, b)


def _max(a, b):
    return cutlass.select_(a > b, a, b)


def _gt_u32(a, b):
    return cutlass.Uint32(a) > cutlass.Uint32(b)


def _ge_u32(a, b):
    return cutlass.Uint32(a) >= cutlass.Uint32(b)


def _gt_u64(a, b):
    return cutlass.Uint64(a) > cutlass.Uint64(b)


def okey(raw):
    """Order-preserving uint32 key of the fp32 bits."""
    return cutlass.select_(
        raw < cutlass.Int32(0), ~raw, raw | cutlass.Int32(-2147483648)
    )


def _ld(arr, idx):
    """arr[idx] (a free function: the DSL carries the receivers of method calls through dynamic regions)."""
    return arr.load(idx=idx)


def _st(arr, v, idx):
    arr.store(v, idx=idx)


def _bit(m, j: int):
    return shr_u32(m, cutlass.Int32(j)) & cutlass.Int32(1)


class Smem:
    """Shared memory of one CTA (the CUDA `struct Smem`; the V2 per-warp counts are not used by V4)."""

    def __init__(self):
        sp = cutlass.AddressSpace.smem
        self.spid = cutlass.Array(
            cutlass.Int32, MAX_CAP, space=sp, alignment=16
        )  # main-KV page rows (single rows)
        self.pk = cutlass.Array(
            cutlass.Int64, MAX_PART, space=sp, alignment=16
        )  # pool / merge: key << 32 | ~page
        self.hist = cutlass.Array(
            cutlass.Int32, 256 + 4, space=sp, alignment=16
        )  # radix; [256..259] ctl
        self.ckey = cutlass.Array(cutlass.Int32, 32, space=sp, alignment=16)
        self.cpg = cutlass.Array(cutlass.Int32, 32, space=sp, alignment=16)
        self.ws = cutlass.Array(
            cutlass.Int32, 32, space=sp, alignment=16
        )  # per-warp witness
        self.wsel = cutlass.Array(
            cutlass.Int32, TOPK, space=sp, alignment=16
        )  # winners' pages (any order)
        self.cnt = cutlass.Array(cutlass.Int32, 4, space=sp, alignment=16)


@cute.jit
def _diag(diag: cutlass.Int64, has_diag: cutlass.Constexpr, slot, pred):
    if cutlass.const_expr(has_diag):
        if pred:
            red_add_global(
                diag + cutlass.Int64(slot) * cutlass.Int64(4), cutlass.Int32(1)
            )


@cute.jit
def _store_page_order(
    s,
    need,
    base,
    orw: cutlass.Int64,
    lane,
    from_spid: cutlass.Constexpr,
    prw: cutlass.Int64,
):
    """Winners' pages in s.wsel[0..need) -> page-order store at orw[base + pos] (warp 0; V4: smem reads)."""
    pg = _sel(
        lane < need,
        _ld(s.wsel, _min(lane, cutlass.Int32(TOPK - 1))),
        cutlass.Int32(INT_MAX),
    )
    pos = cutlass.Int32(0)
    for j in cutlass.range_constexpr(TOPK):
        pos = pos + _sel(_ld(s.wsel, j) < pg, cutlass.Int32(1), cutlass.Int32(0))
    if lane < need:
        if cutlass.const_expr(from_spid):
            v = _ld(s.spid, pg)
        else:
            v = ldcg_u32(prw + cutlass.Int64(pg) * cutlass.Int64(4), cutlass.Int32(1))
        st_s32(orw + cutlass.Int64(base + pos) * cutlass.Int64(4), v)


@cute.jit
def _store_partial(s, need, srw: cutlass.Int64, c0, part: cutlass.Int64, lane):
    """Chunk CTA: winners in s.wsel[0..need) (chunk-local pages) -> 16 entries key << 32 | ~(c0 + page), zero tail."""
    e = cutlass.Int64(0)
    if lane < need:
        pg = _ld(s.wsel, _min(lane, cutlass.Int32(TOPK - 1)))
        key = okey(
            ldcg_u32(srw + cutlass.Int64(pg) * cutlass.Int64(4), cutlass.Int32(1))
        )
        e = pack64(key, ~(c0 + pg))
    if lane < TOPK:
        st_u64(part + cutlass.Int64(lane) * cutlass.Int64(8), e)


@cute.jit
def _radix_select(k, incm, need, s, tid, kpt: cutlass.Constexpr):
    """Exact top-`need` (key desc, page asc) among the included keys (bit j of incm): block-wide MSD radix select,
    8-bit passes until <= 32 candidates, mass-tie path in page order. Winners -> s.wsel[0..need) (warp 0 synced)."""
    warp = tid >> 5
    lane = tid & 31
    prefix = cutlass.Int32(0)
    krem = need
    sh = cutlass.Int32(24)
    npass = cutlass.Int32(0)
    go = cutlass.Int32(1)
    cute.arch.sync_threads()
    while go != cutlass.Int32(0):
        _st(s.hist, cutlass.Int32(0), idx=tid)  # NT == 256 bins
        if tid == 0:
            _st(s.hist, cutlass.Int32(0), idx=256 + 3)
        cute.arch.sync_threads()
        first = npass == cutlass.Int32(0)
        for j in cutlass.range_constexpr(kpt):
            if (_bit(incm, j) != cutlass.Int32(0)) & (
                first | (shr_u32(k[j], sh + cutlass.Int32(8)) == prefix)
            ):
                atom_add_shared(
                    smem_addr(s.hist, shr_u32(k[j], sh) & cutlass.Int32(255)),
                    cutlass.Int32(1),
                )
        cute.arch.sync_threads()
        if warp == 0:
            c = [
                _ld(s.hist, cutlass.Int32(255 - j) - cutlass.Int32(8) * lane)
                for j in range(8)
            ]
            tot = cutlass.Int32(0)
            for j in cutlass.range_constexpr(8):
                tot = tot + c[j]
            incl = tot
            for o in cutlass.range_constexpr(5):
                t = shfl_up(incl, cutlass.Int32(1 << o))
                incl = _sel(lane >= cutlass.Int32(1 << o), incl + t, incl)
            a = incl - tot
            if (a < krem) & (incl >= krem):
                bin_ = cutlass.Int32(-1)
                above = cutlass.Int32(0)
                inb = cutlass.Int32(0)
                for j in cutlass.range_constexpr(8):
                    hit = (bin_ < cutlass.Int32(0)) & (a + c[j] >= krem)
                    bin_ = _sel(
                        hit, cutlass.Int32(255 - j) - cutlass.Int32(8) * lane, bin_
                    )
                    above = _sel(hit, a, above)
                    inb = _sel(hit, c[j], inb)
                    a = a + c[j]
                kr = krem - above
                _st(s.hist, (prefix << cutlass.Int32(8)) | bin_, idx=256)
                _st(s.hist, kr, idx=257)
                _st(
                    s.hist,
                    _sel(
                        need - kr + inb <= cutlass.Int32(32),
                        cutlass.Int32(1),
                        cutlass.Int32(0),
                    ),
                    idx=258,
                )
        cute.arch.sync_threads()
        prefix = _ld(s.hist, 256)
        krem = _ld(s.hist, 257)
        stop = _ld(s.hist, 258)
        if stop != cutlass.Int32(0):
            go = cutlass.Int32(0)
        else:
            sh = sh - cutlass.Int32(8)
            npass = npass + cutlass.Int32(1)
            go = _sel(npass < cutlass.Int32(4), cutlass.Int32(1), cutlass.Int32(0))
    if sh >= cutlass.Int32(0):
        thr = prefix << sh
        for j in cutlass.range_constexpr(kpt):
            if (_bit(incm, j) != cutlass.Int32(0)) & _ge_u32(k[j], thr):
                at = atom_add_shared(smem_addr(s.hist, 256 + 3), cutlass.Int32(1))
                _st(s.ckey, k[j], idx=at)
                _st(s.cpg, tid + cutlass.Int32(NT * j), idx=at)
    else:
        tv = prefix
        nt = krem
        for j in cutlass.range_constexpr(kpt):
            if (_bit(incm, j) != cutlass.Int32(0)) & _gt_u32(k[j], tv):
                at = atom_add_shared(smem_addr(s.hist, 256 + 3), cutlass.Int32(1))
                _st(s.ckey, k[j], idx=at)
                _st(s.cpg, tid + cutlass.Int32(NT * j), idx=at)
        run = cutlass.Int32(0)
        for j in cutlass.range_constexpr(kpt):
            if run < nt:  # block-uniform
                tie = (_bit(incm, j) != cutlass.Int32(0)) & (k[j] == tv)
                bm = ballot(_sel(tie, cutlass.Int32(1), cutlass.Int32(0)))
                cute.arch.sync_threads()
                if lane == 0:
                    _st(s.hist, popc(bm), idx=warp)
                cute.arch.sync_threads()
                woff = cutlass.Int32(0)
                tot = cutlass.Int32(0)
                for w in cutlass.range_constexpr(W):
                    cw = _ld(s.hist, w)
                    woff = woff + _sel(cutlass.Int32(w) < warp, cw, cutlass.Int32(0))
                    tot = tot + cw
                if tie & (run + woff + popc(bm & lanemask_lt()) < nt):
                    at = atom_add_shared(smem_addr(s.hist, 256 + 3), cutlass.Int32(1))
                    _st(s.ckey, k[j], idx=at)
                    _st(s.cpg, tid + cutlass.Int32(NT * j), idx=at)
                run = run + tot
    cute.arch.sync_threads()
    if warp == 0:
        nc = _ld(s.hist, 256 + 3)
        ki = _sel(lane < nc, _ld(s.ckey, lane), cutlass.Int32(0))
        pi = _sel(lane < nc, _ld(s.cpg, lane), cutlass.Int32(INT_MAX))
        r = cutlass.Int32(0)
        for j in cutlass.range_constexpr(32):
            kj = shfl_idx(ki, cutlass.Int32(j))
            pj = shfl_idx(pi, cutlass.Int32(j))
            r = r + _sel(
                _gt_u32(kj, ki) | ((kj == ki) & (pj < pi)),
                cutlass.Int32(1),
                cutlass.Int32(0),
            )
        win = (lane < nc) & (r < need)
        bw = ballot(_sel(win, cutlass.Int32(1), cutlass.Int32(0)))
        if win:
            _st(s.wsel, pi, idx=popc(bw & lanemask_lt()))
        cute.arch.sync_warp()


@cute.jit
def _select_body(
    s,
    srw: cutlass.Int64,
    orw: cutlass.Int64,
    cap,
    init,
    fend,
    flo,
    need,
    tid,
    diag: cutlass.Int64,
    has_diag: cutlass.Constexpr,
    kpt: cutlass.Constexpr,
    part_mode: cutlass.Constexpr,
    part: cutlass.Int64,
    c0,
):
    """Witness-select body for one row (or one chunk when `part_mode`: srw and pages chunk-local, winners -> `part`)."""
    warp = tid >> 5
    lane = tid & 31
    # one batch of independent loads (the predicate depends only on pre-wait data)
    k = []
    for j in cutlass.range_constexpr(kpt):
        p = tid + cutlass.Int32(NT * j)
        k.append(
            ldcg_u32(
                srw + cutlass.Int64(p) * cutlass.Int64(4),
                _sel(p < cap, cutlass.Int32(1), cutlass.Int32(0)),
            )
        )
    incm = cutlass.Int32(0)
    lm = cutlass.Int32(0)
    for j in cutlass.range_constexpr(kpt):
        p = tid + cutlass.Int32(NT * j)
        k[j] = okey(k[j])
        v = (p < cap) & (p >= init) & (p < fend)  # non-forced, selectable
        incm = incm | _sel(
            v, cutlass.Int32(-2147483648 if j == 31 else 1 << j), cutlass.Int32(0)
        )
        lm = _sel(v & _gt_u32(k[j], lm), k[j], lm)
    # per-warp top-R lane maxima (one lane removed per round): S = R-th largest
    cur = lm
    sv_w = cutlass.Int32(0)
    for r in cutlass.range_constexpr(R):
        m = redux_max_u32(cur)
        sv_w = m
        if cutlass.const_expr(r + 1 < R):
            b = ballot(_sel(cur == m, cutlass.Int32(1), cutlass.Int32(0)))
            cur = _sel(lane == ctz(b), cutlass.Int32(0), cur)
    if lane == 0:
        _st(s.ws, sv_w, idx=warp)
    cute.arch.sync_threads()
    # T = K-th largest S_w, K = ceil(need / R): K warps x R distinct threads hold keys >= T
    kk = (need + cutlass.Int32(R - 1)) // cutlass.Int32(R)
    sv = _sel(lane < W, _ld(s.ws, _min(lane, cutlass.Int32(W - 1))), cutlass.Int32(0))
    c = cutlass.Int32(0)
    for v in cutlass.range_constexpr(W):
        c = c + _sel(_ge_u32(_ld(s.ws, v), sv), cutlass.Int32(1), cutlass.Int32(0))
    tmax = redux_max_u32(_sel((lane < W) & (c >= kk), sv, cutlass.Int32(0)))
    T = _sel(kk > cutlass.Int32(W), cutlass.Int32(0), tmax)
    km = cutlass.Int32(0)
    for j in cutlass.range_constexpr(kpt):
        km = km | _sel(
            (_bit(incm, j) != cutlass.Int32(0)) & _ge_u32(k[j], T),
            cutlass.Int32(-2147483648 if j == 31 else 1 << j),
            cutlass.Int32(0),
        )
    if km != cutlass.Int32(0):
        at = atom_add_shared(smem_addr(s.cnt, 0), popc(km))
        for j in cutlass.range_constexpr(kpt):
            bj = _bit(km, j)
            if (bj != cutlass.Int32(0)) & (at < cutlass.Int32(POOL)):
                _st(s.pk, pack64(k[j], ~(tid + cutlass.Int32(NT * j))), idx=at)
            at = at + bj
    cute.arch.sync_threads()
    n = _ld(s.cnt, 0)
    if (n <= cutlass.Int32(POOL)) & (n >= need):  # certificate (block-uniform)
        if warp == 0:
            e0 = _sel(lane < n, _ld(s.pk, lane), cutlass.Int64(0))
            e1 = _sel(
                lane + cutlass.Int32(32) < n,
                _ld(s.pk, lane + cutlass.Int32(32)),
                cutlass.Int64(0),
            )
            r0 = cutlass.Int32(0)
            r1 = cutlass.Int32(0)
            if n <= cutlass.Int32(32):
                for t in cutlass.range_constexpr(32):
                    r0 = r0 + _sel(
                        _gt_u64(_ld(s.pk, t), e0), cutlass.Int32(1), cutlass.Int32(0)
                    )
            else:
                for t in cutlass.range_constexpr(POOL):
                    x = _ld(s.pk, t)
                    r0 = r0 + _sel(_gt_u64(x, e0), cutlass.Int32(1), cutlass.Int32(0))
                    r1 = r1 + _sel(_gt_u64(x, e1), cutlass.Int32(1), cutlass.Int32(0))
            w0 = (lane < n) & (r0 < need)
            w1 = (lane + cutlass.Int32(32) < n) & (r1 < need)
            b0 = ballot(_sel(w0, cutlass.Int32(1), cutlass.Int32(0)))
            b1 = ballot(_sel(w1, cutlass.Int32(1), cutlass.Int32(0)))
            lt = lanemask_lt()
            if w0:
                _st(s.wsel, ~lo32(e0), idx=popc(b0 & lt))
            if w1:
                _st(s.wsel, ~lo32(e1), idx=popc(b0) + popc(b1 & lt))
            cute.arch.sync_warp()
            if cutlass.const_expr(part_mode):
                _store_partial(s, need, srw, c0, part, lane)
            else:
                _store_page_order(s, need, flo, orw, lane, True, orw)
            _diag(
                diag,
                has_diag,
                _sel(
                    n <= cutlass.Int32(32),
                    cutlass.Int32(DIAG_FAST32),
                    cutlass.Int32(DIAG_FAST64),
                ),
                lane == 0,
            )
    else:
        _radix_select(
            k, incm, need, s, tid, kpt
        )  # exact fallback on the register keys (no reload)
        if warp == 0:
            if cutlass.const_expr(part_mode):
                _store_partial(s, need, srw, c0, part, lane)
            else:
                _store_page_order(s, need, flo, orw, lane, True, orw)
            _diag(diag, has_diag, cutlass.Int32(DIAG_FALLBACK), lane == 0)


SPID_BATCH = 8  # independent page-row loads in flight per thread before their shared-memory stores


@cute.jit
def _stage_rows(s, prw: cutlass.Int64, cap, tid):
    """s.spid[p] = page_rows[p] for p < cap: SPID_BATCH predicated loads first, then the stores."""
    for b in cutlass.range(tid, cap, NT * SPID_BATCH):
        v = [
            ldcg_u32(
                prw + cutlass.Int64(b + cutlass.Int32(NT * i)) * cutlass.Int64(4),
                _sel(
                    b + cutlass.Int32(NT * i) < cap, cutlass.Int32(1), cutlass.Int32(0)
                ),
            )
            for i in range(SPID_BATCH)
        ]
        for i in cutlass.range_constexpr(SPID_BATCH):
            if b + cutlass.Int32(NT * i) < cap:
                _st(s.spid, v[i], idx=b + cutlass.Int32(NT * i))


@cute.jit
def _single_tiers(
    s, srw, orw, cap, init, fend, flo, need, tid, diag, has_diag: cutlass.Constexpr
):
    """The one-CTA select: 6, 16 or 32 keys per thread."""
    if cap <= cutlass.Int32(NT * 6):
        _select_body(
            s,
            srw,
            orw,
            cap,
            init,
            fend,
            flo,
            need,
            tid,
            diag,
            has_diag,
            6,
            False,
            cutlass.Int64(0),
            cutlass.Int32(0),
        )
    elif cap <= cutlass.Int32(NT * 16):
        _select_body(
            s,
            srw,
            orw,
            cap,
            init,
            fend,
            flo,
            need,
            tid,
            diag,
            has_diag,
            16,
            False,
            cutlass.Int64(0),
            cutlass.Int32(0),
        )
    else:
        _select_body(
            s,
            srw,
            orw,
            cap,
            init,
            fend,
            flo,
            need,
            tid,
            diag,
            has_diag,
            MAX_CAP // NT,
            False,
            cutlass.Int64(0),
            cutlass.Int32(0),
        )


@cute.kernel
def select_kernel(
    sc: cutlass.Int64, sc_st: cutlass.Int64, sc_sh: cutlass.Int64,  # scores fp32 base, strides (elements)
    nv: cutlass.Int64, pt: cutlass.Int64, pt_s0: cutlass.Int64, num_cols: cutlass.Int32,
    out: cutlass.Int64, out_sh: cutlass.Int64, out_st: cutlass.Int64, H: cutlass.Int32,
    init: cutlass.Int32, local: cutlass.Int32, diag: cutlass.Int64,
    chunk: cutlass.Int32, single_max: cutlass.Int32, part: cutlass.Int64, cnt: cutlass.Int64,
    split_mode: cutlass.Constexpr, has_diag: cutlass.Constexpr,
):  # fmt: skip
    s = Smem()
    tid, _, _ = cute.arch.thread_idx()
    row, split, _ = cute.arch.block_idx()
    _, nsplit_grid, _ = cute.arch.grid_dim()
    tid = cutlass.Int32(tid)
    row = cutlass.Int32(row)
    split = cutlass.Int32(split)
    hh = row % H
    tt = row // H
    srw = sc + (cutlass.Int64(tt) * sc_st + cutlass.Int64(hh) * sc_sh) * cutlass.Int64(
        4
    )
    prw = pt + cutlass.Int64(tt) * pt_s0 * cutlass.Int64(4)
    orw = out + (
        cutlass.Int64(hh) * out_sh + cutlass.Int64(tt) * out_st
    ) * cutlass.Int64(4)
    warp = tid >> 5
    lane = tid & 31
    # ---- pre-wait: n_valid and page rows are front outputs ----
    nvv = _min(
        ldcg_u32(nv + cutlass.Int64(tt) * cutlass.Int64(4), cutlass.Int32(1)), num_cols
    )
    if nvv <= cutlass.Int32(TOPK):
        if split == cutlass.Int32(0):
            if tid < cutlass.Int32(TOPK):
                v = _sel(
                    tid < nvv,
                    ldcg_u32(
                        prw + cutlass.Int64(tid) * cutlass.Int64(4),
                        _sel(tid < nvv, cutlass.Int32(1), cutlass.Int32(0)),
                    ),
                    cutlass.Int32(-1),
                )
                st_s32(orw + cutlass.Int64(tid) * cutlass.Int64(4), v)
            _diag(diag, has_diag, cutlass.Int32(DIAG_COPY), tid == cutlass.Int32(0))
            pdl_wait()
            pdl_trigger()
    else:
        cap = _min(nvv, cutlass.Int32(MAX_CAP))
        fend = nvv - local
        flo = _min(init, cap)
        fhi = _max(cutlass.Int32(0), cap - _max(fend, flo))
        need = cutlass.Int32(TOPK) - flo - fhi
        nsplit = cutlass.Int32(1)
        if cutlass.const_expr(split_mode):
            nsplit = _sel(
                (need <= cutlass.Int32(0)) | (cap <= single_max),
                cutlass.Int32(1),
                (cap + chunk - cutlass.Int32(1)) // chunk,
            )
        if (
            split < nsplit
        ):  # else: no work for this chunk slot (exits before the wait; counts as a trigger)
            if cutlass.const_expr(split_mode):
                if (nsplit == cutlass.Int32(1)) & (
                    need > cutlass.Int32(0)
                ):  # then cap <= single_max
                    _stage_rows(s, prw, cap, tid)
            else:
                _stage_rows(s, prw, cap, tid)
            if need <= cutlass.Int32(0):
                # more forced pages than slots: the lowest 16 forced pages (all keys tie at the forced key)
                if tid < cutlass.Int32(TOPK):
                    src = _sel(tid < flo, tid, cap - fhi + (tid - flo))
                    st_s32(
                        orw + cutlass.Int64(tid) * cutlass.Int64(4),
                        ldcg_u32(
                            prw + cutlass.Int64(src) * cutlass.Int64(4),
                            cutlass.Int32(1),
                        ),
                    )
                pdl_wait()
                pdl_trigger()
            else:
                # forced pages are the extreme pages: init at positions [0, flo), local at [16 - fhi, 16)
                if split == cutlass.Int32(0):
                    if tid < flo:
                        st_s32(
                            orw + cutlass.Int64(tid) * cutlass.Int64(4),
                            ldcg_u32(
                                prw + cutlass.Int64(tid) * cutlass.Int64(4),
                                cutlass.Int32(1),
                            ),
                        )
                    elif (tid >= cutlass.Int32(TOPK) - fhi) & (
                        tid < cutlass.Int32(TOPK)
                    ):
                        src = cap - (cutlass.Int32(TOPK) - tid)
                        st_s32(
                            orw + cutlass.Int64(tid) * cutlass.Int64(4),
                            ldcg_u32(
                                prw + cutlass.Int64(src) * cutlass.Int64(4),
                                cutlass.Int32(1),
                            ),
                        )
                if tid == cutlass.Int32(0):
                    _st(s.cnt, cutlass.Int32(0), idx=0)
                if tid < cutlass.Int32(POOL):
                    _st(
                        s.pk, cutlass.Int64(0), idx=tid
                    )  # V4 ranks over a fixed, zero-padded pool
                if tid < cutlass.Int32(TOPK):
                    _st(s.wsel, cutlass.Int32(INT_MAX), idx=tid)
                cute.arch.sync_threads()
                pdl_wait()
                pdl_trigger()
                if cutlass.const_expr(not split_mode):
                    _single_tiers(
                        s, srw, orw, cap, init, fend, flo, need, tid, diag, has_diag
                    )
                else:
                    if nsplit == cutlass.Int32(1):
                        _single_tiers(
                            s, srw, orw, cap, init, fend, flo, need, tid, diag, has_diag
                        )
                    else:
                        # ---- chunk CTA: exact local top-min(need, candidates) of pages [c0, c1) ----
                        c0 = split * chunk
                        c1 = _min(cap, c0 + chunk)
                        lo = _max(init, c0)
                        hi = _min(_min(fend, cap), c1)
                        avail = _max(cutlass.Int32(0), hi - lo)
                        prt = part + (
                            cutlass.Int64(row) * cutlass.Int64(nsplit_grid)
                            + cutlass.Int64(split)
                        ) * cutlass.Int64(TOPK * 8)
                        srw_c = srw + cutlass.Int64(c0) * cutlass.Int64(4)
                        if avail <= need:
                            # every candidate of the chunk is a local winner
                            if warp == 0:
                                e = cutlass.Int64(0)
                                if lane < avail:
                                    key = okey(
                                        ldcg_u32(
                                            srw
                                            + cutlass.Int64(lo + lane)
                                            * cutlass.Int64(4),
                                            cutlass.Int32(1),
                                        )
                                    )
                                    e = pack64(key, ~(lo + lane))
                                if lane < cutlass.Int32(TOPK):
                                    st_u64(
                                        prt + cutlass.Int64(lane) * cutlass.Int64(8), e
                                    )
                        elif c1 - c0 <= cutlass.Int32(NT * 6):
                            _select_body(
                                s,
                                srw_c,
                                orw,
                                c1 - c0,
                                init - c0,
                                fend - c0,
                                flo,
                                need,
                                tid,
                                diag,
                                has_diag,
                                6,
                                True,
                                prt,
                                c0,
                            )
                        else:
                            _select_body(
                                s,
                                srw_c,
                                orw,
                                c1 - c0,
                                init - c0,
                                fend - c0,
                                flo,
                                need,
                                tid,
                                diag,
                                has_diag,
                                ONE_CAP // NT,
                                True,
                                prt,
                                c0,
                            )
                        # ---- arrive; the last chunk CTA of the row merges (no CTA ever waits) ----
                        threadfence()  # the partial entries (written by warp 0) before the arrival
                        cute.arch.sync_threads()
                        if tid == cutlass.Int32(0):
                            _st(
                                s.cnt,
                                atom_add_global(
                                    cnt + cutlass.Int64(row) * cutlass.Int64(4),
                                    cutlass.Int32(1),
                                ),
                                idx=1,
                            )
                        cute.arch.sync_threads()
                        if _ld(s.cnt, 1) == nsplit - cutlass.Int32(1):
                            threadfence()
                            n = nsplit * cutlass.Int32(TOPK)  # <= MAX_PART == NT
                            rp = part + cutlass.Int64(row) * cutlass.Int64(
                                nsplit_grid
                            ) * cutlass.Int64(TOPK * 8)
                            e = ldcg_u64(
                                rp + cutlass.Int64(tid) * cutlass.Int64(8),
                                _sel(tid < n, cutlass.Int32(1), cutlass.Int32(0)),
                            )
                            _st(s.pk, e, idx=tid)
                            if tid < cutlass.Int32(TOPK):
                                _st(s.wsel, cutlass.Int32(INT_MAX), idx=tid)
                            if tid == cutlass.Int32(0):
                                st_s32(
                                    cnt + cutlass.Int64(row) * cutlass.Int64(4),
                                    cutlass.Int32(0),
                                )  # re-arm
                            cute.arch.sync_threads()
                            if e != cutlass.Int64(0):
                                r = cutlass.Int32(0)
                                for t in cutlass.range(n):
                                    r = r + _sel(
                                        _gt_u64(_ld(s.pk, t), e),
                                        cutlass.Int32(1),
                                        cutlass.Int32(0),
                                    )
                                if r < need:
                                    _st(s.wsel, ~lo32(e), idx=r)
                            cute.arch.sync_threads()
                            if warp == 0:
                                _store_page_order(s, need, flo, orw, lane, False, prw)
                                _diag(
                                    diag,
                                    has_diag,
                                    cutlass.Int32(DIAG_MERGED),
                                    lane == 0,
                                )


@cute.jit
def select_launch(
    sc: cutlass.Int64, sc_st: cutlass.Int64, sc_sh: cutlass.Int64,
    nv: cutlass.Int64, pt: cutlass.Int64, pt_s0: cutlass.Int64, num_cols: cutlass.Int32,
    out: cutlass.Int64, out_sh: cutlass.Int64, out_st: cutlass.Int64, rows: cutlass.Int32, H: cutlass.Int32,
    init: cutlass.Int32, local: cutlass.Int32, diag: cutlass.Int64,
    chunk: cutlass.Int32, single_max: cutlass.Int32, part: cutlass.Int64, cnt: cutlass.Int64, splits: cutlass.Int32,
    split_mode: cutlass.Constexpr, has_diag: cutlass.Constexpr, use_pdl: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):  # fmt: skip
    select_kernel(sc, sc_st, sc_sh, nv, pt, pt_s0, num_cols, out, out_sh, out_st, H, init, local, diag, chunk,
                single_max, part, cnt, split_mode, has_diag).launch(
        grid=(rows, splits, 1), block=(NT, 1, 1), stream=stream, use_pdl=use_pdl)  # fmt: skip
