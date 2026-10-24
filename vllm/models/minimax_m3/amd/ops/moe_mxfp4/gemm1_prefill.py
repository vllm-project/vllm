# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MiniMax-M3 prefill MoE stage 1 (a4w4): the dense 4-wave fp4 GEMM per (m-tile of
sorted rows, n-tile of 128 intermediate columns), aiter ``moe_sorting`` layout:

    h[row, :] = swiglu_oai(x[tok] @ W_gate[e]^T, x[tok] @ W_up[e]^T)
    out_q[row, :], out_scale[row, :] = mxfp4_quant(h[row, :])      (per 32 cols)

A rows are gathered by token; padded rows (token == n_tokens) fall outside the
buffer resource and load zeros. The two LDS N-halves are the gate slab and the
matching up slab of W13, so SwiGLU is elementwise in registers. The epilogue
quantizes like aiter's FlyDSL stage 1 and writes the scales in the sorted
e8m0-shuffled layout gemm2 reads.

Layouts (all bytes):
  A        [n_tokens, H/2]                 per-token fp4 (aiter per_1x32 quant)
  A_scale  [pad32(max_sorted), H/32]       sorted rows, e8m0-shuffled
                                           (aiter mxfp4_moe_sort)
  W13      [E, 2I, H/2]                    aiter shuffle_weight(layout=(16,16))
  W13_sc   [E*2I, H/32]                    aiter e8m0_shuffle
  OUT_Q    [num_m_blocks*BLOCK_M, I/2]     sorted rows, fp4
  OUT_sc   [pad32(num_m_blocks*BLOCK_M), I/32]  sorted rows, e8m0-shuffled
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import (
    buffer_ops,  # the copy shipped in the vLLM image
)
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

_N_WAVES = 4


def divmod(a, b):
    """Divmod for DSL values (the builtin rejects them)."""
    return (a // b, a % b)


def swizzle_128(row, col):
    """The dense kernel's 128-B-row XOR swizzle: (row, col) -> (row', col')."""
    offset = row * 128 + col
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // 128, swizzled_offset % 128


SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0


class _Buf:
    def __init__(self, base_ptr, byte_off):
        self.base_ptr = base_ptr
        self.byte_off = byte_off

    @property
    def ptr(self):
        return fx.add_offset(self.base_ptr, self.byte_off)


_gep = buffer_ops.get_element_ptr


def _lds_ptr_t():
    return _ir.Type.parse("!llvm.ptr<3>")


def _asm_void(operands, asm_string, constraints, clobbers=""):
    """Side-effecting void inline asm (LLVM sees no memory op -> no waitcnt added)."""
    if clobbers:
        constraints = f"{constraints},{clobbers}"
    _llvm.inline_asm(None, operands, asm_string, constraints, has_side_effects=True)


def wait_barrier(count):
    """``s_waitcnt vmcnt(count) lgkmcnt(0)`` + ``s_barrier``"""
    _rocdl.s_waitcnt(vmcnt=count, lgkmcnt=0)
    _rocdl.s_barrier()


def _uniform_i32(value):
    """Cast to i32 and force a wave-uniform SGPR value for scalar inline-asm
    operands."""
    raw = fx.as_ir_value(value) if not isinstance(value, _ir.Value) else value
    if raw.type != _T.i32:
        raw = fx.as_ir_value(fx.Int32(raw))
    return _rocdl.readfirstlane(_T.i32, raw)


def _swizzled_col(row, col):
    """The swizzled 128-B-row byte column ``swizzle_128`` lands (row, col) on."""
    r, c = swizzle_128(row, col)
    return c


class G2SLoaderAsm:
    """global -> LDS, 16 B per lane per step; ``gl_offsets[step]`` is the
    loop-invariant per-lane byte offset, the K-step goes in soffset."""

    def __init__(self, rsrc, gl_offsets, n_load_steps, wave_id):
        self.rsrc = fx.as_ir_value(rsrc)
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id

    @property
    def _step_stride(self):
        # m0 (LDS byte) advance per step: 4 waves x 64 lanes x 16 B.
        return _N_WAVES * 1024

    def set_wave_base(self, base_ptr):
        # The wave-uniform LDS base, readfirstlane'd into an SGPR ONCE.
        wb = fx.Int32(fx.ptrtoint(base_ptr)) + fx.Int32(self.wave_id * 1024)
        self._wave_base_s = _rocdl.readfirstlane(_T.i32, fx.as_ir_value(wb))

    def _lds_base_sgpr(self, lds_dst):
        m0 = fx.Int32(self._wave_base_s) + fx.Int32(lds_dst.byte_off)
        return fx.as_ir_value(m0)

    def _voffset(self, step):
        return fx.as_ir_value(fx.Int32(self.gl_offsets[step]))

    def _emit(self, lds_dst, k_offset, step):
        # m0 idiom (gcnasm async_copy): set m0 for step 0, then s_add for the rest.
        voff = self._voffset(step)
        soff = _uniform_i32(k_offset)  # scalar soffset (K-step)
        stride = self._step_stride
        # s_add_u32 writes SCC: declare it, or the compiler may keep a live SCC
        # (e.g. a loop-exit compare) across this asm and branch on garbage.
        if step == 0:
            m0 = self._lds_base_sgpr(lds_dst)
            asm = "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds"
            _asm_void([m0, voff, self.rsrc, soff], asm, "s,v,s,s", "~{scc}")
        else:
            asm = (
                f"s_add_u32 m0, {stride}, m0\nbuffer_load_dwordx4 $0, $1, $2 offen lds"
            )
            _asm_void([voff, self.rsrc, soff], asm, "v,s,s", "~{scc}")

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            self._emit(lds_dst, k_offset, step)

    def load_one(self, lds_dst, k_offset, step):
        self._emit(lds_dst, k_offset, step)


class S2RLoaderFp4:
    """LDS -> reg: per tile the two K=128 fp4 MFMA operands (i32x4 each)."""

    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, dyn_offset, const_offset):
        total_off = lds_src.byte_off + const_offset
        window_base = (total_off // 0x10000) * 0x10000
        imm = total_off - window_base
        assert 0 <= imm <= 0xFFFF
        vaddr = fx.Int32(fx.ptrtoint(lds_src.base_ptr)) + fx.Int32(
            window_base + dyn_offset
        )
        lds_ptr = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(vaddr))
        if imm != 0:
            lds_ptr = _gep(lds_ptr, static_byte_offset=imm)
        vec4_i32 = _ir.VectorType.get([4], fx.Int32.ir_type)
        load = _llvm.LoadOp(vec4_i32, lds_ptr, alignment=16)
        return Vec(load.result)

    def _dyn_offset(self, step, preshuffled):
        row = self.wave_idx * (self.n_tiles * 16) + self.lane_id % 16
        col = (self.lane_id // 16) * 16 + step * 64
        if const_expr(preshuffled):
            return (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
        row_swz, col_swz = swizzle_128(row, col)
        return row_swz * 128 + col_swz

    def load(self, lds_src, preshuffled=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            for step in range_constexpr(2):
                dyn = self._dyn_offset(step, preshuffled)
                v = self._vec_load_16xf8(lds_src, dyn, i * 2048)
                halves.append(v.bitcast(fx.Int32))
            frag.append(halves)
        return frag

    def load_one(self, lds_src, i, ksub, preshuffled=False):
        dyn = self._dyn_offset(ksub, preshuffled)
        v = self._vec_load_16xf8(lds_src, dyn, i * 2048)
        return v.bitcast(fx.Int32)


def _flat_frag(frag):
    out = []
    for t in frag:
        out.append(fx.as_ir_value(t[0]))
        out.append(fx.as_ir_value(t[1]))
    return out


def _unflat_frag(flat, n_tiles):
    return [[flat[2 * i], flat[2 * i + 1]] for i in range(n_tiles)]


def _g2s_thunks(g2s, dst, gl_off, n_steps):
    return [lambda s=s: g2s.load_one(dst, gl_off, s) for s in range(n_steps)]


def _riffle(glb, lds):
    """Interleave the global and LDS thunk lists proportionally like aiter's asm"""
    if not glb or not lds:
        return list(glb) + list(lds)
    out = []
    step = len(lds) / len(glb)
    li = 0
    for gi, t in enumerate(glb):
        out.append(t)
        upto = int(round((gi + 1) * step))
        out += lds[li:upto]
        li = upto
    return out + lds[li:]


def _s2r_thunks(s2r, src, holder, n, pre):
    ts = []
    for i in range(n):
        for ks in range(_FP4_PACK):

            def f(i=i, ks=ks):
                if holder[i] is None:
                    holder[i] = [None, None]
                holder[i][ks] = s2r.load_one(src, i, ks, preshuffled=pre)

            ts.append(f)
    return ts


def _min(a, b):
    return (a < b).select(a, b)


def _divmod_nonneg(a, b):
    if const_expr(isinstance(b, int) and b > 0 and (b & (b - 1)) == 0):
        sh = b.bit_length() - 1
        return (a >> sh, a & (b - 1)) if const_expr(sh > 0) else (a, 0)
    return divmod(a, b)


# ── FP4 scaled MFMA ──────────────────────────────────────────────────────────
_FP4_CBSZ = 4
_FP4_BLGP = 4
_FP4_PACK = 2  # pack_M = pack_N = pack_K = 2


class Mfma16x16x128Fp4:
    """fp4 16x16x128 scaled MFMA over an (n_tiles_a x n_tiles_b) quadrant,
    accumulators pinned in AGPR (inline asm ``=a,...,0``). See the dense
    kernel for the opsel / operand-swap notes."""

    def __init__(self, n_tiles_a, n_tiles_b):
        assert n_tiles_a % _FP4_PACK == 0 and n_tiles_b % _FP4_PACK == 0
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.res_ty = Vec.make_type(4, fx.Float32)

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _order(self):
        order = []
        j0s = list(range(0, self.n_tiles_b, 2))
        for n, i0 in enumerate(range(0, self.n_tiles_a, 2)):
            for j0 in reversed(j0s) if n % 2 else j0s:
                order += [(i0 + di, j0 + dj) for di in range(2) for dj in range(2)]
        return order

    def call(
        self, a, b, c, sa, sb, interleave=None, zero_acc=False, late=None, late_start=8
    ):
        """``interleave``: thunks spread evenly over the MFMAs (loads, LDS reads).
        ``late``: thunks spread over the MFMAs from ``late_start`` on -- for work that
        reads accumulators finished by the previous call: the MFMAs are inline asm, so
        the
        compiler inserts no MFMA -> v_accvgpr_read hazard waits; 8 MFMAs (128+ cycles)
        cover the last one's latency."""
        thunks = list(interleave) if interleave else []
        late_ids = list(late) if late else []
        nth = [0]
        nlt = [0]
        mth = [0]
        order = self._order()
        n_mfma = _FP4_PACK * len(order)
        slots = (
            {(t * n_mfma) // len(thunks) for t in range(len(thunks))}
            if thunks
            else set()
        )
        n_late_slots = max(n_mfma - late_start, 1)
        lslots = (
            {
                late_start + (t * n_late_slots) // len(late_ids)
                for t in range(len(late_ids))
            }
            if late_ids
            else set()
        )
        for ksub in range_constexpr(_FP4_PACK):
            for i, j in order:
                a_op = a[i][ksub]
                sa_v = sa[i // _FP4_PACK]
                ia = i % _FP4_PACK
                b_op = b[j][ksub]
                sb_v = sb[j // _FP4_PACK]
                jb = j % _FP4_PACK
                if zero_acc and ksub == 0:
                    c[self.idx(i, j)] = self._mfma_agpr(
                        a_op, b_op, None, sa_v, sb_v, ksub, ia, jb
                    )
                else:
                    c[self.idx(i, j)] = self._mfma_agpr(
                        a_op, b_op, c[self.idx(i, j)], sa_v, sb_v, ksub, ia, jb
                    )
                if nth[0] < len(thunks) and mth[0] in slots:
                    thunks[nth[0]]()
                    nth[0] += 1
                if nlt[0] < len(late_ids) and mth[0] in lslots:
                    late_ids[nlt[0]]()
                    nlt[0] += 1
                mth[0] += 1
        while nth[0] < len(thunks):
            thunks[nth[0]]()
            nth[0] += 1
        while nlt[0] < len(late_ids):
            late_ids[nlt[0]]()
            nlt[0] += 1
        return c

    def _mfma_agpr(self, a_op, b_op, acc, sa_v, sb_v, ksub, ia, jb):
        # feeds (B, A) instead of (A, B): C^T = B^T A^T, so lane L ends up holding
        # C[row L%16, 4 consecutive cols 4*(L//16)..].
        a_op, b_op = b_op, a_op
        sa_v, sb_v = sb_v, sa_v
        ia, jb = jb, ia
        opsel = f"op_sel:[{ia},{jb},0]"
        opsel_hi = f"op_sel_hi:[{ksub},{ksub},0]"
        src2 = "$0" if acc is not None else "0"
        asm = (
            f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 "
            f"{opsel} {opsel_hi} cbsz:4 blgp:4"
        )
        ops = [
            fx.as_ir_value(a_op),
            fx.as_ir_value(b_op),
            fx.as_ir_value(sa_v),
            fx.as_ir_value(sb_v),
        ]
        cons = "=a,v,v,v,v"
        if acc is not None:
            ops.append(fx.as_ir_value(acc))
            cons += ",0"
        return _llvm.inline_asm(self.res_ty, ops, asm, cons, has_side_effects=True)


# ── E8M0 scales through LDS ──────────────────────────────────────────────────
_SCALE_QUARTER_BYTES = 1024  # one gather = 4 blocks = one wave's share of one operand
_SCALE_REGION_BYTES = 2 * _SCALE_QUARTER_BYTES  # all of A (or all of B) for one K-step
_SCALE_SLOT_BYTES = _N_WAVES * _SCALE_QUARTER_BYTES  # 4096
_SCALE_SLOTS = 4
_SCALE_LDS_BYTES = _SCALE_SLOTS * _SCALE_SLOT_BYTES  # 16 KB
_SCALE_A_REGION = 0
_SCALE_B_REGION = _SCALE_REGION_BYTES


class ScaleGatherMoE:
    """One ``buffer_load_dwordx4 ... lds`` per wave per K-step: waves 0/1 fetch
    the A-scale blocks (rows of the sorted scale), waves 2/3 the B-scale blocks
    (gate slab + up slab of the expert). A 256-B block = 32 rows x 8 K-blocks;
    lane block ``blk = lane//16`` fetches row-group
    ``G_wave + (blk//2)*half_groups + blk%2``: blocks 0/1 belong to LDS half 0
    (rows R0 / gate), blocks 2/3 to half 1 (rows R1 / up)."""

    def __init__(
        self,
        a_scale,
        b_scale,
        K,
        lane_id,
        wave_id,
        lds_base_ptr,
        a_scale_bytes,
        b_scale_bytes,
        a_wave_groups,
        a_half_groups,
        b_wave_groups,
        b_half_groups,
    ):
        self.row_i32 = (K // 256) * 64  # i32 per 32-row group (32 rows x K/32 bytes)
        self.wave_id = wave_id
        # aiter's buffer_ops returns the raw ROCDL resource (!llvm.ptr<8>) directly
        self.a_rsrc = fx.as_ir_value(
            buffer_ops.create_buffer_resource(
                a_scale, max_size=False, num_records_bytes=a_scale_bytes
            )
        )
        self.b_rsrc = fx.as_ir_value(
            buffer_ops.create_buffer_resource(
                b_scale, max_size=False, num_records_bytes=b_scale_bytes
            )
        )
        self._blk = lane_id // 16
        self._in16 = lane_id % 16
        self._lds_base = fx.Int32(fx.ptrtoint(lds_base_ptr))
        self._a_wave_groups = a_wave_groups
        self._a_half_groups = a_half_groups
        self._b_wave_groups = b_wave_groups
        self._b_half_groups = b_half_groups

    def set_wave_base(self, a_base_row, b_base_row):
        """``a_base_row``: first sorted row of the m-tile; ``b_base_row``: first
        W13 row (gate) of the n-tile, both multiples of 32."""
        wid = fx.Int32(_rocdl.readfirstlane(_T.i32, fx.as_ir_value(self.wave_id)))
        self._wave_base_s = fx.as_ir_value(
            self._lds_base + wid * fx.Int32(_SCALE_QUARTER_BYTES)
        )
        is_a = wid < fx.Int32(2)
        q = wid % fx.Int32(2)
        g_a = a_base_row // fx.Int32(32) + q * fx.Int32(self._a_wave_groups)
        g_b = b_base_row // fx.Int32(32) + q * fx.Int32(self._b_wave_groups)
        self._G = _uniform_i32(is_a.select(g_a, g_b))
        self._HS = _uniform_i32(
            is_a.select(fx.Int32(self._a_half_groups), fx.Int32(self._b_half_groups))
        )
        self._rsrc = fx.arith.select(is_a, self.a_rsrc, self.b_rsrc)
        self._soff0 = _uniform_i32(fx.Int32(0))

    def gather(self, kstep, slot):
        grp = (
            fx.Int32(self._G) + (self._blk // 2) * fx.Int32(self._HS) + (self._blk % 2)
        )
        i32_off = (
            grp * fx.Int32(self.row_i32)
            + fx.Int32(kstep) * fx.Int32(64)
            + self._in16 * fx.Int32(4)
        )
        voff = fx.as_ir_value(i32_off * fx.Int32(4))  # bytes
        addr = fx.Int32(self._wave_base_s) + fx.Int32(slot) * fx.Int32(
            _SCALE_SLOT_BYTES
        )
        asm = "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds"
        _asm_void(
            [fx.as_ir_value(addr), voff, self._rsrc, self._soff0],
            asm,
            "s,v,s,s",
            "~{m0}",
        )


class ScaleLoaderLDS:
    def __init__(self, n_tiles, lane_id, quarter, lds_base_ptr, region_off):
        assert n_tiles % _FP4_PACK == 0
        self.n_groups = n_tiles // _FP4_PACK
        self.lane_id = lane_id
        self._region_base = (
            fx.Int32(fx.ptrtoint(lds_base_ptr))
            + fx.Int32(region_off)
            + quarter * fx.Int32(_SCALE_QUARTER_BYTES)
        )

    def _slot_wave_byte(self, slot):
        return self._region_base + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES)

    def read_half(self, slot, half):
        L = self.lane_id
        base = self._slot_wave_byte(slot) + fx.Int32((L // 4) * 16 + (L % 4) * 4)
        grp_list = []
        for gi in range_constexpr(self.n_groups):
            blk = half * 2 + gi
            vaddr = base + fx.Int32(blk * 256)
            lds_ptr = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(vaddr))
            load = _llvm.LoadOp(fx.Int32.ir_type, lds_ptr, alignment=4)
            grp_list.append(fx.Int32(load.result))
        return grp_list

    def read(self, slot):
        return self.read_half(slot, 0), self.read_half(slot, 1)


# ── epilogue helpers ─────────────────────────────────────────────────────────
def _fmax(a, b):
    return (a > b).select(a, b)


def _fmin(a, b):
    return (a < b).select(a, b)


def _swiglu_oai(g, u):
    """MiniMax-M3 activation, op for op the production stage-1 epilogue (aiter
    mixed_moe_gemm_2stage ``swiglu_mul_vec4``): g clamped above, u clamped both
    sides, t = (g * alpha) * (-log2 e), sigmoid = rcp(1 + exp2(t)), g * sig * (u + 1).
    The two separate multiplies matter: folding the constant changes the last bit."""
    lim = fx.Float32(SWIGLU_LIMIT)
    g = _fmin(g, lim)
    u = _fmax(_fmin(u, lim), fx.Float32(-SWIGLU_LIMIT))
    t = (g * fx.Float32(SWIGLU_ALPHA)) * fx.Float32(-1.4426950408889634)
    e = fx.Float32(_rocdl.exp2(_T.f32, t.ir_value()))
    sig = fx.Float32(_rocdl.rcp(_T.f32, (fx.Float32(1.0) + e).ir_value()))
    return g * sig * (u + fx.Float32(1.0))


def _round_bf16x2(a, b):
    """(a, b) -> the f32 values of their bf16 roundings (RNE, v_cvt_pk_bf16_f32):
    production stage 1 stores bf16 and quantises from it."""
    w = Vec.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    return _as_f32(w << 16), _as_f32(w & fx.Int32(-65536))


def _quant_prep_fp4(h8):
    """Production's inter-stage quant on 8 values of one 32-block: round them to
    bf16 and return (rounded values, this lane's |max|)."""
    hb = []
    for k in range_constexpr(0, 8, 2):
        lo, hi = _round_bf16x2(h8[k], h8[k + 1])
        hb += [lo, hi]
    amax = fx.math.absf(hb[0])
    for v in range_constexpr(1, 8):
        amax = _fmax(amax, fx.math.absf(hb[v]))
    return hb, amax


def _e8m0_roundup_fp4(amax):
    """Aiter's default MX scale rule (kDefaultMxScaleRoundMode = RoundUp):
    ceil_pow2(amax / 6) as a biased exponent (fp4 max = 6): the block's max lands in
    (3, 6]. Exponent 0xFF (NaN/Inf) is not bumped."""
    u = _bits(amax * fx.Float32(1.0 / 6.0))
    e = (u >> 23) & fx.Int32(0xFF)
    bump = ((u & fx.Int32(0x7FFFFF)) != fx.Int32(0)) & (e < fx.Int32(0xFF))
    return bump.select(e + fx.Int32(1), e)


def _bits(f):
    return fx.Float32(f).bitcast(fx.Int32)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _cvt_pk_fp4(old, a, b, scale_f32, sel):
    """v_cvt_scalef32_pk_fp4_f32: two f32 / 2^(e8m0-127) -> 2 fp4 into byte ``sel``."""
    return fx.Int32(_rocdl.cvt_scalef32_pk_fp4_f32(_T.i32, old, a, b, scale_f32, sel))


def _permlane16_swap(d_a, d_b):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = _rocdl.permlane16_swap(
        pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False
    )
    return fx.Int32(_llvm.extractvalue(_T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(_T.i32, res, [1])
    )


def compile_moe_gemm1(
    *,
    H: int,
    I: int,  # noqa: E741
    E: int,
    BLOCK_M: int = 128,
):
    """Grouped fp4 gemm1 for one (H, I, E, BLOCK_M). ``BLOCK_M`` must equal the
    ``moe_sorting`` block size the sorted inputs were built with (128 or 256).

    Block order comes from an int32 table built on the GPU by ``tile_map.py``
    (one small kernel; ``tile_map[grid_size]`` holds the number of valid entries,
    so no host sync): ``tile_map[remapped block] = m_tile << 3 | n_tile`` (-1 =
    nothing to do), laid out expert by expert and n-slab-major inside an expert,
    so the 32 CUs of one XCD chew through one expert with the same 768 KB
    gate/up slab of W13 in L2. The hardware deals
    consecutive block ids round-robin over the 8 XCDs, so block id b is first
    remapped to ``(b % 8) * (grid / 8) + b // 8`` = a contiguous chunk of the
    table per XCD."""
    K = H
    BLOCK_K = 256
    BLOCK_K_BYTES = BLOCK_K // 2
    BLOCK_N = 256  # 128 gate cols + the matching 128 up cols
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_TILES_A = LDS_BLOCK_M // 2 // 16  # 16-row tiles per wave per LDS half
    N_TILES_B = LDS_BLOCK_N // 2 // 16  # = 4

    assert BLOCK_M in (128, 256)
    assert K % BLOCK_K == 0 and I % LDS_BLOCK_N == 0 and (2 * I) % 256 == 0
    K_ITERS = K // BLOCK_K
    UNROLL = 4 if (K_ITERS - 4) % 4 == 0 else 2
    assert K_ITERS >= 4 and (K_ITERS - 4) % UNROLL == 0, K_ITERS
    N_ACCUMS = N_TILES_A * N_TILES_B
    K_BYTES = K // 2

    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES  # 8 KB (BM128) / 16 KB (BM256)
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES  # 16 KB
    A_BUFS = 4 * a_lds_size
    LDS_TILES_BYTES = A_BUFS + 4 * b_lds_size

    # scale geometry (32-row groups)
    A_WAVE_GROUPS = N_TILES_A // 2  # groups per wave per LDS half
    A_HALF_GROUPS = LDS_BLOCK_M // 32
    B_WAVE_GROUPS = N_TILES_B // 2
    B_HALF_GROUPS = I // 32  # gate slab -> up slab
    I_BYTES = I // 2
    SCALE_COLS_OUT = I // 32  # e8m0 per output row
    OUT_SC_BLOCKS_PER_ROW32 = SCALE_COLS_OUT // 8

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, LDS_TILES_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, _SCALE_LDS_BYTES, 16]

    @flyc.kernel
    def kernel_gemm1(
        A: fx.Tensor,
        W13: fx.Tensor,
        OUT_Q: fx.Tensor,
        A_scale: fx.Tensor,
        W13_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        tile_map_t: fx.Tensor,
        grid_size: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _base_ptr = lds.all_lds.ptr

        a_cur0 = _Buf(_base_ptr, 0 * a_lds_size)
        a_cur1 = _Buf(_base_ptr, 1 * a_lds_size)
        a_next0 = _Buf(_base_ptr, 2 * a_lds_size)
        a_next1 = _Buf(_base_ptr, 3 * a_lds_size)
        b_cur0 = _Buf(_base_ptr, A_BUFS + 0 * b_lds_size)
        b_cur1 = _Buf(_base_ptr, A_BUFS + 1 * b_lds_size)
        b_next0 = _Buf(_base_ptr, A_BUFS + 2 * b_lds_size)
        b_next1 = _Buf(_base_ptr, A_BUFS + 3 * b_lds_size)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        ids_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BLOCK_M * 4)
        )
        eid_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4
        )
        # split the valid entries [0, tile_map[grid_size]) evenly over the 8 XCDs
        intra_xcd, xcd = _divmod_nonneg(fx.block_idx.x, 8)
        tm_rsrc = buffer_ops.create_buffer_resource(
            tile_map_t, max_size=False, num_records_bytes=(grid_size + 1) * 4
        )
        n_valid = fx.Int32(
            buffer_ops.buffer_load(
                tm_rsrc, grid_size, vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        per_xcd = (n_valid + fx.Int32(7)) // fx.Int32(8)
        remapped = xcd * per_xcd + intra_xcd
        in_chunk = (intra_xcd < per_xcd) & (remapped < n_valid)
        entry = fx.Int32(
            buffer_ops.buffer_load(
                tm_rsrc,
                in_chunk.select(remapped, fx.Int32(0)),
                vec_width=1,
                dtype=fx.Int32,
            )
        )
        entry = in_chunk.select(entry, fx.Int32(-1))
        tile_i = entry >> 3
        tile_j = entry & 7
        block_valid = entry >= 0
        # ---- routing: this m-tile's expert ----
        expert = fx.Int32(
            buffer_ops.buffer_load(eid_rsrc, tile_i, vec_width=1, dtype=fx.Int32)
        )
        m_base = tile_i * BLOCK_M

        if block_valid:
            wave_i = wave_id // 2
            wave_j = wave_id % 2

            # ---- A rows: gathered. token(sorted row) * K_BYTES + swizzled col ----
            def _a_gather_offsets(half):
                offs = []
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id % 8) * 16
                    sid = fx.Int32(
                        buffer_ops.buffer_load(
                            ids_rsrc,
                            m_base + half * LDS_BLOCK_M + row,
                            vec_width=1,
                            dtype=fx.Int32,
                        )
                    )
                    tok = sid & fx.Int32(
                        0x00FFFFFF
                    )  # padded rows: tok == n_tokens -> OOB -> zeros
                    offs.append(tok * fx.Int32(K_BYTES) + _swizzled_col(row, col))
                return offs

            gl_off_a0 = _a_gather_offsets(0)
            gl_off_a1 = _a_gather_offsets(1)

            # ---- B: gate slab rows [e*2I + j*128, +128), up slab +I ----
            def _b_offsets():
                offs = []
                for rnd in range_constexpr(N_TILES_B):
                    row = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id // 8) * 16
                    offs.append(
                        (row // 16) * (K_BYTES * 16)
                        + (row % 16) * 16
                        + (col // 64) * 1024
                        + ((col % 64) // 16) * 256
                        + (col % 16)
                    )
                return offs

            gl_off_b = _b_offsets()
            b_row0 = expert * (2 * I) + tile_j * LDS_BLOCK_N
            B0_gl_offset = b_row0 * K_BYTES
            B1_gl_offset = B0_gl_offset + I * K_BYTES
            A_K_STEP = BLOCK_K_BYTES
            B_K_STEP = 2 * 1024

            mfma = Mfma16x16x128Fp4(N_TILES_A, N_TILES_B)

            _scale_base_ptr = lds.scale_lds.ptr
            scale_gather = ScaleGatherMoE(
                A_scale,
                W13_scale,
                K,
                lane_id,
                wave_id,
                _scale_base_ptr,
                a_scale_bytes,
                E * 2 * I * (K // 32),
                A_WAVE_GROUPS,
                A_HALF_GROUPS,
                B_WAVE_GROUPS,
                B_HALF_GROUPS,
            )
            scale_gather.set_wave_base(m_base, b_row0)
            a_scale_ld = ScaleLoaderLDS(
                N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION
            )
            b_scale_ld = ScaleLoaderLDS(
                N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION
            )

            def _slot(k):
                return fx.Int32(k) % fx.Int32(_SCALE_SLOTS)

            def _gather_scale_thunks(k, slot):
                return [lambda: scale_gather.gather(k, slot)]

            a_rsrc = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=n_tokens * K_BYTES
            )
            b_rsrc = buffer_ops.create_buffer_resource(
                W13, max_size=False, num_records_bytes=E * (2 * I) * K_BYTES
            )
            a0_g2s = G2SLoaderAsm(a_rsrc, gl_off_a0, N_TILES_A, wave_id)
            a1_g2s = G2SLoaderAsm(a_rsrc, gl_off_a1, N_TILES_A, wave_id)
            b_g2s = G2SLoaderAsm(b_rsrc, gl_off_b, N_TILES_B, wave_id)
            a0_g2s.set_wave_base(_base_ptr)
            a1_g2s.set_wave_base(_base_ptr)
            b_g2s.set_wave_base(_base_ptr)
            a_s2r = S2RLoaderFp4(wave_i, N_TILES_A)
            b_s2r = S2RLoaderFp4(wave_j, N_TILES_B)

            scale_gather.gather(0, _slot(0))
            scale_gather.gather(1, _slot(1))
            scale_gather.gather(2, _slot(2))

            a0_g2s.load(a_cur0, fx.Int32(0 * A_K_STEP))
            b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
            b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
            a1_g2s.load(a_cur1, fx.Int32(0 * A_K_STEP))

            a0_g2s.load(a_next0, fx.Int32(1 * A_K_STEP))
            b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
            b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)
            a1_g2s.load(a_next1, fx.Int32(1 * A_K_STEP))

            # 3 gathers + a_cur0 landed: b0/b1/a1 + the 4 next batches may fly
            wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))
            a0_frag = a_s2r.load(a_cur0)
            # b_cur0 AND b_cur1 landed: a1 + the 4 next batches may fly
            wait_barrier((3 * N_TILES_A) + (2 * N_TILES_B))
            b0_frag = b_s2r.load(b_cur0, preshuffled=True)
            b1_frag = b_s2r.load(b_cur1, preshuffled=True)

            sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0))
            sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0))
            sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)

            # a step issues a0 (NA), b0 (NB), [SEG2], b1 (NB), gather (1), a1 (NA);
            # the waits are exact: one load too many stalls on a fetch from HBM
            _MAIN_VMCNT = 2 * N_TILES_A + 2 * N_TILES_B + 1
            _SEG2_VMCNT = 2 * N_TILES_A + N_TILES_B + 1
            # step 0 follows the prologue's issue order; one count looser races
            _STEP0_VMCNT = 2 * N_TILES_A + 2 * N_TILES_B
            _STEP0_SEG2_VMCNT = 2 * N_TILES_A + N_TILES_B

            def _read_scale_thunks(kc_idx, holder):
                s = _slot(kc_idx)

                def _r(dst, ld, half, _s=s):
                    holder[dst] = ld.read_half(_s, half)

                return [
                    lambda: _r(0, a_scale_ld, 0),
                    lambda: _r(1, a_scale_ld, 1),
                    lambda: _r(2, b_scale_ld, 0),
                    lambda: _r(3, b_scale_ld, 1),
                ]

            def _one_step(
                kc,
                a0f,
                b0f,
                b1f_in,
                sc,
                accs,
                bufs,
                zero_acc=False,
                top_vmcnt=None,
                seg2_vmcnt=None,
            ):
                top_vmcnt = _MAIN_VMCNT if top_vmcnt is None else top_vmcnt
                seg2_vmcnt = _SEG2_VMCNT if seg2_vmcnt is None else seg2_vmcnt
                ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
                saR0, saR1, sbC0, sbC1 = sc
                c00f, c01f, c10f, c11f = accs
                kc_i = fx.Int32(kc)

                _a1 = [None] * N_TILES_A
                _a0n = [None] * N_TILES_A
                _b0n = [None] * N_TILES_B
                _b1n = [None] * N_TILES_B
                ak = (kc_i + fx.Int32(2)) * fx.Int32(A_K_STEP)
                bk = (kc_i + fx.Int32(2)) * fx.Int32(B_K_STEP)
                a0_off = ak
                a1_off = ak
                b0_off = fx.Int32(B0_gl_offset) + bk
                b1_off = fx.Int32(B1_gl_offset) + bk

                _scn = [None, None, None, None]
                _rd_scn = _read_scale_thunks(kc_i + 1, _scn)
                _gk = _min(kc_i + fx.Int32(3), fx.Int32(K_ITERS - 1))
                _sc_gather = _gather_scale_thunks(_gk, _slot(_gk))

                wait_barrier(top_vmcnt)
                il = (
                    _riffle(
                        _g2s_thunks(a0_g2s, ac0, a0_off, N_TILES_A),
                        _s2r_thunks(a_s2r, ac1, _a1, N_TILES_A, False),
                    )
                    + _rd_scn[:2]
                )
                c00f = mfma.call(
                    a0f, b0f, c00f, saR0, sbC0, interleave=il, zero_acc=zero_acc
                )

                il = _riffle(_g2s_thunks(b_g2s, bc0, b0_off, N_TILES_B), _rd_scn[2:])
                c01f = mfma.call(
                    a0f, b1f_in, c01f, saR0, sbC1, interleave=il, zero_acc=zero_acc
                )
                a1f = _a1

                wait_barrier(seg2_vmcnt)
                il = (
                    _riffle(
                        _g2s_thunks(b_g2s, bc1, b1_off, N_TILES_B),
                        _s2r_thunks(a_s2r, an0, _a0n, N_TILES_A, False),
                    )
                    + _sc_gather
                )
                c10f = mfma.call(
                    a1f, b0f, c10f, saR1, sbC0, interleave=il, zero_acc=zero_acc
                )
                a0nf = _a0n

                il = _riffle(
                    _g2s_thunks(a1_g2s, ac1, a1_off, N_TILES_A),
                    _s2r_thunks(b_s2r, bn0, _b0n, N_TILES_B, True)
                    + _s2r_thunks(b_s2r, bn1, _b1n, N_TILES_B, True),
                )
                c11f = mfma.call(
                    a1f, b1f_in, c11f, saR1, sbC1, interleave=il, zero_acc=zero_acc
                )
                b0nf = _b0n
                b1nf = _b1n

                sc_next = (_scn[0], _scn[1], _scn[2], _scn[3])
                new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
                return a0nf, b0nf, b1nf, sc_next, (c00f, c01f, c10f, c11f), new_bufs

            bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

            def _swap_bufs(bufs):
                ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
                return (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)

            n_a = 2 * N_TILES_A
            n_b = 2 * N_TILES_B
            n_ga = N_TILES_A // _FP4_PACK
            n_gb = N_TILES_B // _FP4_PACK
            n_sc = 2 * n_ga + 2 * n_gb
            _R = fx.as_ir_value

            def _flat_sc(sc):
                saR0, saR1, sbC0, sbC1 = sc
                return (
                    [_R(v) for v in saR0]
                    + [_R(v) for v in saR1]
                    + [_R(v) for v in sbC0]
                    + [_R(v) for v in sbC1]
                )

            def _unflat_sc(flat):
                o = 0
                saR0 = list(flat[o : o + n_ga])
                o += n_ga
                saR1 = list(flat[o : o + n_ga])
                o += n_ga
                sbC0 = list(flat[o : o + n_gb])
                o += n_gb
                sbC1 = list(flat[o : o + n_gb])
                o += n_gb
                return (saR0, saR1, sbC0, sbC1)

            _accs0 = (
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
            )
            a0f, b0f, b1f, sc, accs, _ = _one_step(
                0,
                a0_frag,
                b0_frag,
                b1_frag,
                sc0,
                _accs0,
                bufs0,
                zero_acc=True,
                top_vmcnt=_STEP0_VMCNT,
                seg2_vmcnt=_STEP0_SEG2_VMCNT,
            )
            a0f, b0f, b1f, sc, accs, _ = _one_step(
                1, a0f, b0f, b1f, sc, accs, _swap_bufs(bufs0)
            )

            init_state = (
                _flat_frag(a0f)
                + _flat_frag(b0f)
                + _flat_frag(b1f)
                + _flat_sc(sc)
                + [_R(x) for x in accs[0]]
                + [_R(x) for x in accs[1]]
                + [_R(x) for x in accs[2]]
                + [_R(x) for x in accs[3]]
            )
            for kk, state in range(2, K_ITERS - 2, UNROLL, init=init_state):  # type: ignore[call-overload]
                off = 0
                a0f = _unflat_frag(state[off : off + n_a], N_TILES_A)
                off += n_a
                b0f = _unflat_frag(state[off : off + n_b], N_TILES_B)
                off += n_b
                b1f = _unflat_frag(state[off : off + n_b], N_TILES_B)
                off += n_b
                sc = _unflat_sc(state[off : off + n_sc])
                off += n_sc
                c00f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c01f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c10f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c11f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                accs = (c00f, c01f, c10f, c11f)

                bufs = bufs0
                for u in range_constexpr(UNROLL):
                    a0f, b0f, b1f, sc, accs, bufs = _one_step(
                        kk + u, a0f, b0f, b1f, sc, accs, bufs
                    )

                new_state = (
                    _flat_frag(a0f)
                    + _flat_frag(b0f)
                    + _flat_frag(b1f)
                    + _flat_sc(sc)
                    + [_R(x) for x in accs[0]]
                    + [_R(x) for x in accs[1]]
                    + [_R(x) for x in accs[2]]
                    + [_R(x) for x in accs[3]]
                )
                state = yield new_state

            off = 0
            a0_frag = _unflat_frag(state[off : off + n_a], N_TILES_A)
            off += n_a
            b0_frag = _unflat_frag(state[off : off + n_b], N_TILES_B)
            off += n_b
            b1_frag = _unflat_frag(state[off : off + n_b], N_TILES_B)
            off += n_b
            sc = _unflat_sc(state[off : off + n_sc])
            off += n_sc
            c00_frag = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c01_frag = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c10_frag = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c11_frag = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS

            # Tail step K_ITERS - 2
            saR0, saR1, sbC0, sbC1 = sc
            _scn = [None, None, None, None]
            _rd_scn = _read_scale_thunks(fx.Int32(K_ITERS - 1), _scn)
            _a1 = [None] * N_TILES_A
            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
            il = _s2r_thunks(a_s2r, a_cur1, _a1, N_TILES_A, False) + _rd_scn
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, saR0, sbC0, interleave=il)
            a1_frag = _a1
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, saR0, sbC1)
            _a0n = [None] * N_TILES_A
            _b0n = [None] * N_TILES_B
            _b1n = [None] * N_TILES_B
            wait_barrier(1 * N_TILES_A)
            il = (
                _s2r_thunks(a_s2r, a_next0, _a0n, N_TILES_A, False)
                + _s2r_thunks(b_s2r, b_next0, _b0n, N_TILES_B, True)
                + _s2r_thunks(b_s2r, b_next1, _b1n, N_TILES_B, True)
            )
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, saR1, sbC0, interleave=il)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, saR1, sbC1)
            a0_frag = _a0n
            b0_frag = _b0n
            b1_frag = _b1n

            # Tail step K_ITERS - 1 (its A half 1 sits in a_next1; the scf.if
            # rewriter must not see the outer buffer names reassigned here).
            _a1 = [None] * N_TILES_A
            wait_barrier(0)
            saR0, saR1, sbC0, sbC1 = (_scn[0], _scn[1], _scn[2], _scn[3])
            il = _s2r_thunks(a_s2r, a_next1, _a1, N_TILES_A, False)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, saR0, sbC0, interleave=il)
            a1_frag = _a1
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, saR0, sbC1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, saR1, sbC0)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, saR1, sbC1)

            # ---- epilogue: swiglu-OAI + MXFP4 quant, sorted-row output ----
            out_rsrc = buffer_ops.create_buffer_resource(
                OUT_Q,
                max_size=False,
                num_records_bytes=num_m_blocks * (BLOCK_M * I_BYTES),
            )
            osc_rsrc = buffer_ops.create_buffer_resource(
                OUT_scale,
                max_size=False,
                num_records_bytes=num_m_blocks * (BLOCK_M * SCALE_COLS_OUT),
            )
            g = lane_id // 16
            r16 = lane_id % 16
            # gate column (in I units) this wave starts at; scale col group of it
            col_base = tile_j * LDS_BLOCK_N + wave_j * (N_TILES_B * 16)
            colgrp_base = col_base // 32

            def _epilogue(c_gate, c_up, base_row):
                """One quadrant pair (rows base_row + ti*16 + r16, wave's 64 gate
                cols)."""
                for p in range_constexpr(N_TILES_B // 2):  # 32-col group p
                    colgrp = colgrp_base + p
                    sc_in_block = (colgrp % 4) * 64 + r16 * 4 + ((colgrp % 8) // 4) * 2
                    e8m0_of_ti = []
                    for ti in range_constexpr(N_TILES_A):
                        gv = Vec(c_gate[mfma.idx(ti, 2 * p)])
                        gw = Vec(c_gate[mfma.idx(ti, 2 * p + 1)])
                        uv = Vec(c_up[mfma.idx(ti, 2 * p)])
                        uw = Vec(c_up[mfma.idx(ti, 2 * p + 1)])
                        h = [
                            _swiglu_oai(fx.Float32(gv[v]), fx.Float32(uv[v]))
                            for v in range_constexpr(4)
                        ] + [
                            _swiglu_oai(fx.Float32(gw[v]), fx.Float32(uw[v]))
                            for v in range(4)
                        ]
                        h, amax = _quant_prep_fp4(h)
                        # the 4 lanes {L, L^16, L^32, L^48} hold the same row
                        amax = _fmax(amax, amax.shuffle_xor(16, 64))
                        amax = _fmax(amax, amax.shuffle_xor(32, 64))
                        e8m0 = _e8m0_roundup_fp4(amax)
                        e8m0_of_ti.append(e8m0)
                        scale_f = _as_f32(e8m0 << 23)
                        pa = _cvt_pk_fp4(fx.Int32(0), h[0], h[1], scale_f, 0)
                        pa = _cvt_pk_fp4(pa, h[2], h[3], scale_f, 1)
                        pb = _cvt_pk_fp4(fx.Int32(0), h[4], h[5], scale_f, 0)
                        pb = _cvt_pk_fp4(pb, h[6], h[7], scale_f, 1)
                        pa, pb = _permlane16_swap(pa, pb)
                        dword = pa | (pb << 16)
                        row = base_row + ti * 16 + r16
                        # after the swap lane group g holds tile (2p + g%2), cols
                        # (g//2)*8 .. +8
                        col = col_base + (2 * p + (g % 2)) * 16 + (g // 2) * 8
                        buffer_ops.buffer_store(
                            dword,
                            out_rsrc,
                            row * I_BYTES + col // 2,
                            offset_is_bytes=True,
                        )
                    # scales: rows ti (h=0) and ti+1 (h=1) of the same 32-row group ->
                    # i16
                    for tp in range_constexpr(N_TILES_A // 2):
                        row32 = (base_row // 32) + tp
                        blk = row32 * OUT_SC_BLOCKS_PER_ROW32 + colgrp // 8
                        pair = e8m0_of_ti[2 * tp] | (e8m0_of_ti[2 * tp + 1] << 8)
                        pair16 = fx.Int32(pair).to(fx.Int16)
                        buffer_ops.buffer_store(
                            pair16,
                            osc_rsrc,
                            blk * 256 + sc_in_block,
                            offset_is_bytes=True,
                        )

            row_r0 = m_base + wave_i * (N_TILES_A * 16)
            row_r1 = row_r0 + LDS_BLOCK_M
            _epilogue(c00_frag, c01_frag, row_r0)
            _epilogue(c10_frag, c11_frag, row_r1)

    @flyc.jit
    def launch_gemm1(
        A: fx.Tensor,
        W13: fx.Tensor,
        OUT_Q: fx.Tensor,
        A_scale: fx.Tensor,
        W13_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        a_scale_bytes: fx.Int32,
        tile_map_t: fx.Tensor,
        grid_size: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = grid_size
        kernel_gemm1(
            A,
            W13,
            OUT_Q,
            A_scale,
            W13_scale,
            OUT_scale,
            sorted_ids,
            sorted_expert_ids,
            n_tokens,
            num_m_blocks,
            a_scale_bytes,
            tile_map_t,
            grid_size,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1
