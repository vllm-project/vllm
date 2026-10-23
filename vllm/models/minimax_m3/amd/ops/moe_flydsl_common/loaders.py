# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL MoE loaders helpers."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
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
    """i32 in a wave-uniform SGPR, for scalar inline-asm operands."""
    raw = fx.as_ir_value(value) if not isinstance(value, _ir.Value) else value
    if raw.type != _T.i32:
        raw = fx.as_ir_value(fx.Int32(raw))
    return _rocdl.readfirstlane(_T.i32, raw)


def _swizzled_col(row, col):
    """The swizzled 128-B-row byte column ``swizzle_128`` lands (row, col) on."""
    r, c = swizzle_128(row, col)
    return c


class G2SLoaderAsm:
    """global -> LDS, 16 B per lane per step: per-lane offsets in VGPRs, the K-step
    in soffset. With ``step_bytes`` all steps share ``gl_offsets[0]`` (one VGPR) and
    advance soffset by that stride; ``soff_bytes`` is added to every soffset."""

    def __init__(
        self, rsrc, gl_offsets, n_load_steps, wave_id, step_bytes=None, soff_bytes=0
    ):
        self.rsrc = fx.as_ir_value(rsrc)
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.step_bytes = step_bytes
        self.soff_bytes = soff_bytes

    @property
    def _step_stride(self):
        # m0 (LDS byte) advance per step: 4 waves x 64 lanes x 16 B.
        return _N_WAVES * 1024

    def set_wave_base(self, base_ptr):
        # the wave-uniform LDS base, in an SGPR once
        wb = fx.Int32(fx.ptrtoint(base_ptr)) + fx.Int32(self.wave_id * 1024)
        self._wave_base_s = _rocdl.readfirstlane(_T.i32, fx.as_ir_value(wb))

    def _lds_base_sgpr(self, lds_dst):
        m0 = fx.Int32(self._wave_base_s) + fx.Int32(lds_dst.byte_off)
        return fx.as_ir_value(m0)

    def _voffset(self, step):
        if self.step_bytes is not None:
            return fx.as_ir_value(fx.Int32(self.gl_offsets[0]))
        return fx.as_ir_value(fx.Int32(self.gl_offsets[step]))

    def _emit(self, lds_dst, k_offset, step):
        # set m0 at step 0, then s_add it for the rest
        voff = self._voffset(step)
        extra = self.soff_bytes + (step * self.step_bytes if self.step_bytes else 0)
        if extra:
            k_offset = fx.Int32(k_offset) + fx.Int32(extra)
        soff = _uniform_i32(k_offset)
        stride = self._step_stride
        # s_add_u32 writes SCC: without ~{scc} a live SCC (loop exit) gets clobbered
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


class S2RLoader128B:
    """LDS -> registers: two 16-byte halves of a swizzled 128-byte row."""

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
        vec_i32 = _ir.VectorType.get([4], fx.Int32.ir_type)
        return Vec(_llvm.LoadOp(vec_i32, lds_ptr, alignment=16).result)

    def _offset(self, unit, preshuffled):
        row = self.wave_idx * (self.n_tiles * 16) + self.lane_id % 16
        if const_expr(preshuffled):
            return (row // 8) * 1024 + (row % 8) * 16 + unit * 128
        row_swz, col_swz = swizzle_128(row, unit * 16)
        return row_swz * 128 + col_swz

    def _dyn_offset(self, step, preshuffled):
        return self._offset(self.lane_id // 16 + step * 4, preshuffled)

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
        return self._vec_load_16xf8(lds_src, dyn, i * 2048).bitcast(fx.Int32)


def _flat_frag(frag):
    return [fx.as_ir_value(x) for t in frag for x in t]


def _unflat_frag(flat, n_tiles, arity=2):
    return [list(flat[arity * i : arity * i + arity]) for i in range(n_tiles)]


def _g2s_thunks(g2s, dst, gl_off, n_steps):
    return [lambda s=s: g2s.load_one(dst, gl_off, s) for s in range(n_steps)]


def _riffle(glb, lds):
    """Interleave global and LDS thunks proportionally."""
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


_FP4_PACK = 2  # pack_M = pack_N = pack_K = 2
_SCALE_QUARTER_BYTES = 1024  # one gather = 4 blocks = one wave's share of one operand
_SCALE_REGION_BYTES = 2 * _SCALE_QUARTER_BYTES  # all of A (or all of B) for one K-step
_SCALE_SLOT_BYTES = _N_WAVES * _SCALE_QUARTER_BYTES
_SCALE_SLOTS = 4
_SCALE_LDS_BYTES = _SCALE_SLOTS * _SCALE_SLOT_BYTES
_SCALE_A_REGION = 0
_SCALE_B_REGION = _SCALE_REGION_BYTES


class ScaleGatherMoE:
    """One ``buffer_load_dwordx4 ... lds`` per wave per K-step: waves 0/1 fetch A
    scales, waves 2/3 B scales (the expert's gate + up slabs). A 256 B block = 32
    rows x 8 K-blocks; lane blocks ``lane // 16`` 0/1 go to LDS half 0 (R0 / gate),
    2/3 to half 1 (R1 / up)."""

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


def _fmax(a, b):
    return (a > b).select(a, b)


def _fmin(a, b):
    return (a < b).select(a, b)


def _swiglu_oai(g, u):
    """SwiGLU-OAI, op for op aiter's stage-1 ``swiglu_mul_vec4``; the two separate
    multiplies of ``t`` matter (folding them changes the last bit)."""
    lim = fx.Float32(SWIGLU_LIMIT)
    g = _fmin(g, lim)
    u = _fmax(_fmin(u, lim), fx.Float32(-SWIGLU_LIMIT))
    t = (g * fx.Float32(SWIGLU_ALPHA)) * fx.Float32(-1.4426950408889634)
    e = fx.Float32(_rocdl.exp2(_T.f32, t.ir_value()))
    sig = fx.Float32(_rocdl.rcp(_T.f32, (fx.Float32(1.0) + e).ir_value()))
    return g * sig * (u + fx.Float32(1.0))


def _bits(f):
    return fx.Float32(f).bitcast(fx.Int32)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _permlane16_swap(d_a, d_b):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = _rocdl.permlane16_swap(
        pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False
    )
    return fx.Int32(_llvm.extractvalue(_T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(_T.i32, res, [1])
    )
