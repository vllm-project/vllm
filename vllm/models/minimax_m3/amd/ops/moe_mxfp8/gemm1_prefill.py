# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill MoE gate/up GEMM (fp8 x fp8, MXFP8 scales) on sorted rows, with
SwiGLU-OAI and the MXFP8 quant of its output (the gemm2 input).

Layouts (bytes):
  A        [n_tokens, H]                   per-token fp8 (aiter per_1x32 quant)
  A_scale  [pad32(max_sorted), H/32]       sorted rows, e8m0-shuffled
  W13      [E, I/16, 2, H/64, 4, 16, 16]   shuffle_weight(is_guinterleave, gate_up)
  W13_sc   [E*I/16, H/256, 4, 16] dwords   shuffle_scale(..., True, True)
  OUT_Q    [num_m_blocks*BLOCK_M, I]       sorted rows, fp8 e4m3
  OUT_sc   [pad32(num_m_blocks*BLOCK_M), I/32]  sorted rows, e8m0-shuffled
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.epilogue import (
    _cvt_pk_fp8,
    _maxf_nn,
    _undef_i32,
    _v2i32,
)
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.loaders import (
    _N_WAVES,
    _SCALE_A_REGION,
    _SCALE_B_REGION,
    _SCALE_LDS_BYTES,
    _SCALE_SLOTS,
    G2SLoaderAsm,
    S2RLoader128B,
    ScaleGatherMoE,
    ScaleLoaderLDS,
    _as_f32,
    _bits,
    _Buf,
    _divmod_nonneg,
    _flat_frag,
    _fmax,
    _g2s_thunks,
    _min,
    _permlane16_swap,
    _riffle,
    _s2r_thunks,
    _swiglu_oai,
    _swizzled_col,
    _unflat_frag,
    wait_barrier,
)

BLOCK_K = 128  # K bytes per pipeline step
# LLVM does not see the inline-asm MFMA -> v_accvgpr_read hazard and may read a final
# accumulator too early (BLOCK_M 256 lost the last K-step). The last step releases
# each group through ``hold`` one group later; 24 wait states cover one MFMA.
_MFMA_RESULT_WAIT = "s_nop 7\n s_nop 7\n s_nop 7"


def _e8m0_roundup_fp8(amax):
    """Biased e8m0 of ceil_pow2(amax / 448), at least 1 (0 would give NaN codes)."""
    u = _bits(amax * fx.Float32(1.0 / 448.0))
    e = (u >> 23) & fx.Int32(0xFF)
    bump = ((u & fx.Int32(0x7FFFFF)) != fx.Int32(0)) & (e < fx.Int32(0xFF))
    e = bump.select(e + fx.Int32(1), e)
    return (e < fx.Int32(1)).select(fx.Int32(1), e)


def _pack8(h0, h1):
    """K [16*klane, +16) and [64 + 16*klane, +16) -> one 8 x i32 MFMA operand."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements(
        [a[k] for k in range(4)] + [b[k] for k in range(4)], fx.Int32
    )


class Mfma16x16x128Fp8:
    """Scaled fp8 MFMAs over an (n_tiles_a x n_tiles_b) quadrant, AGPR accumulators."""

    def __init__(self, n_tiles_a, n_tiles_b):
        assert n_tiles_a % 2 == 0
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
        self,
        a,
        b,
        c,
        sa,
        sb,
        gu,
        k2,
        interleave=None,
        zero_acc=False,
        late=None,
        late_start=4,
        sb_index=None,
    ):
        """``a[i]`` / ``b[j]``: [half0, half1]; ``sa[i // 2]``: A scale dword of a
        32-row group; ``sb_index(j, gu) -> (dword, byte)`` of the B scale.
        ``interleave`` / ``late``: thunks spread over the MFMAs."""
        if sb_index is None:
            sb_index = lambda j, gu: (j, gu)  # noqa: E731
        thunks = list(interleave) if interleave else []
        late_thunks = list(late) if late else []
        order = self._order()
        n_mfma = len(order)
        by_slot = {}  # type: dict[int, list]
        for t, th in enumerate(thunks):
            by_slot.setdefault((t * n_mfma) // len(thunks), []).append(th)
        n_late_slots = max(n_mfma - late_start, 1)
        for t, th in enumerate(late_thunks):
            by_slot.setdefault(
                late_start + (t * n_late_slots) // len(late_thunks), []
            ).append(th)
        for m, (i, j) in enumerate(order):
            a_op = _pack8(a[i][0], a[i][1])
            b_op = _pack8(b[j][0], b[j][1])
            acc = None if zero_acc else c[self.idx(i, j)]
            sbj, sbb = sb_index(j, gu)
            c[self.idx(i, j)] = self._mfma_agpr(
                a_op, b_op, acc, sa[i // 2], sb[sbj], i % 2, sbb, k2
            )
            for th in by_slot.get(m, ()):
                th()
        for m in range(
            n_mfma, n_mfma + max(len(thunks), len(late_thunks)) + late_start + 1
        ):
            for th in by_slot.get(m, ()):
                th()
        return c

    def hold(self, c, wait=""):
        """Pins ``c`` here (after asm ``wait``): reads cannot move above this point."""
        if wait:
            _llvm.inline_asm(None, [], wait, "", has_side_effects=True)
        return [
            _llvm.inline_asm(
                self.res_ty, [fx.as_ir_value(v)], "", "=a,0", has_side_effects=True
            )
            for v in c
        ]

    def _mfma_agpr(self, a_op, b_op, acc, sa_v, sb_v, ia, jb, k2):
        # (B, A) order: lane L holds C[L % 16, 4 * (L // 16) : +4]
        a_op, b_op = b_op, a_op
        sa_v, sb_v = sb_v, sa_v
        ia, jb = jb, ia
        opsel = f"op_sel:[{ia},{jb},0]"
        opsel_hi = f"op_sel_hi:[{k2},{k2},0]"
        src2 = "$0" if acc is not None else "0"
        asm = (
            f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 "
            f"{opsel} {opsel_hi} cbsz:0 blgp:0"
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


def compile_moe_gemm1(
    *,
    H: int,
    I: int,  # noqa: E741
    E: int,
    BLOCK_M: int = 128,
):
    """``BLOCK_M``: the ``moe_sorting`` block size of the inputs (128 or 256)."""
    K = H
    BLOCK_K_BYTES = BLOCK_K
    BLOCK_N = 256  # 128 gate + 128 up cols
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_TILES_A = LDS_BLOCK_M // 2 // 16  # 16-row tiles per wave per LDS half
    N_TILES_B = LDS_BLOCK_N // 2 // 16

    assert BLOCK_M in (128, 256)
    assert K % 256 == 0 and I % LDS_BLOCK_N == 0
    K_ITERS = K // BLOCK_K
    UNROLL = 4 if (K_ITERS - 4) % 4 == 0 else 2
    assert K_ITERS >= 4 and (K_ITERS - 4) % UNROLL == 0, K_ITERS
    N_ACCUMS = N_TILES_A * N_TILES_B
    K_BYTES = K
    N_BITS = (I // 128 - 1).bit_length()
    N0 = I // 16  # 16-column (gate + up) groups per expert

    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES
    A_BUFS = 4 * a_lds_size
    LDS_TILES_BYTES = A_BUFS + 4 * b_lds_size

    # scales: A in 32-row groups, W13 in 16-column groups (256 B = 8 K-groups)
    A_WAVE_GROUPS = N_TILES_A // 2  # groups per wave per LDS half
    A_HALF_GROUPS = LDS_BLOCK_M // 32
    B_WAVE_GROUPS = 4  # consecutive groups per wave quarter
    B_HALF_GROUPS = 2
    SCALE_COLS_OUT = I // 32
    OUT_SC_BLOCKS_PER_ROW32 = SCALE_COLS_OUT // 8
    W_BYTES = (2 * I) * K
    assert W_BYTES <= 0xFFFFFFFF, "buffer resources address 4 GB"

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
        # tile_map[grid_size] = valid entries, split evenly over the 8 XCDs
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
        tile_i = entry >> N_BITS
        tile_j = entry & ((1 << N_BITS) - 1)
        block_valid = entry >= 0
        expert = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc, tile_i, vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        m_base = tile_i * BLOCK_M

        if block_valid:
            wave_i = wave_id // 2
            wave_j = wave_id % 2

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
                    # padded rows: tok == n_tokens, out of bounds -> zeros
                    tok = sid & fx.Int32(0x00FFFFFF)
                    offs.append(tok * fx.Int32(K_BYTES) + _swizzled_col(row, col))
                return offs

            gl_off_a0 = _a_gather_offsets(0)
            gl_off_a1 = _a_gather_offsets(1)

            # B: LDS row r of half gu (gate 0 / up 1) is in W13 block n0 * 2 + gu,
            # n0 = tile_j * 8 + r // 16
            weight_group0 = tile_j * fx.Int32(LDS_BLOCK_N // 16)
            group0 = expert * fx.Int32(N0) + weight_group0

            def _b_offsets(gu):
                offs = []
                for rnd in range_constexpr(N_TILES_B):
                    row = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id // 8) * 16
                    blk = (weight_group0 + fx.Int32(row // 16)) * fx.Int32(
                        2
                    ) + fx.Int32(gu)
                    offs.append(
                        blk * fx.Int32(K_BYTES * 16)
                        + fx.Int32(
                            (row % 16) * 16
                            + (col // 64) * 1024
                            + ((col % 64) // 16) * 256
                            + (col % 16)
                        )
                    )
                return offs

            gl_off_b0 = _b_offsets(0)
            gl_off_b1 = _b_offsets(1)
            A_K_STEP = BLOCK_K_BYTES
            B_K_STEP = 2 * 1024  # 128 B of K = two 1 KB blocks

            mfma = Mfma16x16x128Fp8(N_TILES_A, N_TILES_B)

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
            scale_gather.set_wave_base(m_base, group0 * fx.Int32(32))
            a_scale_ld = ScaleLoaderLDS(
                N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION
            )
            b_scale_ld = ScaleLoaderLDS(
                N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION
            )

            def _slot(g):
                return fx.Int32(g) % fx.Int32(_SCALE_SLOTS)

            a_rsrc = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=n_tokens * K_BYTES
            )
            b_rsrc = buffer_ops.create_buffer_resource(
                W13,
                max_size=False,
                num_records_bytes=W_BYTES,
                base_byte_offset=fx.Int64(expert) * W_BYTES,
            )
            a0_g2s = G2SLoaderAsm(a_rsrc, gl_off_a0, N_TILES_A, wave_id)
            a1_g2s = G2SLoaderAsm(a_rsrc, gl_off_a1, N_TILES_A, wave_id)
            b0_g2s = G2SLoaderAsm(b_rsrc, gl_off_b0, N_TILES_B, wave_id)
            b1_g2s = G2SLoaderAsm(b_rsrc, gl_off_b1, N_TILES_B, wave_id)
            for ld in (a0_g2s, a1_g2s, b0_g2s, b1_g2s):
                ld.set_wave_base(_base_ptr)
            a_s2r = S2RLoader128B(wave_i, N_TILES_A)
            b_s2r = S2RLoader128B(wave_j, N_TILES_B)

            # scale block g covers steps 2g, 2g + 1; block g >= 2 is gathered on
            # step 2g - 3
            scale_gather.gather(0, _slot(0))
            scale_gather.gather(1, _slot(1))

            a0_g2s.load(a_cur0, fx.Int32(0 * A_K_STEP))
            b0_g2s.load(b_cur0, fx.Int32(0 * B_K_STEP))
            b1_g2s.load(b_cur1, fx.Int32(0 * B_K_STEP))
            a1_g2s.load(a_cur1, fx.Int32(0 * A_K_STEP))

            a0_g2s.load(a_next0, fx.Int32(1 * A_K_STEP))
            b0_g2s.load(b_next0, fx.Int32(1 * B_K_STEP))
            b1_g2s.load(b_next1, fx.Int32(1 * B_K_STEP))
            a1_g2s.load(a_next1, fx.Int32(1 * A_K_STEP))

            # gathers + a_cur0 landed: b0/b1/a1 + the 4 next batches may fly
            wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))
            a0_frag = a_s2r.load(a_cur0)
            # b_cur0 AND b_cur1 landed: a1 + the 4 next batches may fly
            wait_barrier((3 * N_TILES_A) + (2 * N_TILES_B))
            b0_frag = b_s2r.load(b_cur0, preshuffled=True)
            b1_frag = b_s2r.load(b_cur1, preshuffled=True)

            sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0))
            sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0))
            sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0 + sc0_sbC1)

            # Step kc issues a0, b0, [SEG2] b1, the scale gather (odd kc), a1 for step
            # kc + 2. Top wait: step kc - 2 complete; SEG2 wait: a0 / b0 / b1 of step
            # kc - 1 landed. Step kc - 1 gathered iff its parity k2 == 0.
            def _top_vmcnt(k2):
                return 2 * N_TILES_A + 2 * N_TILES_B + (1 - k2)

            def _seg2_vmcnt(k2):
                return 2 * N_TILES_A + N_TILES_B + (1 - k2)

            # step 0 follows the prologue, which has no gather after a_cur1
            _STEP0_VMCNT = 2 * N_TILES_A + 2 * N_TILES_B
            _STEP0_SEG2_VMCNT = 2 * N_TILES_A + N_TILES_B

            def _read_scale_thunks(kc_idx, holder):
                s = _slot(fx.Int32(kc_idx) // fx.Int32(2))

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
                k2,
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
                """``kc``: step index; ``k2``: its parity as a Python int."""
                kc_i = fx.Int32(kc)
                top_vmcnt = _top_vmcnt(k2) if top_vmcnt is None else top_vmcnt
                seg2_vmcnt = _seg2_vmcnt(k2) if seg2_vmcnt is None else seg2_vmcnt
                ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
                saR0, saR1, sbC = sc
                c00f, c01f, c10f, c11f = accs

                _a1 = [None] * N_TILES_A
                _a0n = [None] * N_TILES_A
                _b0n = [None] * N_TILES_B
                _b1n = [None] * N_TILES_B
                a_off = (kc_i + fx.Int32(2)) * fx.Int32(A_K_STEP)
                b_off = (kc_i + fx.Int32(2)) * fx.Int32(B_K_STEP)

                _scn = [None, None, None, None]  # type: list
                _rd_scn = _read_scale_thunks(kc_i + fx.Int32(1), _scn)
                # odd steps only: a list repeat, as an if would become an scf dispatch
                _gk = _min(
                    (kc_i + fx.Int32(3)) // fx.Int32(2), fx.Int32(K_ITERS // 2 - 1)
                )
                _sc_gather = [lambda: scale_gather.gather(_gk, _slot(_gk))] * k2

                wait_barrier(top_vmcnt)
                il = (
                    _riffle(
                        _g2s_thunks(a0_g2s, ac0, a_off, N_TILES_A),
                        _s2r_thunks(a_s2r, ac1, _a1, N_TILES_A, False),
                    )
                    + _rd_scn[:2]
                )
                c00f = mfma.call(
                    a0f, b0f, c00f, saR0, sbC, 0, k2, interleave=il, zero_acc=zero_acc
                )

                il = _riffle(_g2s_thunks(b0_g2s, bc0, b_off, N_TILES_B), _rd_scn[2:])
                c01f = mfma.call(
                    a0f,
                    b1f_in,
                    c01f,
                    saR0,
                    sbC,
                    1,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                )
                a1f = _a1

                wait_barrier(seg2_vmcnt)
                il = (
                    _riffle(
                        _g2s_thunks(b1_g2s, bc1, b_off, N_TILES_B),
                        _s2r_thunks(a_s2r, an0, _a0n, N_TILES_A, False),
                    )
                    + _sc_gather
                )
                c10f = mfma.call(
                    a1f, b0f, c10f, saR1, sbC, 0, k2, interleave=il, zero_acc=zero_acc
                )
                a0nf = _a0n

                il = _riffle(
                    _g2s_thunks(a1_g2s, ac1, a_off, N_TILES_A),
                    _s2r_thunks(b_s2r, bn0, _b0n, N_TILES_B, True)
                    + _s2r_thunks(b_s2r, bn1, _b1n, N_TILES_B, True),
                )
                c11f = mfma.call(
                    a1f,
                    b1f_in,
                    c11f,
                    saR1,
                    sbC,
                    1,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                )
                b0nf = _b0n
                b1nf = _b1n

                sc_next = (_scn[0], _scn[1], _scn[2] + _scn[3])
                new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
                return a0nf, b0nf, b1nf, sc_next, (c00f, c01f, c10f, c11f), new_bufs

            bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

            def _swap_bufs(bufs):
                ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
                return (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)

            n_a = 2 * N_TILES_A
            n_b = 2 * N_TILES_B
            n_ga = N_TILES_A // 2
            n_sc = 2 * n_ga + N_TILES_B
            _R = fx.as_ir_value

            def _flat_sc(sc):
                saR0, saR1, sbC = sc
                return (
                    [_R(v) for v in saR0] + [_R(v) for v in saR1] + [_R(v) for v in sbC]
                )

            def _unflat_sc(flat):
                o = 0
                saR0 = list(flat[o : o + n_ga])
                o += n_ga
                saR1 = list(flat[o : o + n_ga])
                o += n_ga
                sbC = list(flat[o : o + N_TILES_B])
                return (saR0, saR1, sbC)

            _accs0 = tuple([None] * N_ACCUMS for _ in range(4))
            a0f, b0f, b1f, sc, accs, _ = _one_step(
                0,
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
                1, 1, a0f, b0f, b1f, sc, accs, _swap_bufs(bufs0)
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
            # FlyDSL rewrites range(init=...) to a loop carrying state.
            for kk, state in range(  # type: ignore[call-overload]
                2, K_ITERS - 2, UNROLL, init=init_state
            ):
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
                    # kk is even: kk + u has u's parity
                    a0f, b0f, b1f, sc, accs, bufs = _one_step(
                        kk + u, u % 2, a0f, b0f, b1f, sc, accs, bufs
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

            # tail step K_ITERS - 2 (even)
            saR0, saR1, sbC = sc
            _scn = [None, None, None, None]  # type: list
            _rd_scn = _read_scale_thunks(fx.Int32(K_ITERS - 1), _scn)
            _a1 = [None] * N_TILES_A
            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
            il = _s2r_thunks(a_s2r, a_cur1, _a1, N_TILES_A, False) + _rd_scn
            c00_frag = mfma.call(
                a0_frag, b0_frag, c00_frag, saR0, sbC, 0, 0, interleave=il
            )
            a1_frag = _a1
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, saR0, sbC, 1, 0)
            _a0n = [None] * N_TILES_A
            _b0n = [None] * N_TILES_B
            _b1n = [None] * N_TILES_B
            wait_barrier(1 * N_TILES_A)
            il = (
                _s2r_thunks(a_s2r, a_next0, _a0n, N_TILES_A, False)
                + _s2r_thunks(b_s2r, b_next0, _b0n, N_TILES_B, True)
                + _s2r_thunks(b_s2r, b_next1, _b1n, N_TILES_B, True)
            )
            c10_frag = mfma.call(
                a1_frag, b0_frag, c10_frag, saR1, sbC, 0, 0, interleave=il
            )
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, saR1, sbC, 1, 0)
            a0_frag = _a0n
            b0_frag = _b0n
            b1_frag = _b1n

            # tail step K_ITERS - 1 (odd; A half 1 in a_next1)
            _a1 = [None] * N_TILES_A
            wait_barrier(0)
            saR0, saR1, sbC = (_scn[0], _scn[1], _scn[2] + _scn[3])
            il = _s2r_thunks(a_s2r, a_next1, _a1, N_TILES_A, False)
            # each final group is released one group later (_MFMA_RESULT_WAIT)
            c00_frag = mfma.call(
                a0_frag, b0_frag, c00_frag, saR0, sbC, 0, 1, interleave=il
            )
            a1_frag = _a1
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, saR0, sbC, 1, 1)
            c00_frag = mfma.hold(c00_frag)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, saR1, sbC, 0, 1)
            c01_frag = mfma.hold(c01_frag)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, saR1, sbC, 1, 1)
            c10_frag = mfma.hold(c10_frag)
            c11_frag = mfma.hold(c11_frag, wait=_MFMA_RESULT_WAIT)

            out_rsrc = buffer_ops.create_buffer_resource(
                OUT_Q, max_size=False, num_records_bytes=num_m_blocks * (BLOCK_M * I)
            )
            osc_rsrc = buffer_ops.create_buffer_resource(
                OUT_scale,
                max_size=False,
                num_records_bytes=num_m_blocks * (BLOCK_M * SCALE_COLS_OUT),
            )
            g = lane_id // 16
            r16 = lane_id % 16
            col_base = tile_j * LDS_BLOCK_N + wave_j * (N_TILES_B * 16)
            colgrp_base = col_base // 32

            def _epilogue(c_gate, c_up, base_row):
                """Rows ``base_row + ti * 16 + r16`` x the wave's 64 gate columns."""
                for p in range_constexpr(N_TILES_B // 2):
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
                        amax = _maxf_nn(fx.math.absf(h[0]), fx.math.absf(h[1]))
                        for v in range_constexpr(2, 8):
                            amax = _maxf_nn(amax, fx.math.absf(h[v]))
                        # the 4 lanes {L, L^16, L^32, L^48} hold the same row
                        amax = _fmax(amax, amax.shuffle_xor(16, 64))
                        amax = _fmax(amax, amax.shuffle_xor(32, 64))
                        e8m0 = _e8m0_roundup_fp8(amax)
                        e8m0_of_ti.append(e8m0)
                        scale_f = _as_f32(e8m0 << 23)
                        da = _cvt_pk_fp8(_undef_i32(), h[0], h[1], scale_f, False)
                        da = _cvt_pk_fp8(da, h[2], h[3], scale_f, True)
                        db = _cvt_pk_fp8(_undef_i32(), h[4], h[5], scale_f, False)
                        db = _cvt_pk_fp8(db, h[6], h[7], scale_f, True)
                        da, db = _permlane16_swap(da, db)
                        row = base_row + ti * 16 + r16
                        # after the swap: tile 2p + g % 2, cols (g // 2) * 8 + [0, 8)
                        col = col_base + (2 * p + (g % 2)) * 16 + (g // 2) * 8
                        buffer_ops.buffer_store(
                            _v2i32(da, db),
                            out_rsrc,
                            row * I + col,
                            offset_is_bytes=True,
                        )
                    # the e8m0 of tiles 2tp, 2tp + 1 (one 32-row group) as one i16
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
