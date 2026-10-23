# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``gemm2_prefill`` on 256-row m-tiles (each wave 128 x 128, 256 AGPR accumulators):
W2 streams once per 256 sorted rows. W2 and A run two K-steps ahead (2 LDS stages) to
fit 160 KB of LDS. Same partials as the 128-row kernel.

Layouts (bytes):
  A         [num_m_blocks*BM, I]         sorted rows, fp8 (gemm1 OUT_Q)
  A_scale   [pad32(num_m_blocks*BM), I/32]  sorted rows, e8m0-shuffled (gemm1 OUT_sc)
  W2        [E, H, I]                    aiter shuffle_weight (16 x 64 K blocks)
  W2_sc     [E*H/32, I/256, 4, 16] dwords aiter shuffle_scale
  sorted_w  [num_m_blocks*BM]            f32 routing weight per sorted row
  OUT       [n_tokens*topk, H]           bf16
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.epilogue import (
    _NUM_XCDS,
    _STORE_CPOL,
    _bf16x2,
    _lds_load_vec,
    _lds_store_vec,
    _v2i32,
)
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.loaders import (
    _N_WAVES,
    _SCALE_A_REGION,
    _SCALE_B_REGION,
    _SCALE_SLOT_BYTES,
    G2SLoaderAsm,
    S2RLoader128B,
    ScaleGatherMoE,
    ScaleLoaderLDS,
    _asm_void,
    _Buf,
    _divmod_nonneg,
    _flat_frag,
    _g2s_thunks,
    _riffle,
    _s2r_thunks,
    _swizzled_col,
    _unflat_frag,
    wait_barrier,
)

from .gemm1_prefill import BLOCK_K, Mfma16x16x128Fp8

_SCALE_SLOTS2 = 4


def compile_moe_gemm2_bm256(
    *,
    H: int,
    I: int,  # noqa: E741
    E: int,
    topk: int,
    n_split: int = 2,
    sort_block_m: int = 128,
):
    """``compile_moe_gemm2`` on 256-row tiles; ``sort_block_m`` must be 256."""
    BM = 256
    BN = 256
    K = I
    K_BYTES = K
    BLOCK_K_BYTES = BLOCK_K
    EID_SHIFT = (sort_block_m // BM).bit_length() - 1
    assert sort_block_m in (BM, 2 * BM)
    N_TILES_ALL = H // BN
    assert H % BN == 0 and N_TILES_ALL % n_split == 0
    NT = N_TILES_ALL // n_split  # n-tiles per CTA
    K_ITERS = K // BLOCK_K
    assert K_ITERS in (6, 12, 24) and K % 256 == 0
    SCALE_BLOCKS = K_ITERS // 2
    LDS_BLOCK_M = BM // 2
    LDS_BLOCK_N = BN // 2
    N_TILES_A = LDS_BLOCK_M // 2 // 16
    N_TILES_B = LDS_BLOCK_N // 2 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    NB = N_TILES_B
    NA = N_TILES_A
    OUT_ROW_BYTES = H * 2
    W2_BYTES = H * K
    assert W2_BYTES <= 0xFFFFFFFF
    B_TILE_BYTES = BN * K_BYTES

    B_STAGES = 2
    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES
    A_BUFS = 4 * a_lds_size  # 2 stages x 2 halves
    LDS_TILES_BYTES = A_BUFS + 2 * B_STAGES * b_lds_size
    SCALE_LDS_BYTES2 = _SCALE_SLOTS2 * _SCALE_SLOT_BYTES
    # wave-private epilogue staging: 16 rows x the wave's 128 bf16 cols
    STG_ROW = 2 * N_TILES_B * 16 * 2  # the wave's two 128-B line segments
    STG_CH = STG_ROW // 16
    STG_ROWS = 16  # one (row half, a-tile) at a time
    STG_WAVE = STG_ROWS * STG_ROW
    STG_LDS_BYTES = _N_WAVES * STG_WAVE
    N_FLUSH = (STG_ROWS * 2) // 8  # dwordx4 stores per lane per staged block
    assert LDS_TILES_BYTES + SCALE_LDS_BYTES2 + STG_LDS_BYTES <= 160 * 1024

    A_WAVE_GROUPS = N_TILES_A // 2
    A_HALF_GROUPS = LDS_BLOCK_M // 32
    B_WAVE_GROUPS = 2  # 32-row groups per wave (64 W2 rows)
    B_HALF_GROUPS = LDS_BLOCK_N // 32
    SC_COLS = K // 32
    W2_SC_BYTES = E * H * SC_COLS

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, LDS_TILES_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, SCALE_LDS_BYTES2, 16]
        stage_lds: fx.Array[fx.Int8, STG_LDS_BYTES, 16]

    @flyc.kernel
    def kernel_gemm2_bm256(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _base_ptr = lds.all_lds.ptr
        _scale_base_ptr = lds.scale_lds.ptr
        _stg_ptr = lds.stage_lds.ptr

        a_cur0 = _Buf(_base_ptr, 0 * a_lds_size)
        a_cur1 = _Buf(_base_ptr, 1 * a_lds_size)
        a_next0 = _Buf(_base_ptr, 2 * a_lds_size)
        a_next1 = _Buf(_base_ptr, 3 * a_lds_size)
        # B: 2 stages x 2 halves
        b_st = [
            [_Buf(_base_ptr, A_BUFS + (2 * st + hf) * b_lds_size) for hf in range(2)]
            for st in range(B_STAGES)
        ]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        g4 = lane_id // 16
        r16 = lane_id % 16

        # work item (m-tile, n-chunk); consecutive m-tiles share an XCD
        nv_rsrc = buffer_ops.create_buffer_resource(
            num_valid_ids, max_size=False, num_records_bytes=4
        )
        num_valid = fx.Int32(
            buffer_ops.buffer_load(
                nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        n_work = (num_valid // fx.Int32(BM)) * fx.Int32(n_split)
        per_xcd = (n_work + fx.Int32(_NUM_XCDS - 1)) // fx.Int32(_NUM_XCDS)
        intra, xcd = _divmod_nonneg(fx.block_idx.x, _NUM_XCDS)
        work = xcd * per_xcd + intra
        block_valid = (intra < per_xcd) & (work < n_work)
        work_safe = block_valid.select(work, fx.Int32(0))
        tile_i, chunk = _divmod_nonneg(work_safe, n_split)
        m_base = tile_i * BM
        eid_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4
        )
        expert = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc,
                tile_i >> EID_SHIFT,
                vec_width=1,
                dtype=fx.Int32,
                is_scalar=True,
            )
        )
        ids_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BM * 4)
        )
        if const_expr(EID_SHIFT > 0):
            # a 256-sort pads each expert: a 128-row tile starting with padding is empty
            first_sid = fx.Int32(
                buffer_ops.buffer_load(
                    ids_rsrc, m_base, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            block_valid = block_valid & ((first_sid & fx.Int32(0x00FFFFFF)) < n_tokens)
        chunk_n0 = chunk * NT
        # rotate the n-tile sweep inside same-expert runs: neighbours find W2 in L2
        ROT_STRIDE, ROT_GATE = 2, 3
        _d = fx.Int32(ROT_GATE)
        _lo_ok = tile_i >= _d
        _hi_ok = tile_i + _d < num_m_blocks
        _e_lo = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc,
                _lo_ok.select(tile_i - _d, fx.Int32(0)) >> EID_SHIFT,
                vec_width=1,
                dtype=fx.Int32,
                is_scalar=True,
            )
        )
        _e_hi = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc,
                _hi_ok.select(tile_i + _d, fx.Int32(0)) >> EID_SHIFT,
                vec_width=1,
                dtype=fx.Int32,
                is_scalar=True,
            )
        )
        rot_on = (_lo_ok & (_e_lo == expert)) | (_hi_ok & (_e_hi == expert))
        nt_rot = rot_on.select(
            _divmod_nonneg(tile_i * fx.Int32(ROT_STRIDE), NT)[1], fx.Int32(0)
        )

        def _pn(nt):
            """Sweep-order n-tile -> n-tile of the chunk."""
            x = nt + nt_rot
            return (x >= fx.Int32(NT)).select(x - fx.Int32(NT), x)

        def _n_glob(nt):
            """Global n-tile of sweep-order ``nt``, clamped to the last one."""
            ntc = (nt >= fx.Int32(NT)).select(fx.Int32(NT - 1), nt)
            return chunk_n0 + _pn(ntc)

        if block_valid:
            sw_rsrc = buffer_ops.create_buffer_resource(
                sorted_weights,
                max_size=False,
                num_records_bytes=num_m_blocks * (BM * 4),
            )
            a_rsrc = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=num_m_blocks * (BM * K_BYTES)
            )
            b_rsrc = buffer_ops.create_buffer_resource(
                W2,
                max_size=False,
                num_records_bytes=W2_BYTES,
                base_byte_offset=fx.Int64(expert) * W2_BYTES,
            )
            # padded rows (tok == n_tokens) fall outside the resource
            out_rows = n_tokens * fx.Int32(topk)
            out_rsrc = buffer_ops.create_buffer_resource(
                OUT, max_size=False, num_records_bytes=out_rows * OUT_ROW_BYTES
            )

            def _a_offsets(half):
                offs = []
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id % 8) * 16
                    grow = m_base + half * LDS_BLOCK_M + row
                    offs.append(grow * fx.Int32(K_BYTES) + _swizzled_col(row, col))
                return offs

            # rounds and halves move whole 32-row groups (same swizzle): one per-lane
            # base plus soffset strides
            gl_off_a = _a_offsets(0)[:1]
            A_RND_BYTES = _N_WAVES * 8 * K_BYTES
            A_HALF_BYTES = LDS_BLOCK_M * K_BYTES

            # W2 offsets inside an n-tile; the n-tile goes in soffset
            e_row0 = expert * fx.Int32(H)

            def _b_offsets(hf):
                offs = []
                for rnd in range_constexpr(N_TILES_B):
                    row = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id // 8) * 16
                    wrow = fx.Int32(hf * LDS_BLOCK_N + row)
                    offs.append(
                        (wrow // fx.Int32(16)) * fx.Int32(K_BYTES * 16)
                        + fx.Int32(
                            (row % 16) * 16
                            + (col // 64) * 1024
                            + ((col % 64) // 16) * 256
                            + (col % 16)
                        )
                    )
                return offs

            gl_off_b = _b_offsets(0)[:1]
            B_RND_BYTES = (_N_WAVES * 8 // 16) * K_BYTES * 16
            B_HALF_BYTES = (LDS_BLOCK_N // 16) * K_BYTES * 16
            A_K_STEP = BLOCK_K_BYTES
            B_K_STEP = 2 * 1024

            mfma = Mfma16x16x128Fp8(N_TILES_A, N_TILES_B)
            sb_index = lambda j, hf: (j // 2, j % 2)  # noqa: E731  (32-row groups)

            scale_gather = ScaleGatherMoE(
                A_scale,
                W2_scale,
                K,
                lane_id,
                wave_id,
                _scale_base_ptr,
                num_m_blocks * (BM * SC_COLS),
                W2_SC_BYTES,
                A_WAVE_GROUPS,
                A_HALF_GROUPS,
                B_WAVE_GROUPS,
                B_HALF_GROUPS,
            )

            def _set_scale_tile(nt):
                scale_gather.set_wave_base(m_base, e_row0 + _n_glob(nt) * fx.Int32(BN))

            a_scale_ld = ScaleLoaderLDS(
                N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION
            )
            b_scale_ld = ScaleLoaderLDS(
                N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION
            )

            def _slot(nt, g):
                return (fx.Int32(nt) * fx.Int32(SCALE_BLOCKS) + fx.Int32(g)) % fx.Int32(
                    _SCALE_SLOTS2
                )

            a0_g2s = G2SLoaderAsm(a_rsrc, gl_off_a, N_TILES_A, wave_id, A_RND_BYTES)
            a1_g2s = G2SLoaderAsm(
                a_rsrc, gl_off_a, N_TILES_A, wave_id, A_RND_BYTES, A_HALF_BYTES
            )
            b0_g2s = G2SLoaderAsm(b_rsrc, gl_off_b, N_TILES_B, wave_id, B_RND_BYTES)
            b1_g2s = G2SLoaderAsm(
                b_rsrc, gl_off_b, N_TILES_B, wave_id, B_RND_BYTES, B_HALF_BYTES
            )
            for ld in (a0_g2s, a1_g2s, b0_g2s, b1_g2s):
                ld.set_wave_base(_base_ptr)
            a_s2r = S2RLoader128B(wave_i, N_TILES_A)
            b_s2r = S2RLoader128B(wave_j, N_TILES_B)

            def _orow(sid):
                tok = sid & fx.Int32(0x00FFFFFF)  # padded: tok == n_tokens -> OOB
                slot = (sid >> 24) & fx.Int32(0xFF)
                return tok * fx.Int32(topk) + slot

            # routing weights of the lane's rows; output offsets of the rows it
            # flushes (store k: staged row 8 (k % 2) + lane // 8, segment k // 2)
            wave_row0 = m_base + wave_i * (N_TILES_A * 16)
            row_w = []
            for h in range_constexpr(2):
                for ti in range_constexpr(N_TILES_A):
                    row = wave_row0 + fx.Int32(h * LDS_BLOCK_M + ti * 16) + r16
                    row_w.append(
                        fx.Float32(
                            buffer_ops.buffer_load(
                                sw_rsrc, row, vec_width=1, dtype=fx.Float32
                            )
                        )
                    )
            flush_off = []
            for h in range_constexpr(2):
                for ti in range_constexpr(NA):
                    for k in range_constexpr(2):
                        row = (
                            wave_row0
                            + fx.Int32(h * LDS_BLOCK_M + ti * 16 + 8 * k)
                            + lane_id // 8
                        )
                        sid = fx.Int32(
                            buffer_ops.buffer_load(
                                ids_rsrc, row, vec_width=1, dtype=fx.Int32
                            )
                        )
                        flush_off.append(_orow(sid) * fx.Int32(OUT_ROW_BYTES))
            stg_base = fx.Int32(fx.ptrtoint(_stg_ptr)) + wave_id * fx.Int32(STG_WAVE)

            def _stg_addr(row, chunk, half):
                """Staged (row, 16-B chunk, 8-B half), chunks XOR-swizzled by row;
                inline asm keeps it at the use instead of in hoisted VGPRs."""
                base = stg_base + row * fx.Int32(STG_ROW) + fx.Int32(half * 8)
                x = fx.Int32(
                    _llvm.inline_asm(
                        fx.Int32.ir_type,
                        [fx.as_ir_value(fx.Int32(chunk)), fx.as_ir_value(row % STG_CH)],
                        "v_xor_b32 $0, $1, $2",
                        "=v,v,v",
                        has_side_effects=True,
                    )
                )
                return fx.Int32(
                    _llvm.inline_asm(
                        fx.Int32.ir_type,
                        [fx.as_ir_value(x), fx.as_ir_value(base)],
                        "v_lshl_add_u32 $0, $1, 4, $2",
                        "=v,v,v",
                        has_side_effects=True,
                    )
                )

            # prologue = steps -2, -1 of the steady-state issue order
            _set_scale_tile(fx.Int32(0))
            scale_gather.gather(0, _slot(0, 0))
            b_soff0 = _n_glob(fx.Int32(0)) * fx.Int32(B_TILE_BYTES)
            b0_g2s.load(b_st[0][0], b_soff0 + fx.Int32(0 * B_K_STEP))
            b1_g2s.load(b_st[0][1], b_soff0 + fx.Int32(0 * B_K_STEP))
            a0_g2s.load(a_cur0, fx.Int32(0 * A_K_STEP))
            a1_g2s.load(a_cur1, fx.Int32(0 * A_K_STEP))
            b0_g2s.load(b_st[1][0], b_soff0 + fx.Int32(1 * B_K_STEP))
            b1_g2s.load(b_st[1][1], b_soff0 + fx.Int32(1 * B_K_STEP))
            a0_g2s.load(a_next0, fx.Int32(1 * A_K_STEP))
            scale_gather.gather(1, _slot(0, 1))
            a1_g2s.load(a_next1, fx.Int32(1 * A_K_STEP))

            # gather 0, B(0) and a0(0) landed: a1(0), B(1), a0(1), gather, a1(1) may fly
            wait_barrier(NA + 2 * NB + NA + 1 + NA)
            a0_frag = a_s2r.load(a_cur0)
            b0_frag = b_s2r.load(b_st[0][0], preshuffled=True)
            b1_frag = b_s2r.load(b_st[0][1], preshuffled=True)
            sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0, 0))
            sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0, 0))
            sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)

            # Step kc issues B(kc + 2), a0(kc + 2), [MID] the scale gather (odd kc),
            # a1(kc + 2). Top wait: a1(kc) landed; MID wait: a0(kc + 1) landed. The
            # counts leave out the epilogue stores: they retire out of order.
            def _top_vmcnt(kc):
                return 2 * NB + 2 * NA + (1 - kc % 2)

            def _mid_vmcnt(kc):
                return NA + 2 * NB + NA + (1 - kc % 2)

            def _read_scale_thunks(nt, kc_next, holder):
                # the next flat step's scales; Python ints only (an ``if`` here would
                # become an scf.if)
                s = _slot(nt + fx.Int32(kc_next // K_ITERS), (kc_next % K_ITERS) // 2)

                def _r(dst, ld, half, _s=s):
                    holder[dst] = ld.read_half(_s, half)

                return [
                    lambda: _r(0, a_scale_ld, 0),
                    lambda: _r(1, a_scale_ld, 1),
                    lambda: _r(2, b_scale_ld, 0),
                    lambda: _r(3, b_scale_ld, 1),
                ]

            def _one_step(nt, kc, a0f, b0f, b1f_in, sc, accs, abufs):
                """``nt`` is the n-tile; ``kc`` is a constexpr K-step within it."""
                k2 = kc % 2
                ac0, ac1, an0, an1 = abufs
                bst = b_st[kc % B_STAGES]  # read now, refilled for step kc + 2
                bnx = b_st[(kc + 1) % B_STAGES]
                saR0, saR1, sbC0, sbC1 = sc
                c00f, c01f, c10f, c11f = accs
                zero_acc = kc == 0

                _a1 = [None] * NA
                _a0n = [None] * NA
                _b0n = [None] * NB
                _b1n = [None] * NB
                # A and B for flat step kc + 2 (B possibly in tile nt + 1)
                ka = (kc + 2) % K_ITERS
                kb = (kc + B_STAGES) % K_ITERS
                nt_b = nt + fx.Int32((kc + B_STAGES) // K_ITERS)
                a_off = fx.Int32(ka * A_K_STEP)
                b_off = _n_glob(nt_b) * fx.Int32(B_TILE_BYTES) + fx.Int32(kb * B_K_STEP)

                _scn = [None, None, None, None]
                _rd_scn = _read_scale_thunks(nt, kc + 1, _scn)
                # Gather scales three steps ahead, crossing n-tiles as needed.
                future_block = (kc + 3) // 2
                _g_blk = future_block % SCALE_BLOCKS
                _g_slot = _slot(nt + fx.Int32(future_block // SCALE_BLOCKS), _g_blk)
                _sc_gather = [lambda: scale_gather.gather(_g_blk, _g_slot)] * k2

                wait_barrier(_top_vmcnt(kc))
                il = (
                    _riffle(
                        _g2s_thunks(b0_g2s, bst[0], b_off, NB),
                        _s2r_thunks(a_s2r, ac1, _a1, NA, False),
                    )
                    + _rd_scn[:2]
                )
                c00f = mfma.call(
                    a0f,
                    b0f,
                    c00f,
                    saR0,
                    sbC0,
                    0,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                # never riffle two G2S loaders: each one's steps share m0, so
                # interleaving them corrupts the LDS destinations
                il = (
                    _g2s_thunks(b1_g2s, bst[1], b_off, NB)
                    + _g2s_thunks(a0_g2s, ac0, a_off, NA)
                    + _rd_scn[2:]
                )
                c01f = mfma.call(
                    a0f,
                    b1f_in,
                    c01f,
                    saR0,
                    sbC1,
                    1,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                a1f = _a1

                wait_barrier(_mid_vmcnt(kc))
                il = _s2r_thunks(a_s2r, an0, _a0n, NA, False) + _sc_gather
                c10f = mfma.call(
                    a1f,
                    b0f,
                    c10f,
                    saR1,
                    sbC0,
                    0,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                a0nf = _a0n
                il = _riffle(
                    _g2s_thunks(a1_g2s, ac1, a_off, NA),
                    _s2r_thunks(b_s2r, bnx[0], _b0n, NB, True)
                    + _s2r_thunks(b_s2r, bnx[1], _b1n, NB, True),
                )
                c11f = mfma.call(
                    a1f,
                    b1f_in,
                    c11f,
                    saR1,
                    sbC1,
                    1,
                    k2,
                    interleave=il,
                    zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                sc_next = (_scn[0], _scn[1], _scn[2], _scn[3])
                return (
                    a0nf,
                    _b0n,
                    _b1n,
                    sc_next,
                    (c00f, c01f, c10f, c11f),
                    (an0, an1, ac0, ac1),
                )

            def _stage(accs, h, ti):
                """Routing-weighted bf16 of rows (``h``, ``ti``) into the staging
                buffer; a wave's LDS ops run in order, after the last flush's reads."""
                w = row_w[h * NA + ti]
                for hf in range_constexpr(2):
                    cq = accs[h * 2 + hf]
                    for j in range_constexpr(NB):
                        v = Vec(cq[mfma.idx(ti, j)])
                        d0 = _bf16x2(fx.Float32(v[0]) * w, fx.Float32(v[1]) * w)
                        d1 = _bf16x2(fx.Float32(v[2]) * w, fx.Float32(v[3]) * w)
                        chunk = fx.Int32(hf * 8 + j * 2) + g4 // 2
                        _lds_store_vec(_v2i32(d0, d1), _stg_addr(r16, chunk, g4 % 2), 2)

            def _flush(nt, h, ti):
                """The staged rows -> full token-major 128-B lines, non-temporal."""
                col_wave = (_n_glob(nt) * fx.Int32(BN) + wave_j * (N_TILES_B * 16)) * 2
                ch_l = lane_id % 8
                _asm_void([], "s_waitcnt lgkmcnt(0)", "")
                for k in range_constexpr(N_FLUSH):
                    hf = k // 2
                    row = fx.Int32(8 * (k % 2)) + lane_id // 8
                    data = _lds_load_vec(_stg_addr(row, hf * 8 + ch_l, 0), 4)
                    buffer_ops.buffer_store(
                        data,
                        out_rsrc,
                        flush_off[(h * NA + ti) * 2 + k % 2]
                        + col_wave
                        + fx.Int32(hf * LDS_BLOCK_N * 2)
                        + ch_l * fx.Int32(16),
                        offset_is_bytes=True,
                        cache_modifier=_STORE_CPOL,
                    )

            def _epilogue(nt, accs):
                for h in range_constexpr(2):
                    for ti in range_constexpr(NA):
                        _stage(accs, h, ti)
                        _flush(nt, h, ti)

            abufs0 = (a_cur0, a_cur1, a_next0, a_next1)
            n_a = 2 * NA
            n_b = 2 * NB
            n_sc = 2 * (NA // 2) + 2 * (NB // 2)
            _R = fx.as_ir_value

            def _flat_sc(sc):
                return [_R(v) for part in sc for v in part]

            def _unflat_sc(flat):
                sizes = (NA // 2, NA // 2, NB // 2, NB // 2)
                out, o = [], 0
                for n in sizes:
                    out.append(list(flat[o : o + n]))
                    o += n
                return tuple(out)

            init_state = (
                _flat_frag(a0_frag)
                + _flat_frag(b0_frag)
                + _flat_frag(b1_frag)
                + _flat_sc(sc0)
            )
            # FlyDSL rewrites range(init=...) to a loop carrying state.
            for nt_idx, state in range(  # type: ignore[call-overload]
                0, NT, 1, init=init_state
            ):
                nt = fx.Int32(nt_idx)
                off = 0
                a0f = _unflat_frag(state[off : off + n_a], NA)
                off += n_a
                b0f = _unflat_frag(state[off : off + n_b], NB)
                off += n_b
                b1f = _unflat_frag(state[off : off + n_b], NB)
                off += n_b
                sc = _unflat_sc(state[off : off + n_sc])
                accs = (
                    [None] * N_ACCUMS,
                    [None] * N_ACCUMS,
                    [None] * N_ACCUMS,
                    [None] * N_ACCUMS,
                )
                abufs = abufs0
                # Switch the scale base before the first next-tile gather.
                _set_scale_tile(nt)
                for kc in range_constexpr(K_ITERS):
                    if const_expr(kc == K_ITERS - 4):
                        _set_scale_tile(nt + fx.Int32(1))
                    a0f, b0f, b1f, sc, accs, abufs = _one_step(
                        nt, kc, a0f, b0f, b1f, sc, accs, abufs
                    )
                _epilogue(nt, accs)
                state = (
                    yield _flat_frag(a0f)
                    + _flat_frag(b0f)
                    + _flat_frag(b1f)
                    + _flat_sc(sc)
                )
            # the redundant loads after the last tile must land before the LDS is reused
            wait_barrier(0)

    @flyc.jit
    def launch_gemm2_bm256(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm2_bm256(
            A,
            W2,
            OUT,
            A_scale,
            W2_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            n_tokens,
            num_m_blocks,
            grid_size,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(grid_size, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2_bm256
