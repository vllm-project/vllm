# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill down fp8 GEMM (3072..65536 tokens), token-major partials:

    out[tok*topk + slot, :] = bf16( (h[row, :] @ W2[e]^T) * sorted_weights[row] )

reduced by ``moe_flydsl_common.reduce_bf16`` (``out_mode="bf16"``) or, as unweighted
MXFP8 partials, by ``reduce_fp8`` (``"fp8"``). The gemm1 mainloop (4 waves 2x2, LDS
ping-pong, 128-K steps, ``v_mfma_scale_f32_16x16x128_f8f6f4``, AGPR accumulators)
runs as one flat sequence over the CTA's n-tiles: a CTA owns an m-tile of 128 sorted
rows and sweeps ``NT`` n-tiles of 256 W2 rows (``n_split`` CTAs share the 6144
columns, rotated start inside long same-expert runs). K = 768 is 6 steps per n-tile;
the first loads of n-tile t+1 are issued during n-tile t. W2 is prefetched three
steps ahead (3 LDS stages), A two; LDS: 32 KB A + 96 KB B + 16 KB scales + 16 KB
epilogue staging. Epilogue: accumulators scaled by the routing weight, packed to
bf16, staged 16 rows at a time in a wave-private LDS buffer and flushed token-major
as full 128 B lines (non-temporal); padded rows (tok == n_tokens) fall outside the
output resource. Scale blocks (256 B = 32 rows x 8 K-groups): 3 per operand per
n-tile, block b = 3t + g in LDS slot b % 4, gathered at flat step 2b - 3.

Layouts (bytes):
  A         [num_m_blocks*BM, I]         sorted rows, fp8 (gemm1 OUT_Q)
  A_scale   [pad32(num_m_blocks*BM), I/32]  sorted rows, e8m0-shuffled (gemm1 OUT_sc)
  W2        [E, H, I]                    aiter shuffle_weight (16 x 64 K blocks)
  W2_sc     [E*H/32, I/256, 4, 16] dwords aiter shuffle_scale
  sorted_w  [num_m_blocks*BM]            f32 routing weight per sorted row
  OUT       [n_tokens*topk, H]           bf16 ("bf16") or fp8 ("fp8")
  OUT_sc    [n_tokens*topk, H/32]        e8m0 ("fp8"; any buffer otherwise)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.epilogue import (
    _NUM_XCDS,
    _STORE_CPOL,
    _bf16x2,
    _cvt_pk_fp8,
    _lds_load_vec,
    _lds_store_vec,
    _maxf_nn,
    _undef_i32,
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
    _as_f32,
    _asm_void,
    _Buf,
    _divmod_nonneg,
    _flat_frag,
    _fmax,
    _g2s_thunks,
    _permlane16_swap,
    _riffle,
    _s2r_thunks,
    _swizzled_col,
    _unflat_frag,
    wait_barrier,
)

from .gemm1_prefill import BLOCK_K, Mfma16x16x128Fp8, _e8m0_roundup_fp8

OUT_MODES = ("bf16", "fp8")

_SCALE_SLOTS2 = 4


def compile_moe_gemm2(
    *,
    H: int,
    I: int,  # noqa: E741
    E: int,
    topk: int,
    n_split: int = 2,
    sort_block_m: int = 128,
    out_mode: str = "bf16",
):
    """Grouped fp8 gemm2 for one (H, I, E, topk). ``sort_block_m`` (128 / 256) is the
    sort block of the inputs; the kernel tiles 128 rows and skips the all-padding
    halves of a 256-sort. ``out_mode``: "bf16" = weighted bf16 partials for
    ``moe_flydsl_common.reduce_bf16``; "fp8" = unweighted fp8 e4m3 partials +
    e8m0 per 32 columns (ceil_pow2(amax/448)) for ``reduce_fp8``, half the partial
    traffic, one more quantization. Both deterministic.
    """
    assert out_mode in OUT_MODES, out_mode
    MODE = out_mode
    BM = 128
    BN = 256
    K = I
    K_BYTES = K
    BLOCK_K_BYTES = BLOCK_K
    EID_SHIFT = (sort_block_m // BM).bit_length() - 1  # 0 or 1
    assert sort_block_m in (BM, 2 * BM)
    N_TILES_ALL = H // BN
    assert H % BN == 0 and N_TILES_ALL % n_split == 0
    NT = N_TILES_ALL // n_split  # n-tiles per CTA
    K_ITERS = K // BLOCK_K
    assert K % 256 == 0 and K_ITERS == 6, "the flat schedule is written for K = 768"
    LDS_BLOCK_M = BM // 2
    LDS_BLOCK_N = BN // 2
    N_TILES_A = LDS_BLOCK_M // 2 // 16  # 2
    N_TILES_B = LDS_BLOCK_N // 2 // 16  # 4
    N_ACCUMS = N_TILES_A * N_TILES_B
    NB = N_TILES_B
    NA = N_TILES_A
    OUT_ELEM = 1 if MODE == "fp8" else 2
    OUT_ROW_BYTES = H * OUT_ELEM
    OUT_SC_COLS = H // 32  # e8m0 per output row (fp8 mode)
    W2_BYTES = E * H * K
    assert W2_BYTES <= 0xFFFFFFFF
    B_TILE_BYTES = BN * K_BYTES  # one n-tile of W2 (256 rows)

    B_STAGES = 3
    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES  # 8 KB
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES  # 16 KB
    A_BUFS = 4 * a_lds_size  # 32 KB: 2 stages x 2 halves
    LDS_TILES_BYTES = A_BUFS + 2 * B_STAGES * b_lds_size  # 128 KB
    SCALE_LDS_BYTES2 = _SCALE_SLOTS2 * _SCALE_SLOT_BYTES  # 16 KB
    # epilogue staging, wave-private: 16 rows x (2 halves x 64 cols) bf16, 256 B rows
    STG_ROW = 2 * N_TILES_B * 16 * 2  # 256 B: the wave's two 128-B line segments
    STG_CH = STG_ROW // 16  # 16 chunks of 16 B per staged row
    STG_ROWS = 16  # one (row half, a-tile) at a time
    STG_WAVE = STG_ROWS * STG_ROW  # 4 KB
    STG_LDS_BYTES = _N_WAVES * STG_WAVE  # 16 KB
    N_FLUSH = (STG_ROWS * 2) // 8  # dwordx4 stores per round per lane: 32 lines, 8 each
    assert LDS_TILES_BYTES + SCALE_LDS_BYTES2 + STG_LDS_BYTES <= 160 * 1024

    A_WAVE_GROUPS = N_TILES_A // 2  # 1
    A_HALF_GROUPS = LDS_BLOCK_M // 32  # 2
    B_WAVE_GROUPS = 2  # 32-row groups per wave (64 W2 rows)
    B_HALF_GROUPS = LDS_BLOCK_N // 32  # 4: half 1 = 128 rows further
    SC_COLS = K // 32
    W2_SC_BYTES = E * H * SC_COLS

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, LDS_TILES_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, SCALE_LDS_BYTES2, 16]
        stage_lds: fx.Array[fx.Int8, STG_LDS_BYTES, 16]

    @flyc.kernel
    def kernel_gemm2(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
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
        # B: 3 stages x 2 halves
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

        # ---- work item: (m-tile, n-chunk); consecutive m-tiles share an XCD ----
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
            # a 256-sort pads each expert to 256 rows: a 128-row tile whose first
            # row is the sentinel (tok == n_tokens) holds nothing to compute
            first_sid = fx.Int32(
                buffer_ops.buffer_load(
                    ids_rsrc, m_base, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            block_valid = block_valid & ((first_sid & fx.Int32(0x00FFFFFF)) < n_tokens)
        chunk_n0 = chunk * NT  # first n-tile (global index) of this CTA
        # ---- rotated n-tile sweep inside runs of >= 4 same-expert m-tiles:
        # neighbouring CTAs then find their next W2 tile in L2 ----
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
            """CTA-local logical n-tile (sweep order) -> physical n-tile of the chunk"""
            x = nt + nt_rot
            return (x >= fx.Int32(NT)).select(x - fx.Int32(NT), x)

        def _n_glob(nt):
            """global n-tile index of logical n-tile ``nt`` (clamped to the sweep:
            the redundant loads after the last tile re-fetch the last one)"""
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
                W2, max_size=False, num_records_bytes=W2_BYTES
            )
            # padded sorted rows (tok == n_tokens) fall outside the resource: dropped
            out_rows = n_tokens * fx.Int32(topk)
            out_rsrc = buffer_ops.create_buffer_resource(
                OUT, max_size=False, num_records_bytes=out_rows * OUT_ROW_BYTES
            )
            osc_rsrc = buffer_ops.create_buffer_resource(
                OUT_scale, max_size=False, num_records_bytes=out_rows * OUT_SC_COLS
            )

            # ---- A rows (sorted, contiguous): row * K_BYTES + swizzled col ----
            def _a_offsets(half):
                offs = []
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id % 8) * 16
                    grow = m_base + half * LDS_BLOCK_M + row
                    offs.append(grow * fx.Int32(K_BYTES) + _swizzled_col(row, col))
                return offs

            gl_off_a0 = _a_offsets(0)
            gl_off_a1 = _a_offsets(1)

            # ---- W2: LDS row r of half hf = W2 row expert*H + n*256 + hf*128 + r,
            # 16-row blocks of K_BYTES*16 bytes, a8 preshuffle inside; the n-tile
            # goes in soffset (B_TILE_BYTES per tile) ----
            e_row0 = expert * fx.Int32(H)

            def _b_offsets(hf):
                offs = []
                for rnd in range_constexpr(N_TILES_B):
                    row = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id // 8) * 16
                    wrow = e_row0 + fx.Int32(hf * LDS_BLOCK_N + row)
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

            gl_off_b0 = _b_offsets(0)
            gl_off_b1 = _b_offsets(1)
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
                return (fx.Int32(nt) * fx.Int32(3) + fx.Int32(g)) % fx.Int32(
                    _SCALE_SLOTS2
                )

            a0_g2s = G2SLoaderAsm(a_rsrc, gl_off_a0, N_TILES_A, wave_id)
            a1_g2s = G2SLoaderAsm(a_rsrc, gl_off_a1, N_TILES_A, wave_id)
            b0_g2s = G2SLoaderAsm(b_rsrc, gl_off_b0, N_TILES_B, wave_id)
            b1_g2s = G2SLoaderAsm(b_rsrc, gl_off_b1, N_TILES_B, wave_id)
            for ld in (a0_g2s, a1_g2s, b0_g2s, b1_g2s):
                ld.set_wave_base(_base_ptr)
            a_s2r = S2RLoader128B(wave_i, N_TILES_A)
            b_s2r = S2RLoader128B(wave_j, N_TILES_B)

            # ---- output rows / weights of the 4 rows this lane writes ----
            def _orow(sid):
                tok = sid & fx.Int32(0x00FFFFFF)  # padded: tok == n_tokens -> OOB
                slot = (sid >> 24) & fx.Int32(0xFF)
                return tok * fx.Int32(topk) + slot

            # routing weight of the lane's own rows (row half h, tile ti, r16) and the
            # output row offsets of the rows the lane flushes (per (h, ti), store k:
            # staged row 4k + lane//16)
            wave_row0 = m_base + wave_i * (N_TILES_A * 16)
            row_w = []
            own_off = []  # output row (tok*topk + slot) of the lane's own rows (fp8)
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
                    sid = fx.Int32(
                        buffer_ops.buffer_load(
                            ids_rsrc, row, vec_width=1, dtype=fx.Int32
                        )
                    )
                    own_off.append(_orow(sid))
            flush_off = []
            for h in range_constexpr(2):
                for ti in range_constexpr(NA):
                    for k in range_constexpr(N_FLUSH):
                        row = (
                            wave_row0
                            + fx.Int32(h * LDS_BLOCK_M + ti * 16 + 4 * k)
                            + lane_id // 16
                        )
                        sid = fx.Int32(
                            buffer_ops.buffer_load(
                                ids_rsrc, row, vec_width=1, dtype=fx.Int32
                            )
                        )
                        flush_off.append(_orow(sid) * fx.Int32(OUT_ROW_BYTES))
            stg_base = fx.Int32(fx.ptrtoint(_stg_ptr)) + wave_id * fx.Int32(STG_WAVE)

            def _stg_addr(row, chunk, half):
                """staged (row 0..15, 16-B chunk, 8-B half); chunks XOR-swizzled by the
                row so the tile-wise writes and the line-wise reads spread over the
                banks"""
                return (
                    stg_base
                    + row * fx.Int32(STG_ROW)
                    + ((chunk ^ (row % STG_CH)) * 16 + half * 8)
                )

            # ---- prologue = "steps -3, -2, -1" of the steady-state issue order
            # (B three steps ahead, A two): B(0); B(1) a0(0) a1(0); B(2) a0(1)
            # gather a1(1); block 0's gather up front ----
            _set_scale_tile(fx.Int32(0))
            scale_gather.gather(0, _slot(0, 0))
            b_soff0 = _n_glob(fx.Int32(0)) * fx.Int32(B_TILE_BYTES)
            b0_g2s.load(b_st[0][0], b_soff0 + fx.Int32(0 * B_K_STEP))
            b1_g2s.load(b_st[0][1], b_soff0 + fx.Int32(0 * B_K_STEP))
            b0_g2s.load(b_st[1][0], b_soff0 + fx.Int32(1 * B_K_STEP))
            b1_g2s.load(b_st[1][1], b_soff0 + fx.Int32(1 * B_K_STEP))
            a0_g2s.load(a_cur0, fx.Int32(0 * A_K_STEP))
            a1_g2s.load(a_cur1, fx.Int32(0 * A_K_STEP))
            b0_g2s.load(b_st[2][0], b_soff0 + fx.Int32(2 * B_K_STEP))
            b1_g2s.load(b_st[2][1], b_soff0 + fx.Int32(2 * B_K_STEP))
            a0_g2s.load(a_next0, fx.Int32(1 * A_K_STEP))
            scale_gather.gather(1, _slot(0, 1))
            a1_g2s.load(a_next1, fx.Int32(1 * A_K_STEP))

            # gather 0, B(0) and a0(0) landed: a1(0), B(2), a0(1), gather, a1(1) may fly
            wait_barrier(NA + 2 * NB + NA + 1 + NA)
            a0_frag = a_s2r.load(a_cur0)
            b0_frag = b_s2r.load(b_st[0][0], preshuffled=True)
            b1_frag = b_s2r.load(b_st[0][1], preshuffled=True)
            sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0, 0))
            sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0, 0))
            sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)

            # Per step kc of n-tile nt, in issue order: B(kc+3) b0 b1, a0(kc+2),
            # [MID] scale gather (odd kc), a1(kc+2); B of flat step f in stage f % 3,
            # A in stage f % 2. Loop-top wait: a1(kc) landed = all of step kc-1's
            # loads may fly; MID wait: a0(kc+1) landed. vmcnt counts loads only:
            # stores retire out of order with respect to loads, so the epilogue
            # stores must not be counted (they were: races in the first n-tile).
            def _top_vmcnt(kc):
                return 2 * NB + 2 * NA + (1 - kc % 2)

            def _mid_vmcnt(kc):
                return NA + 2 * NB + NA + (1 - kc % 2)

            def _read_scale_thunks(nt, kc_next, holder):
                # scales of flat step +1: same tile block kc_next//2, or block 0 of
                # the next tile (Python ints only: an ``if`` here would become an
                # scf.if and lose the binding)
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
                """``nt`` loop value (logical n-tile), ``kc`` Python int 0..5."""
                k2 = kc % 2
                ac0, ac1, an0, an1 = abufs
                bst = b_st[kc % 3]  # consumed this step, refilled for step kc + 3
                bnx = b_st[(kc + 1) % 3]  # step kc + 1's halves, read this step
                saR0, saR1, sbC0, sbC1 = sc
                c00f, c01f, c10f, c11f = accs
                zero_acc = kc == 0

                _a1 = [None] * NA
                _a0n = [None] * NA
                _b0n = [None] * NB
                _b1n = [None] * NB
                # A for flat step kc + 2, B for kc + 3 (the latter in tile nt + 1
                # from kc = 3 on)
                ka = (kc + 2) % K_ITERS
                kb = (kc + 3) % K_ITERS
                nt_b = nt + fx.Int32((kc + 3) // K_ITERS)
                a_off = fx.Int32(ka * A_K_STEP)
                b_off = _n_glob(nt_b) * fx.Int32(B_TILE_BYTES) + fx.Int32(kb * B_K_STEP)

                _scn = [None, None, None, None]
                _rd_scn = _read_scale_thunks(nt, kc + 1, _scn)
                # gathers (odd steps): step 1 -> (nt, 2); 3 -> (nt+1, 0); 5 -> (nt+1, 1)
                _g_blk = {1: 2, 3: 0, 5: 1}.get(kc, 1)
                _g_slot = _slot(nt + fx.Int32({1: 0}.get(kc, 1)), _g_blk)
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
                # two G2S loaders are never riffled: each one's steps share m0
                # (s_mov then s_add per step), interleaving them corrupts the LDS
                # destinations
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
                """routing-weighted bf16 of the 16 rows (half ``h``, a-tile ``ti``) x
                the wave's 128 cols into the staging buffer, 8 B per lane per tile.
                LDS ops of one wave execute in order: the writes follow the previous
                flush's reads of the same buffer."""
                w = row_w[h * NA + ti]
                for hf in range_constexpr(2):
                    cq = accs[h * 2 + hf]
                    for j in range_constexpr(NB):
                        v = Vec(cq[mfma.idx(ti, j)])
                        d0 = _bf16x2(fx.Float32(v[0]) * w, fx.Float32(v[1]) * w)
                        d1 = _bf16x2(fx.Float32(v[2]) * w, fx.Float32(v[3]) * w)
                        # cols hf*128 + j*16 + 4*g4 of the wave's 128 -> chunk
                        chunk = fx.Int32(hf * 8 + j * 2) + g4 // 2
                        _lds_store_vec(_v2i32(d0, d1), _stg_addr(r16, chunk, g4 % 2), 2)

            def _flush(nt, h, ti):
                """the staged 16 rows -> token-major full 128-B lines: lane -> line
                (row 4k + lane//16, segment hf = (lane//8)%2), 16 B at chunk lane%8,
                8 lines per store (non-temporal)."""
                col_wave = (_n_glob(nt) * fx.Int32(BN) + wave_j * (N_TILES_B * 16)) * 2
                hf_l = (lane_id // 8) % 2
                ch_l = lane_id % 8
                _asm_void([], "s_waitcnt lgkmcnt(0)", "")
                for k in range_constexpr(N_FLUSH):
                    row = fx.Int32(4 * k) + lane_id // 16
                    data = _lds_load_vec(_stg_addr(row, hf_l * 8 + ch_l, 0), 4)
                    buffer_ops.buffer_store(
                        data,
                        out_rsrc,
                        flush_off[(h * NA + ti) * N_FLUSH + k]
                        + col_wave
                        + hf_l * fx.Int32(LDS_BLOCK_N * 2)
                        + ch_l * fx.Int32(16),
                        offset_is_bytes=True,
                        cache_modifier=_STORE_CPOL,
                    )

            def _epilogue_fp8(nt, accs):
                """unweighted y -> fp8 per 32-col group (e8m0 = ceil_pow2(amax/448)),
                8 B per lane after a permlane16 swap (gemm1's epilogue), the two e8m0
                bytes of a (row, 64-col half) as one 16-bit store from lane group 0"""
                colbase = _n_glob(nt) * fx.Int32(BN) + wave_j * (N_TILES_B * 16)
                for q in range_constexpr(4):
                    h, hf = q // 2, q % 2
                    for ti in range_constexpr(NA):
                        row_off = own_off[h * NA + ti]  # tok*topk + slot
                        e8_of_p = []
                        for p in range_constexpr(NB // 2):
                            gv = Vec(accs[q][mfma.idx(ti, 2 * p)])
                            gw = Vec(accs[q][mfma.idx(ti, 2 * p + 1)])
                            hv = [fx.Float32(gv[v]) for v in range(4)] + [
                                fx.Float32(gw[v]) for v in range(4)
                            ]
                            amax = _maxf_nn(fx.math.absf(hv[0]), fx.math.absf(hv[1]))
                            for v in range_constexpr(2, 8):
                                amax = _maxf_nn(amax, fx.math.absf(hv[v]))
                            amax = _fmax(amax, amax.shuffle_xor(16, 64))
                            amax = _fmax(amax, amax.shuffle_xor(32, 64))
                            e8m0 = _e8m0_roundup_fp8(amax)
                            scale_f = _as_f32(e8m0 << 23)
                            da = _cvt_pk_fp8(_undef_i32(), hv[0], hv[1], scale_f, False)
                            da = _cvt_pk_fp8(da, hv[2], hv[3], scale_f, True)
                            db = _cvt_pk_fp8(_undef_i32(), hv[4], hv[5], scale_f, False)
                            db = _cvt_pk_fp8(db, hv[6], hv[7], scale_f, True)
                            da, db = _permlane16_swap(da, db)
                            # lane group g holds tile (2p + g%2), cols (g//2)*8 .. +8
                            col = (
                                colbase
                                + fx.Int32(hf * LDS_BLOCK_N + 2 * p * 16)
                                + (g4 % 2) * 16
                                + (g4 // 2) * 8
                            )
                            buffer_ops.buffer_store(
                                _v2i32(da, db),
                                out_rsrc,
                                row_off * fx.Int32(OUT_ROW_BYTES) + col,
                                offset_is_bytes=True,
                            )
                            e8_of_p.append(e8m0)
                        # the two 32-col groups of this (row, half): one 16-bit store
                        # from lane group 0 (the group index is even: 2-byte aligned)
                        colgrp = (colbase + fx.Int32(hf * LDS_BLOCK_N)) // 32
                        pair16 = fx.Int32(e8_of_p[0] | (e8_of_p[1] << 8)).to(fx.Int16)
                        buffer_ops.buffer_store(
                            pair16,
                            osc_rsrc,
                            row_off * fx.Int32(OUT_SC_COLS) + colgrp,
                            mask=fx.as_ir_value(g4 == fx.Int32(0)),
                            offset_is_bytes=True,
                        )

            def _epilogue(nt, accs):
                if const_expr(MODE == "fp8"):
                    _epilogue_fp8(nt, accs)
                else:
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
                nt = fx.Int32(
                    nt_idx
                )  # the loop value is an index; i32 everywhere below
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
                # the gather of step 1 (block 2 of this tile) needs this tile's scale
                # base; from step 2 on the gathers belong to n-tile nt + 1. Both are
                # (re)computed from the loop value: the loop body is emitted once.
                _set_scale_tile(nt)
                for kc in range_constexpr(K_ITERS):
                    if const_expr(kc == 2):
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
            # the redundant loads of the tile after the last one must land before
            # the LDS is handed on
            wait_barrier(0)

    @flyc.jit
    def launch_gemm2(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        OUT_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm2(
            A,
            W2,
            OUT,
            A_scale,
            W2_scale,
            OUT_scale,
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

    return launch_gemm2


def gemm2_grid(num_m_blocks: int, n_split: int) -> int:
    """blocks to launch: every (m-tile, chunk) of the allocation, rounded to the 8
    XCDs (the kernel drops the fully padded tail tiles at runtime)."""
    return (num_m_blocks * n_split + _NUM_XCDS - 1) // _NUM_XCDS * _NUM_XCDS
