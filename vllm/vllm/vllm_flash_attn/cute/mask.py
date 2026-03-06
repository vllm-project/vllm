# Copyright (c) 2025, Tri Dao.

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from quack import layout_utils

import vllm.vllm_flash_attn.cute.utils as utils
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK


@cute.jit
def mask_r2p(
    X: cute.Tensor, col_limit: Int32, arch: int = 90, rank1: bool = False
) -> None:
    # Bit manipulation, compiles down to the R2P instruction
    # For sm100: we know that tScS_t2r[i][1] == i, for the particular tmem copy atom we're using.
    # For sm90: instead of comparing limit to 0, 1, 8, 9, 16, 17, ...,
    # we compare a transformed version of limit to 0, 1, 2, 3, 4, 5, ...
    if const_expr(arch == 90):
        col_limit_transformed = col_limit // 8 * 2 + min(col_limit % 8, 2)
    else:
        col_limit_transformed = col_limit
    ncol = const_expr(
        cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape)
    )
    # Ideally we'd move by 32 instead of 24, but mask >> i isn't correct for i == 31
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        # Don't need to clamp to 32 since the shr.u32 instruction does that already
        col_limit_right_s = max(col_limit_transformed - s * 24, 0)
        # 0 -> 0b00...00, 1 -> 0b00...01, ..., 31 -> 0b01...11, 32 -> 0b11...11
        mask = (1 << col_limit_right_s) - 1
        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            in_bound = cutlass.Boolean(mask & (1 << i))
            c = s * 24 + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
                # This is the equivalent of:
                # X[s * 24 + i] = X[s * 24 + i] if col_limit_right_s <= i else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def mask_r2p_transposed(X: cute.Tensor, row_limit_top: Int32, num_rep: int) -> None:
    # Bit manipulation, compiles down to the R2P instruction
    # For sm100: we know that tScS_t2r[i][0] has the form 0, 1, ..., 31, 64, ..., 127
    # or 0, 1, ..., 15, 32, ..., 47, 64, ...
    # We compare a transformed version of limit to 0, 1, 2, 3, 4, 5, ...
    # Here we hardcode for the case of 2 warp groups.
    num_wg = 2
    row_limit_top_transformed = row_limit_top // (num_rep * num_wg) * num_rep + min(
        row_limit_top % (num_rep * num_wg), num_rep
    )
    ncol = cute.size(X.shape)
    # Ideally we'd move by 32 instead of 24, but mask >> i isn't correct for i == 31
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        row_limit_top_s = max(row_limit_top_transformed - s * 24, 0)
        # 0 -> 0b00...00, 1 -> 0b00...01, ..., 31 -> 0b01...11, 32 -> 0b11...11
        mask = (1 << row_limit_top_s) - 1
        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            out_bound = cutlass.Boolean(mask & (1 << i))
            c = s * 24 + i
            X[c] = -Float32.inf if out_bound else X[c]
            # tidx = cute.arch.thread_idx()[0] % 256
            # if tidx == 128:
            #     cute.printf("tidx = {}, s = {}, i = {}, row_limit_top = {}, row_limit_top_s = {}, mask = {}, out_bound = {}", tidx, s, i, row_limit_top, row_limit_top_s, mask, out_bound)


@cute.jit
def mask_r2p_dual_bound(
    X: cute.Tensor,
    col_limit_left: Int32,  # Inclusive lower bound
    col_limit_right: Int32,  # Exclusive upper bound
) -> None:
    """
    Dual-bound masking using two bitmasks for SM100, following mask_r2p.
    Masks elements where: NOT (col_limit_left <= col < col_limit_right)

    Uses bit manipulation to create a range mask:
        mask_right = (1 << right) - 1  -> bits (right-1)..0 are 1
        mask_left  = (1 << left) - 1   -> bits (left-1)..0 are 1
        mask_range = mask_range = mask_right & ~ mask_left -> bits (right-1)..left are 1
    """
    ncol = const_expr(cute.size(X.shape))

    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        right_s = max(col_limit_right - s * 24, 0)
        left_s = max(col_limit_left - s * 24, 0)

        # otherwise cute dsl complains about python int too large to convert into c long
        right_s = min(right_s, 24)
        left_s = min(left_s, 24)

        # bits (right-1)..left are 1
        mask_right = (1 << right_s) - 1
        mask_left = (1 << left_s) - 1
        mask_range = mask_right & ~mask_left

        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            in_bound = cutlass.Boolean(mask_range & (1 << i))
            c = s * 24 + i
            X[c] = X[c] if in_bound else -Float32.inf


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    window_size_left: Int32 | None = None
    window_size_right: Int32 | None = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = (
        1  # only pass in if we're doing PackGQA
    )
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Callable | None] = None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
    ) -> None:
        assert not (mask_causal and mask_local), (
            "mask_causal and mask_local cannot be both True"
        )
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(
            acc_shape if not self.swap_AB else acc_shape[::-1]
        )
        tScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.partition_C(cS), transpose=self.swap_AB
        )
        # We use t0ScS as these indices are known at compile time. We then must subtract the
        # column limit by the thread column offset.
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
        )
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][COL]
        # To handle edge cases of completely masked out rows where n_block_max = 0,
        # we treat negative n_blocks as 0th n_block
        # TODO: find more transparent solution
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                # The compiler now choses not to use R2P
                r2p = const_expr(False and not self.swap_AB)
                if const_expr(not r2p):
                    # traverse column index.
                    for c in cutlass.range(
                        cute.size(tScS_mn.shape[1]), unroll_full=True
                    ):
                        oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
                        for r in cutlass.range(
                            cute.size(tScS_mn.shape[0]), unroll_full=True
                        ):
                            acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
                else:
                    mask_r2p(acc_S_mn, seqlenk_col_limit, arch=90)

        elif const_expr(
            not mask_causal and not mask_local and mask_mod is not None
        ):  # FlexAttention mask mod
            nrow = const_expr(cute.size(tScS_mn.shape[0]))
            ncol = const_expr(cute.size(tScS_mn.shape[1]))
            has_fastdiv = const_expr(
                fastdiv_mods is not None
                and fastdiv_mods[0] is not None
                and fastdiv_mods[1] is not None
            )
            wrap_aux_indices = const_expr(
                has_fastdiv and mask_seqlen and const_expr(aux_tensors is not None)
            )

            for r in cutlass.range_constexpr(nrow):
                # Respect swap_AB: ROW/COL determine which coordinate component corresponds to Q/KV.
                local_row = tScS_mn[r, 0][ROW]
                global_row_idx = local_row + m_block * self.tile_m
                row_for_mod = global_row_idx
                head_idx_for_mod = head_idx
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    head_offset = global_row_idx % self.qhead_per_kvhead_packgqa
                    head_idx_for_mod = (
                        head_idx * self.qhead_per_kvhead_packgqa + head_offset
                    )
                    row_for_mod = global_row_idx // self.qhead_per_kvhead_packgqa
                row_for_seqlen = row_for_mod
                if const_expr(wrap_aux_indices):
                    _, row_for_mod = divmod(row_for_mod, fastdiv_mods[0])

                for col in cutlass.range_constexpr(ncol):
                    col_idx_local = t0ScS_mn[0, col][COL]
                    # Convert to absolute column index
                    global_col_idx = (
                        thr_col_offset + col_idx_local + n_block * self.tile_n
                    )
                    col_for_mod = global_col_idx
                    if const_expr(wrap_aux_indices):
                        _, col_for_mod = divmod(global_col_idx, fastdiv_mods[1])

                    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                    head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
                    q_idx_ssa = utils.scalar_to_ssa(row_for_mod, cutlass.Int32)
                    kv_idx_ssa = utils.scalar_to_ssa(col_for_mod, cutlass.Int32)
                    mask_value = mask_mod(
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        self.seqlen_info,
                        aux_tensors,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    if const_expr(mask_seqlen):
                        out_of_bounds = (row_for_seqlen >= self.seqlen_q) or (
                            global_col_idx >= self.seqlen_k
                        )
                        if out_of_bounds:
                            acc_S_mn[r, col] = -cutlass.Float32.inf
                        else:
                            acc_S_mn[r, col] = (
                                acc_S_mn[r, col] if cond else -cutlass.Float32.inf
                            )
                    else:
                        acc_S_mn[r, col] = (
                            acc_S_mn[r, col] if cond else -cutlass.Float32.inf
                        )

        else:  # Causal or local
            if const_expr(not self.swap_AB):
                # If PackGQA, we split the work of compute divmod among threads in the same row
                threads_per_row = thr_mma.tv_layout_C.shape[0][0]
                mma_m_idx = None
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert not self.swap_AB, "swap_AB with PackGQA not supported yet"
                    assert cute.arch.WARP_SIZE % threads_per_row == 0, (
                        "threads_per_row must divide WARP_SIZE"
                    )
                    assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                    tidx = thr_mma.thr_idx
                    mma_m_idx = (
                        m_block * self.tile_m + tScS_mn[tidx % threads_per_row, 0][0]
                    ) // self.qhead_per_kvhead_packgqa
                causal_row_offset = (
                    1
                    + self.seqlen_k
                    - n_block * self.tile_n
                    - self.seqlen_q
                    - thr_col_offset
                )
                if const_expr(mask_causal):
                    r2p = const_expr(
                        not self.swap_AB
                    )  # R2P trick, see apply_mask_sm100
                    for r in cutlass.range(
                        cute.size(tScS_mn.shape[0]), unroll_full=True
                    ):
                        # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        col_limit_right = row_idx + causal_row_offset
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(
                                col_limit_right, seqlenk_col_limit
                            )
                        if const_expr(not r2p):
                            # traverse column index.
                            for c in cutlass.range(
                                cute.size(tScS_mn.shape[1]), unroll_full=True
                            ):
                                acc_S_mn[r, c] = (
                                    -Float32.inf
                                    if t0ScS_mn[0, c][1] >= col_limit_right
                                    else acc_S_mn[r, c]
                                )
                        else:
                            mask_r2p(
                                acc_S_mn[r, None], col_limit_right, arch=90, rank1=True
                            )
                else:  # Local
                    local_row_offset_right = (
                        causal_row_offset + self.window_size_right
                        if const_expr(self.window_size_right is not None)
                        else None
                    )
                    local_row_offset_left = (
                        causal_row_offset - 1 - self.window_size_left
                        if const_expr(self.window_size_left is not None)
                        else None
                    )
                    for r in cutlass.range(
                        cute.size(tScS_mn.shape[0]), unroll_full=True
                    ):
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        if const_expr(self.window_size_right is not None):
                            col_limit_right = row_idx + local_row_offset_right
                        else:
                            col_limit_right = self.tile_n
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(
                                col_limit_right, seqlenk_col_limit
                            )
                        col_limit_left = (
                            row_idx + local_row_offset_left
                            if const_expr(self.window_size_left is not None)
                            else 0
                        )
                        # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block = {}, r = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", n_block, r, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                        # traverse column index.
                        for c in cutlass.range(
                            cute.size(tScS_mn.shape[1]), unroll_full=True
                        ):
                            col_idx = t0ScS_mn[0, c][1]
                            # only consider the column index, so the row index sets to 0.
                            if col_idx >= col_limit_right or col_idx < col_limit_left:
                                acc_S_mn[r, c] = -Float32.inf
            else:  # swap_AB
                assert self.qhead_per_kvhead_packgqa == 1
                thr_row_offset = tScS_mn[0][ROW]
                causal_row_offset = (
                    seqlenk_col_limit
                    - self.seqlen_q
                    + m_block * self.tile_m
                    + thr_row_offset
                )
                if const_expr(mask_causal):
                    for c in cutlass.range(
                        cute.size(tScS_mn.shape[1]), unroll_full=True
                    ):
                        col0 = t0ScS_mn[0, c][COL]
                        # If col0 is beyond the column limit, we want to mask out the entire
                        # column, by setting row limit to be self.tile_m.
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit and mask_seqlen
                            else col0 - causal_row_offset
                        )
                        for r in cutlass.range(
                            cute.size(tScS_mn.shape[0]), unroll_full=True
                        ):
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if t0ScS_mn[r, 0][ROW] < row_limit_top
                                else acc_S_mn[r, c]
                            )
                else:
                    for c in cutlass.range(
                        cute.size(tScS_mn.shape[1]), unroll_full=True
                    ):
                        col0 = t0ScS_mn[0, c][COL]
                        # If col0 is beyond the column limit, we want to mask out the entire
                        # column, by setting row limit to be self.tile_m.
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit
                            else col0 - causal_row_offset - self.window_size_right
                        )
                        # TODO: do we need col_limit_sink?
                        row_limit_bot = col0 - causal_row_offset + self.window_size_left
                        for r in cutlass.range(
                            cute.size(tScS_mn.shape[0]), unroll_full=True
                        ):
                            row_idx = t0ScS_mn[r, 0][ROW]
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if row_idx < row_limit_top or row_idx > row_limit_bot
                                else acc_S_mn[r, c]
                            )

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Callable | None] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        check_q_boundary: bool = False,
    ) -> None:
        assert not (mask_causal and mask_local), (
            "mask_causal and mask_local cannot be both True"
        )
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(
            acc_shape if not self.swap_AB else acc_shape[::-1]
        )
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        # To handle edge cases of completely masked out rows where n_block_max = 0,
        # we treat negative n_blocks as 0th n_block
        # TODO: find more transparent solution
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n
        r2p = True
        if const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        # if tScS_t2r[i][1] >= seqlenk_col_limit:
                        #     acc_S[i] = -Float32.inf
                        # For some reason the 2 lines above generate really bad SASS
                        acc_S[i] = (
                            -Float32.inf
                            if tScS_t2r[i][1] >= seqlenk_col_limit
                            else acc_S[i]
                        )
                else:
                    mask_r2p(acc_S, seqlenk_col_limit, arch=100, rank1=True)

        elif const_expr(not mask_causal and not mask_local and mask_mod is not None):
            # Block sparse case w/ mask_mod
            has_fastdiv = const_expr(
                fastdiv_mods is not None
                and fastdiv_mods[0] is not None
                and fastdiv_mods[1] is not None
            )
            batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)

            ncol = const_expr(cute.size(tScS_t2r.shape))
            for i in cutlass.range_constexpr(ncol):
                row_coord = tScS_t2r[i][0] if not self.swap_AB else tScS_t2r[i][1]
                col_coord = tScS_t2r[i][1] if not self.swap_AB else tScS_t2r[i][0]
                global_row = row_coord + m_block * self.tile_m
                global_col = col_coord + n_block * self.tile_n

                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert head_divmod is not None
                    mask_row, head_offset = divmod(global_row, head_divmod)
                    head_idx_for_mod = (
                        head_idx * self.qhead_per_kvhead_packgqa + head_offset
                    )
                else:
                    head_idx_for_mod = head_idx
                    mask_row = global_row

                mask_row_for_mod = mask_row
                if const_expr(has_fastdiv and aux_tensors is not None):
                    if check_q_boundary:
                        _, mask_row_for_mod = divmod(mask_row, fastdiv_mods[0])
                global_col_for_mod = global_col
                if const_expr(has_fastdiv and mask_seqlen and aux_tensors is not None):
                    _, global_col_for_mod = divmod(global_col, fastdiv_mods[1])

                head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
                mask_row_ssa = utils.scalar_to_ssa(mask_row_for_mod, cutlass.Int32)
                kv_idx_ssa = utils.scalar_to_ssa(global_col_for_mod, cutlass.Int32)
                mask_value = mask_mod(
                    batch_idx_ssa,
                    head_idx_ssa,
                    mask_row_ssa,
                    kv_idx_ssa,
                    self.seqlen_info,
                    aux_tensors,
                )
                cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                acc_S[i] = acc_S[i] if cond else -Float32.inf
                if const_expr(mask_seqlen):
                    acc_S[i] = -Float32.inf if global_col >= self.seqlen_k else acc_S[i]
                if check_q_boundary:
                    acc_S[i] = -Float32.inf if mask_row >= self.seqlen_q else acc_S[i]

        else:  # Causal or local
            causal_row_offset = (
                1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q
            )
            row_idx = tScS_t2r[0][0] + m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                row_idx = row_idx // self.qhead_per_kvhead_packgqa
            if const_expr(mask_causal):
                col_limit_right = row_idx + causal_row_offset
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                # if cute.arch.thread_idx()[0] % 32 == 0:
                #     cute.printf("tidx = %d, tidx tmem = %d, row_idx = %d, col_limit_right = %d, causal_row_offset = %d\n", cute.arch.thread_idx()[0], thr_tmem_load.thr_idx, row_idx, col_limit_right, causal_row_offset)
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if const_expr(not r2p):
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = (
                            -Float32.inf
                            if tScS_t2r[i][1] >= col_limit_right
                            else acc_S[i]
                        )
                else:
                    mask_r2p(acc_S, col_limit_right, arch=100, rank1=True)
            else:
                local_row_offset_right = (
                    causal_row_offset + self.window_size_right
                    if const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - 1 - self.window_size_left
                    if const_expr(self.window_size_left is not None)
                    else None
                )
                if const_expr(self.window_size_right is not None):
                    col_limit_right = row_idx + local_row_offset_right
                else:
                    col_limit_right = self.tile_n
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                col_limit_left = (
                    row_idx + local_row_offset_left
                    if const_expr(self.window_size_left is not None)
                    else 0
                )
                if const_expr(not r2p):
                    # if cute.arch.thread_idx()[0] == 0 or cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", m_block, n_block, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        col_idx = tScS_t2r[i][1]
                        acc_S[i] = (
                            -Float32.inf
                            if col_idx >= col_limit_right or col_idx < col_limit_left
                            else acc_S[i]
                        )
                else:
                    # XOR-based R2P dual bound masking
                    mask_r2p_dual_bound(acc_S, col_limit_left, col_limit_right)

    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        mask_local: cutlass.Constexpr,
        mask_mod: cutlass.Constexpr[Callable | None] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
        is_full_block: bool = False,
        check_m_boundary: bool = True,
    ) -> None:
        """
        Backward pass: mask S = K @ Q.T where n_block tiles seqlen_k and m_block tiles seqlen_q.

        Coordinate conventio:
        - ROW corresponds to Q (m_block)
        - COL corresponds to KV (n_block)

        is_full_block: If True, skip mask_mod (all elements valid). Only apply seqlen masking.
        check_m_boundary: If False, skip seqlen_q boundary check (optimization for non-boundary m_blocks).
                          When iterating m_blocks in forward order, only the last m_block may be partial.
        """
        assert not (mask_causal and mask_local), (
            "mask_causal and mask_local cannot be both True"
        )
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        assert t0ScS_t2r[0][COL] == 0, "col0 == 0"
        thr_col_offset = tScS_t2r[0][COL]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset

        if const_expr(not mask_causal and not mask_local and mask_mod is not None):
            # Block sparse case with mask_mod (backward)
            #
            # Coordinate convention: ROW → Q (m_block), COL → KV (n_block).
            # These already account for swap_AB.
            #
            # FULL blocks: mask_mod returns True for all elements, so skip it.
            #   Still need seqlen bounds check (elements may be OOB on last m_block).
            # PARTIAL blocks: apply mask_mod element-wise, then seqlen bounds.
            if is_full_block:
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        # Entire tile is OOB for K
                        for i in cutlass.range(
                            cute.size(acc_S.shape), unroll_full=True
                        ):
                            acc_S[i] = -cutlass.Float32.inf
                    elif check_m_boundary:
                        # Last m_block: check Q and K boundaries
                        ncol = const_expr(cute.size(tScS_t2r.shape))
                        for i in cutlass.range_constexpr(ncol):
                            row_coord = tScS_t2r[i][ROW]
                            col_coord = tScS_t2r[i][COL]
                            global_q = row_coord + m_block * self.tile_m
                            global_kv = col_coord + n_block * self.tile_n
                            q_out_of_bounds = global_q >= self.seqlen_q
                            kv_out_of_bounds = global_kv >= self.seqlen_k
                            out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                            acc_S[i] = (
                                -cutlass.Float32.inf if out_of_bounds else acc_S[i]
                            )
            else:
                # Partial block
                has_fastdiv = const_expr(
                    fastdiv_mods is not None
                    and fastdiv_mods[0] is not None
                    and fastdiv_mods[1] is not None
                )
                wrap_aux_indices = const_expr(
                    has_fastdiv and mask_seqlen and const_expr(aux_tensors is not None)
                )
                batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32)

                ncol = const_expr(cute.size(tScS_t2r.shape))
                for i in cutlass.range_constexpr(ncol):
                    row_coord = tScS_t2r[i][ROW]
                    col_coord = tScS_t2r[i][COL]
                    global_q = row_coord + m_block * self.tile_m
                    global_kv = col_coord + n_block * self.tile_n

                    q_idx_for_mod = global_q
                    kv_idx_for_mod = global_kv
                    if const_expr(wrap_aux_indices):
                        _, q_idx_for_mod = divmod(global_q, fastdiv_mods[0])
                        _, kv_idx_for_mod = divmod(global_kv, fastdiv_mods[1])

                    q_idx_ssa = utils.scalar_to_ssa(q_idx_for_mod, cutlass.Int32)
                    kv_idx_ssa = utils.scalar_to_ssa(kv_idx_for_mod, cutlass.Int32)

                    mask_value = mask_mod(
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        self.seqlen_info,
                        aux_tensors,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    acc_S[i] = acc_S[i] if cond else -cutlass.Float32.inf

                    if const_expr(mask_seqlen):
                        # check_m_boundary=False skips q check for non-boundary m_blocks
                        q_out_of_bounds = check_m_boundary and (
                            global_q >= self.seqlen_q
                        )
                        kv_out_of_bounds = global_kv >= self.seqlen_k
                        out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                        acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]

        elif const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                if seqlenk_col_limit <= 0:
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = -cutlass.Float32.inf
        else:  # Causal or local
            thr_row_offset = tScS_t2r[0][ROW]
            seqlenq_row_limit = self.seqlen_q - m_block * self.tile_m - thr_row_offset
            causal_offset = seqlenq_row_limit - seqlenk_col_limit
            if const_expr(mask_causal):
                # tidx = cute.arch.thread_idx()[0] % 256
                # if tidx < 32:
                #     cute.printf("tidx = {}, {} {}, {} {}", tidx, tScS_t2r[0][0], tScS_t2r[0][1], tScS_t2r[1][0], tScS_t2r[1][1])
                row_limit_top = causal_offset
                if const_expr(mask_seqlen):
                    # If col is beyond the column limit, we want to mask out the entire
                    # column, by setting row limit to be self.tile_m.
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m
                r2p = True
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = (
                            -cutlass.Float32.inf
                            if t0ScS_t2r[i][ROW] < row_limit_top
                            else acc_S[i]
                        )
                else:
                    num_rep = cute.size(tScS_t2r, mode=[0])  # 16 or 32
                    mask_r2p_transposed(acc_S, row_limit_top, num_rep)
            else:
                if const_expr(self.window_size_right is not None):
                    row_limit_top = causal_offset - self.window_size_right
                else:
                    row_limit_top = 0
                if const_expr(self.window_size_left is not None):
                    row_limit_bot = causal_offset + self.window_size_left
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m
                for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                    row_idx = t0ScS_t2r[i][ROW]
                    local_mask = row_idx < row_limit_top
                    if const_expr(self.window_size_left is not None):
                        local_mask |= row_idx > row_limit_bot
                    acc_S[i] = -cutlass.Float32.inf if local_mask else acc_S[i]
