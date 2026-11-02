# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode index scores over TP-sharded FP8 or NVFP4 keys and exchange top-k
candidates."""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float16, Float32, Int32, Int64, Uint8, Uint32, cute
from cutlass.cute.nvgpu import cpasync, warp
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import EVICT_FIRST, cvt, mma_sync, simple_tma_copy
from vllm.models.minimax_m3.nvidia.ops.index_decode_score import (
    _fp8_to_f16_mma_fragments,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units

BLOCK = 128
SUPPORTED_TP = (2, 4)
DEFAULT_SPLIT_K = 128
MAX_K_TILES = 8192  # Matches the unified score buffer.
TOPK = 16
CANDIDATE_ROW_SIZE = 2 * TOPK + 1


class IndexDecodeScoreShardedKernel:
    PAGES = 4  # Shard pages per pipeline stage.
    BAR_MMA = 1
    num_stages = 2

    def __init__(
        self,
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
        world_size: int = 4,
        page_size: int = 128,
        use_pdl: bool = False,
        prefetch_history: bool = False,
    ):
        self.dtype = Float8E4M3FN
        self.num_heads = num_heads
        self.max_decode_query_len = max_decode_query_len
        self.split_k = split_k
        self.head_dim = head_dim
        self.world_size = world_size
        self.page_size = page_size
        self.use_pdl = use_pdl
        self.prefetch_history = prefetch_history

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [bs * runtime_decode_query_len, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, BLOCK // world_size, head_dim]
        block_table: cute.Tensor,  # [bs, max_pages]
        scores: cute.Tensor,  # [total_tokens, num_heads, score_columns]
        seq_lens: cute.Tensor,  # [bs]
        rank: Int32,  # this rank's token offset inside every block
        stream: CUstream,
    ):
        dtype = self.dtype
        num_heads = self.num_heads
        head_dim = self.head_dim
        num_stages = self.num_stages
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = num_heads * MAX_DQL
        ROWS = self.page_size // self.world_size
        assert BLOCK_Q <= 32
        assert head_dim * dtype.width == 128 * 8  # one 128 B row per token

        batch = seq_lens.shape[0]
        decode_query_len = gQ.shape[0] // batch
        grid = (batch, self.split_k, 1)
        block = (32 * 5, 1, 1)

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems = 128 * 8 // dtype.width

        sQ_layout = cute.make_layout(
            (MAX_DQL, num_heads, (elems, head_dim // elems)),
            stride=(elems, MAX_DQL * elems, (1, BLOCK_Q * elems)),
        )
        sQ_layout = cute.make_composed_layout(swizzle_128B, 0, sQ_layout)
        Q_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gQ, (None, None, elems)),
            sQ_layout,
            cta_tiler=(MAX_DQL, num_heads, head_dim),
        )

        # Page p of stage s occupies buffer s * PAGES + p.
        sK_layout = cute.make_layout(
            (1, ROWS, (elems, head_dim // elems), self.PAGES * num_stages),
            stride=(0, elems, (1, ROWS * elems), ROWS * head_dim),
        )
        sK_layout = cute.make_composed_layout(swizzle_128B, 0, sK_layout)
        K_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gK_cache, (None, None, elems)),
            sK_layout,
            cta_tiler=(1, ROWS, head_dim),
        )

        self.kernel(
            Q_tma,
            K_tma,
            block_table,
            scores,
            seq_lens,
            decode_query_len,
            rank,
        ).launch(grid=grid, block=block, stream=stream, use_pdl=self.use_pdl)

    @cute.kernel
    def kernel(
        self,
        Q_tma: cpasync.TmaInfo,
        K_tma: cpasync.TmaInfo,
        block_table: cute.Tensor,
        scores: cute.Tensor,
        seq_lens: cute.Tensor,
        decode_query_len,
        rank,
    ):
        batch_id, split_id, _ = cute.arch.block_idx()
        _, split_k, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        NUM_HEADS = self.num_heads
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = NUM_HEADS * MAX_DQL
        PAGES = self.PAGES
        TP = self.world_size
        ROWS = self.page_size // TP
        head_dim = self.head_dim
        dtype = self.dtype
        MMA_N = 8
        num_stages = self.num_stages
        Q_TILES = cute.ceil_div(BLOCK_Q, MMA_N)
        row_elems = 128 * 8 // dtype.width
        page_bytes = ROWS * head_dim * (dtype.width // 8)

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            dtype,
            K_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=K_tma.smem_layout.inner,
        )
        # alias sQ with the 1st page of the 1st stage of sK
        sQ_tma = cute.make_tensor(
            sK[0, None, None, 0].iterator, layout=Q_tma.smem_layout.outer
        )
        sQ = cute.coalesce(
            cute.group_modes(sQ_tma, 0, 2),
            target_profile=(BLOCK_Q, (row_elems, head_dim // row_elems)),
        )

        # Keep each column's four page maxima contiguous.
        EPI_Q = Q_TILES * MMA_N
        epi_buffer = smem.allocate_tensor(
            Float32, cute.make_layout((EPI_Q, PAGES), stride=(PAGES, 1)), 16
        )

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        if cutlass.const_expr(self.use_pdl and not self.prefetch_history):
            cute.arch.griddepcontrol_wait()
        seqlen = seq_lens[batch_id]
        num_blocks = cute.ceil_div(seqlen, self.page_size)
        num_super = cute.ceil_div(num_blocks, PAGES)
        prefetched = Int32(0)

        if split_id < num_super:
            if warp_id == 0:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(num_stages):
                        cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                        cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
                    cute.arch.mbarrier_init_fence()
            elif warp_id == 1:
                cpasync.prefetch_descriptor(Q_tma.atom)
                cpasync.prefetch_descriptor(K_tma.atom)
            cute.arch.sync_threads()

            if cutlass.const_expr(self.prefetch_history):  # noqa: SIM102
                # Stage 0 aliases Q. Only stage 1 may hold K before the wait.
                if (split_id + 1) * PAGES * self.page_size <= seqlen - decode_query_len:
                    prefetched = Int32(1)
                    if warp_id == 4:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                tma_full_mbar + 1, PAGES * page_bytes
                            )
                        for p in cutlass.range_constexpr(PAGES):
                            page_id = block_table[batch_id, split_id * PAGES + p]
                            simple_tma_copy(
                                K_tma.atom,
                                K_tma.tma_tensor[page_id, None, None],
                                sK[None, None, None, PAGES + p],
                                tma_full_mbar + 1,
                                cache_policy=EVICT_FIRST,
                            )
        if cutlass.const_expr(self.use_pdl):
            if cutlass.const_expr(self.prefetch_history):
                cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        if split_id < num_super:
            if warp_id == 4:
                # TMA warp
                tma_stage = 0
                tma_parity = 1

                gQ_tile = cute.local_tile(
                    cute.domain_offset(
                        (batch_id * decode_query_len, 0, 0),
                        Q_tma.tma_tensor,
                    ),
                    tiler=(MAX_DQL, NUM_HEADS, head_dim),
                    coord=(0, 0, 0),
                )
                cute.arch.mbarrier_wait(tma_empty_mbar, tma_parity)
                with cute.arch.elect_one():
                    Q_size = BLOCK_Q * head_dim * (dtype.width // 8)
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_full_mbar, Q_size)
                simple_tma_copy(Q_tma.atom, gQ_tile, sQ_tma, tma_full_mbar)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    tma_parity ^= 1

                for super_id in range(split_id, num_super, split_k):
                    k_mbar = tma_full_mbar + tma_stage
                    if super_id != split_id or prefetched == 0:
                        cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                k_mbar, PAGES * page_bytes
                            )
                        for p in cutlass.range_constexpr(PAGES):
                            # Reload the last page; causal masking yields -inf.
                            tile = super_id * PAGES + p
                            if tile >= num_blocks:
                                tile = num_blocks - 1
                            page_id = block_table[batch_id, tile]
                            simple_tma_copy(
                                K_tma.atom,
                                K_tma.tma_tensor[page_id, None, None],
                                sK[None, None, None, tma_stage * PAGES + p],
                                k_mbar,
                                cache_policy=EVICT_FIRST,
                            )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

            else:
                # MMA warp w handles shard page w of each stage.
                sK_pages = cute.make_tensor(
                    sK.iterator,
                    cute.make_layout(
                        (ROWS, (row_elems, head_dim // row_elems), PAGES, num_stages),
                        stride=(
                            row_elems,
                            (1, ROWS * row_elems),
                            ROWS * head_dim,
                            PAGES * ROWS * head_dim,
                        ),
                    ),
                )
                sK_warp = sK_pages[None, None, warp_id, None]
                q_start = seqlen - decode_query_len

                elems = 128 // dtype.width  # 16B
                MMA_K = 32 * 8 // dtype.width  # 32B

                sK_ldsm = cute.zipped_divide(
                    sK_warp, (16, cute.make_layout((elems, 2)), 1)
                )
                sQ_ldsm = cute.zipped_divide(sQ, (MMA_N, cute.make_layout((elems, 4))))

                sK_ldsm = sK_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]
                sQ_ldsm = sQ_ldsm[(lane_id % MMA_N, (None, lane_id // 8)), None]

                ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
                ldsm_atom = cute.make_copy_atom(ldsm_op, dtype)

                rQ = cute.make_rmem_tensor(
                    ((elems // 2, 2), head_dim // (MMA_K * 2), Q_TILES), dtype
                )
                rK = cute.make_rmem_tensor(
                    (elems, ROWS // 16, head_dim // MMA_K), dtype
                )
                rC = cute.make_rmem_tensor((4, ROWS // 16, Q_TILES), Float32)

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar, 0)
                cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)
                for q in cutlass.range_constexpr(Q_TILES):
                    cute.copy(ldsm_atom, sQ_ldsm[None, (q, None)], rQ[None, None, q])
                cute.arch.mbarrier_arrive(tma_empty_mbar)

                tma_stage = 1 % self.num_stages
                tma_parity = 0
                if tma_stage == 0:
                    tma_parity ^= 1

                rQ_f16 = cute.make_rmem_tensor(
                    (4, head_dim // MMA_K, Q_TILES, 2), Float16
                )
                q_lower, q_upper = _fp8_to_f16_mma_fragments(rQ)
                rQ_f16[None, None, None, 0].store(q_lower.load())
                rQ_f16[None, None, None, 1].store(q_upper.load())

                for super_id in range(split_id, num_super, split_k):
                    rC.fill(0.0)

                    if warp_id == 0:
                        cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_parity)
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

                    for k in cutlass.range_constexpr(head_dim // MMA_K):
                        cute.copy(
                            ldsm_atom,
                            sK_ldsm[None, (None, k, tma_stage)],
                            rK[None, None, k],
                        )
                        for m in cutlass.range_constexpr(ROWS // 16):
                            rK_lower, rK_upper = _fp8_to_f16_mma_fragments(
                                rK[None, m, k]
                            )
                            for n in cutlass.range_constexpr(Q_TILES):
                                rC[None, m, n] = mma_sync(
                                    rK_lower,
                                    rQ_f16[None, k, n, 0],
                                    rC[None, m, n],
                                )
                                rC[None, m, n] = mma_sync(
                                    rK_upper,
                                    rQ_f16[None, k, n, 1],
                                    rC[None, m, n],
                                )

                    cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                    self.store_page_maxima(
                        rC,
                        epi_buffer,
                        scores,
                        batch_id,
                        super_id,
                        warp_id,
                        lane_id,
                        q_start,
                        decode_query_len,
                        rank,
                    )

                    tma_stage = (tma_stage + 1) % self.num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

    @cute.jit
    def store_page_maxima(
        self,
        rC: cute.Tensor,  # [4, ROWS // 16, Q_TILES] m16n8 accumulators
        epi_buffer: cute.Tensor,
        scores: cute.Tensor,
        batch_id,
        super_id,
        warp_id,
        lane_id,
        q_start,
        decode_query_len,
        rank,
    ):
        """Mask, reduce this warp's shard page to B128 maxima, and store them."""
        ROWS = self.page_size // self.world_size
        Q_TILES = cute.ceil_div(self.num_heads * self.max_decode_query_len, 8)
        k_base = self.page_k_base(super_id, warp_id, rank)
        rMax = cute.make_rmem_tensor((2, Q_TILES), Float32)
        rMax.fill(float("-inf"))
        for m in cutlass.range_constexpr(ROWS // 16):
            self.fold_tile_maxima(rC[None, m, None], rMax, m, lane_id, k_base, q_start)
        self.store_maxima(
            rMax,
            epi_buffer,
            scores,
            batch_id,
            super_id,
            warp_id,
            lane_id,
            decode_query_len,
        )

    @cute.jit
    def page_k_base(self, super_id, warp_id, rank):
        """Token position of this warp's first shard row."""
        tile = super_id * self.PAGES + warp_id
        # P128 interleaves rows; P256 assigns contiguous rank slices.
        k_base = tile * self.page_size + rank
        if cutlass.const_expr(self.page_size == 256):
            k_base = tile * self.page_size + rank * (self.page_size // self.world_size)
        return k_base

    @cute.jit
    def fold_tile_maxima(
        self,
        rC_m: cute.Tensor,  # [4, Q_TILES] m16n8 accumulators of rows 16m..16m+15
        rMax: cute.Tensor,  # [2, Q_TILES] running maxima of this lane's columns
        m,
        lane_id,
        k_base,
        q_start,
    ):
        """Causally mask a 16-row tile and fold it into the page maxima."""
        MAX_DQL = self.max_decode_query_len
        TP = self.world_size
        Q_TILES = cute.ceil_div(self.num_heads * MAX_DQL, 8)
        for q in cutlass.range_constexpr(Q_TILES):
            for v in cutlass.range_constexpr(4):
                # Accumulator v is row 8 * (v // 2) + lane // 4, column v % 2.
                i = m * 2 + v // 2
                j = v % 2
                col = q * 8 + (lane_id % 4) * 2 + j
                q_pos = q_start + col % MAX_DQL
                k_pos = k_base + (i * 8 + lane_id // 4) * TP
                if cutlass.const_expr(self.page_size == 256):
                    k_pos = k_base + i * 8 + lane_id // 4
                rC_m[v, q] = rC_m[v, q] if q_pos >= k_pos else float("-inf")
                rMax[j, q] = cute.arch.fmax(rMax[j, q], rC_m[v, q])

    @cute.jit
    def store_maxima(
        self,
        rMax: cute.Tensor,  # [2, Q_TILES] per-lane page maxima
        epi_buffer: cute.Tensor,
        scores: cute.Tensor,
        batch_id,
        super_id,
        warp_id,
        lane_id,
        decode_query_len,
    ):
        """Reduce page maxima across the warp's rows and store each B128 block."""
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = self.num_heads * MAX_DQL
        PAGES = self.PAGES
        MMA_N = 8
        Q_TILES = cute.ceil_div(BLOCK_Q, MMA_N)

        for q in cutlass.range_constexpr(Q_TILES):
            # warp-reduction among lanes 0,4,8,12,...
            for i in cutlass.range_constexpr(3):
                offset = 4 << i
                other0 = cute.arch.shuffle_sync_bfly(
                    rMax[0, q], offset=offset, mask=-1, mask_and_clamp=31
                )
                other1 = cute.arch.shuffle_sync_bfly(
                    rMax[1, q], offset=offset, mask=-1, mask_and_clamp=31
                )
                rMax[0, q] = cute.arch.fmax(rMax[0, q], other0)
                rMax[1, q] = cute.arch.fmax(rMax[1, q], other1)

            # lanes 0..3 hold columns q * 8 + 2 * lane + {0, 1}
            if lane_id * 2 < MMA_N:
                epi_buffer[q * MMA_N + lane_id * 2 + 0, warp_id] = rMax[0, q]
                epi_buffer[q * MMA_N + lane_id * 2 + 1, warp_id] = rMax[1, q]
        cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

        # Padded pages are -inf.
        if warp_id == 0 and lane_id < BLOCK_Q:
            head_id = lane_id // MAX_DQL
            q_local_pos = lane_id - head_id * MAX_DQL
            if q_local_pos < decode_query_len:
                t = batch_id * decode_query_len + q_local_pos
                for p in cutlass.range_constexpr(PAGES):
                    block_id = super_id * PAGES + p
                    scores[t, head_id, block_id] = epi_buffer[lane_id, p]

    @cache
    @staticmethod
    def compile(
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
        world_size: int = 4,
        page_size: int = 128,
        use_pdl: bool = False,
        prefetch_history: bool = False,
        score_columns: int = MAX_K_TILES,
    ):
        bs = cute.sym_int()
        total_tokens = cute.sym_int()
        dtype = Float8E4M3FN

        q = make_fake_tensor(
            dtype, (total_tokens, num_heads, head_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            dtype, (cute.sym_int(), page_size // world_size, head_dim), divisibility=16
        )
        block_table = make_fake_tensor(Int32, (bs, cute.sym_int()), divisibility=1)
        scores = make_fake_tensor(
            Float32, (total_tokens, num_heads, score_columns), divisibility=4
        )
        seq_lens = make_fake_tensor(Int32, (bs,), divisibility=1)
        kernel = IndexDecodeScoreShardedKernel(
            num_heads,
            max_decode_query_len,
            split_k,
            head_dim,
            world_size,
            page_size,
            use_pdl,
            prefetch_history,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            q,
            k_cache,
            block_table,
            scores,
            seq_lens,
            Int32(0),
            stream,
            options="--enable-tvm-ffi",
        )


class IndexDecodeScoreShardedNvfp4Kernel(IndexDecodeScoreShardedKernel):
    """FP8 Q against NVFP4 shard pages of M3.1's index cache.

    A shard page holds this rank's rows of an NVFP4 index page:
    ``[ROWS x 64 B e2m1 codes][ROWS x 8 B e4m3 group scales]``, the low nibble
    the even dim and one scale per 16 dims. With ``round_k`` each K value is
    rounded once to e4m3, as the SM100/SM103 Q8KV4 decode indexer scores it;
    without it K keeps the exact e2m1 x scale product, as on SM107. K is fed
    as FP16 to the same ``mma.sync`` as the FP8 kernel. ldmatrix gives each
    lane eight consecutive codes per 16 B row slice, so every lane reads the
    matching eight Q bytes itself instead of ldmatrix-loading Q.
    """

    DATA_BYTES = 64
    SCALE_BYTES = 8

    def __init__(self, *args, round_k: bool = True, **kw):
        super().__init__(*args, **kw)
        self.round_k = round_k

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [bs * runtime_decode_query_len, num_heads, head_dim]
        gK_data: cute.Tensor,  # [num_pages, ROWS, 64] uint8, page pitch strided
        gK_scale: cute.Tensor,  # [num_pages, ROWS // 2, 16] uint8, two rows each
        block_table: cute.Tensor,  # [bs, max_pages]
        scores: cute.Tensor,  # [total_tokens, num_heads, score_columns]
        seq_lens: cute.Tensor,  # [bs]
        rank: Int32,  # this rank's token offset inside every block
        stream: CUstream,
    ):
        dtype = self.dtype
        num_heads = self.num_heads
        head_dim = self.head_dim
        num_stages = self.num_stages
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = num_heads * MAX_DQL
        ROWS = self.page_size // self.world_size
        DATA = self.DATA_BYTES
        assert BLOCK_Q <= 32
        assert head_dim == 2 * DATA

        batch = seq_lens.shape[0]
        decode_query_len = gQ.shape[0] // batch
        grid = (batch, self.split_k, 1)
        block = (32 * 5, 1, 1)

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        elems = 128 * 8 // dtype.width

        sQ_layout = cute.make_layout(
            (MAX_DQL, num_heads, (elems, head_dim // elems)),
            stride=(elems, MAX_DQL * elems, (1, BLOCK_Q * elems)),
        )
        sQ_layout = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, sQ_layout)
        Q_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gQ, (None, None, elems)),
            sQ_layout,
            cta_tiler=(MAX_DQL, num_heads, head_dim),
        )

        # 64 B code rows take the 64 B swizzle, so ldmatrix is conflict-free.
        sK_layout = cute.make_layout(
            (1, ROWS, (DATA, 1), self.PAGES * num_stages),
            stride=(0, DATA, (1, ROWS * DATA), ROWS * DATA),
        )
        sK_layout = cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0, sK_layout)
        K_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gK_data, (None, None, DATA)),
            sK_layout,
            cta_tiler=(1, ROWS, DATA),
        )

        # TMA boxes need 16 B rows: one box row holds two token rows of scales.
        sS_layout = cute.make_layout(
            (1, ROWS // 2, 16, self.PAGES * num_stages),
            stride=(0, 16, 1, ROWS * self.SCALE_BYTES),
        )
        sS_layout = cute.make_composed_layout(cute.make_swizzle(0, 4, 3), 0, sS_layout)
        S_tma = cpasync.make_tiled_tma_atom(
            tma_g2s, gK_scale, sS_layout, cta_tiler=(1, ROWS // 2, 16)
        )

        self.kernel(
            Q_tma,
            K_tma,
            S_tma,
            block_table,
            scores,
            seq_lens,
            decode_query_len,
            rank,
        ).launch(grid=grid, block=block, stream=stream, use_pdl=self.use_pdl)

    @cute.kernel
    def kernel(
        self,
        Q_tma: cpasync.TmaInfo,
        K_tma: cpasync.TmaInfo,
        S_tma: cpasync.TmaInfo,
        block_table: cute.Tensor,
        scores: cute.Tensor,
        seq_lens: cute.Tensor,
        decode_query_len,
        rank,
    ):
        batch_id, split_id, _ = cute.arch.block_idx()
        _, split_k, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        NUM_HEADS = self.num_heads
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = NUM_HEADS * MAX_DQL
        PAGES = self.PAGES
        ROWS = self.page_size // self.world_size
        DATA = self.DATA_BYTES
        SCALE = self.SCALE_BYTES
        head_dim = self.head_dim
        dtype = self.dtype
        MMA_N = 8
        num_stages = self.num_stages
        Q_TILES = cute.ceil_div(BLOCK_Q, MMA_N)
        row_elems = 128 * 8 // dtype.width
        page_bytes = ROWS * (DATA + SCALE)

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            Uint8,
            K_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=K_tma.smem_layout.inner,
        )
        sS = smem.allocate_tensor(
            Uint8,
            S_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=S_tma.smem_layout.inner,
        )
        # Q keeps its own 128 B-swizzled buffer; it still rides stage 0's barriers.
        sQ_tma = smem.allocate_tensor(
            dtype,
            Q_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=Q_tma.smem_layout.inner,
        )
        sQ = cute.coalesce(
            cute.group_modes(sQ_tma, 0, 2),
            target_profile=(BLOCK_Q, (row_elems, head_dim // row_elems)),
        )

        # Keep each column's four page maxima contiguous.
        EPI_Q = Q_TILES * MMA_N
        epi_buffer = smem.allocate_tensor(
            Float32, cute.make_layout((EPI_Q, PAGES), stride=(PAGES, 1)), 16
        )

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        if cutlass.const_expr(self.use_pdl and not self.prefetch_history):
            cute.arch.griddepcontrol_wait()
        seqlen = seq_lens[batch_id]
        num_blocks = cute.ceil_div(seqlen, self.page_size)
        num_super = cute.ceil_div(num_blocks, PAGES)
        prefetched = Int32(0)

        if split_id < num_super:
            if warp_id == 0:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(num_stages):
                        cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                        cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
                    cute.arch.mbarrier_init_fence()
            elif warp_id == 1:
                cpasync.prefetch_descriptor(Q_tma.atom)
                cpasync.prefetch_descriptor(K_tma.atom)
                cpasync.prefetch_descriptor(S_tma.atom)
            cute.arch.sync_threads()

            if cutlass.const_expr(self.prefetch_history):  # noqa: SIM102
                # Stage 0 carries Q. Only stage 1 may hold K before the wait.
                if (split_id + 1) * PAGES * self.page_size <= seqlen - decode_query_len:
                    prefetched = Int32(1)
                    if warp_id == 4:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                tma_full_mbar + 1, PAGES * page_bytes
                            )
                        for p in cutlass.range_constexpr(PAGES):
                            page_id = block_table[batch_id, split_id * PAGES + p]
                            simple_tma_copy(
                                K_tma.atom,
                                K_tma.tma_tensor[page_id, None, None],
                                sK[None, None, None, PAGES + p],
                                tma_full_mbar + 1,
                                cache_policy=EVICT_FIRST,
                            )
                            simple_tma_copy(
                                S_tma.atom,
                                S_tma.tma_tensor[page_id, None, None],
                                sS[None, None, None, PAGES + p],
                                tma_full_mbar + 1,
                                cache_policy=EVICT_FIRST,
                            )
        if cutlass.const_expr(self.use_pdl):
            if cutlass.const_expr(self.prefetch_history):
                cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        if split_id < num_super:
            if warp_id == 4:
                # TMA warp
                tma_stage = 0
                tma_parity = 1

                gQ_tile = cute.local_tile(
                    cute.domain_offset(
                        (batch_id * decode_query_len, 0, 0),
                        Q_tma.tma_tensor,
                    ),
                    tiler=(MAX_DQL, NUM_HEADS, head_dim),
                    coord=(0, 0, 0),
                )
                cute.arch.mbarrier_wait(tma_empty_mbar, tma_parity)
                with cute.arch.elect_one():
                    Q_size = BLOCK_Q * head_dim * (dtype.width // 8)
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_full_mbar, Q_size)
                simple_tma_copy(Q_tma.atom, gQ_tile, sQ_tma, tma_full_mbar)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    tma_parity ^= 1

                for super_id in range(split_id, num_super, split_k):
                    k_mbar = tma_full_mbar + tma_stage
                    if super_id != split_id or prefetched == 0:
                        cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                k_mbar, PAGES * page_bytes
                            )
                        for p in cutlass.range_constexpr(PAGES):
                            # Reload the last page; causal masking yields -inf.
                            tile = super_id * PAGES + p
                            if tile >= num_blocks:
                                tile = num_blocks - 1
                            page_id = block_table[batch_id, tile]
                            simple_tma_copy(
                                K_tma.atom,
                                K_tma.tma_tensor[page_id, None, None],
                                sK[None, None, None, tma_stage * PAGES + p],
                                k_mbar,
                                cache_policy=EVICT_FIRST,
                            )
                            simple_tma_copy(
                                S_tma.atom,
                                S_tma.tma_tensor[page_id, None, None],
                                sS[None, None, None, tma_stage * PAGES + p],
                                k_mbar,
                                cache_policy=EVICT_FIRST,
                            )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

            else:
                # MMA warp w handles shard page w of each stage.
                sK_pages = cute.make_tensor(
                    sK.iterator,
                    cute.make_layout(
                        (ROWS, DATA, PAGES, num_stages),
                        stride=(DATA, 1, ROWS * DATA, PAGES * ROWS * DATA),
                    ),
                )
                sK_warp = sK_pages[None, None, warp_id, None]
                # A row's eight scales as two words, one per 32 B code span.
                WORDS = SCALE // 4
                sS_pages = cute.make_tensor(
                    cute.make_ptr(
                        Uint32,
                        sS.iterator.toint(),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout(
                        (ROWS, WORDS, PAGES, num_stages),
                        stride=(WORDS, 1, ROWS * WORDS, PAGES * ROWS * WORDS),
                    ),
                )
                sS_warp = sS_pages[None, None, warp_id, None]
                q_start = seqlen - decode_query_len

                # Each ldmatrix x4 covers 16 rows x 32 B: matrices are rows
                # {0-7, 8-15} of the low then high 16 B, four bytes per lane.
                sK_ldsm = cute.zipped_divide(
                    sK_warp, (16, cute.make_layout((16, 2)), 1)
                )
                sK_ldsm = sK_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]

                ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
                ldsm_atom = cute.make_copy_atom(ldsm_op, Uint8)

                SPANS = DATA // 32  # 32 B ldmatrix spans per row
                # One 16-row tile at a time, folded into the page maxima as it
                # finishes, so the page's accumulators never all stay live.
                rK = cute.make_rmem_tensor(16, Uint8)
                rC = cute.make_rmem_tensor((4, Q_TILES), Float32)
                rMax = cute.make_rmem_tensor((2, Q_TILES), Float32)

                # Lane j of a quad owns codes [8j, 8j + 8) of every 32-code slice.
                quad = lane_id % 4
                row0 = lane_id // 4
                # Span k's word holds groups 4k..4k+3; slice h uses byte 2h + j // 2
                # of row r's word (prmt bytes 0-3), then of row r + 8's (4-7).
                scale_sel = Uint32(quad // 2) * 0x1111 + 0x6420

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar, 0)
                cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)
                # Q words [w, slice, q]: f16 pairs of Q bytes slice*32 + 8j + 2w.
                rQ = cute.make_rmem_tensor((4, 2 * SPANS, Q_TILES), Uint32)
                rQ8 = cute.make_rmem_tensor(8, dtype)
                rQ8_u32 = cute.recast_tensor(rQ8, Uint32)
                for q in cutlass.range_constexpr(Q_TILES):
                    col = q * MMA_N + row0
                    for s in cutlass.range_constexpr(2 * SPANS):
                        rQ8_u32[0] = Uint32(0)
                        rQ8_u32[1] = Uint32(0)
                        # Columns past BLOCK_Q are padding; keep them finite.
                        if col < BLOCK_Q:
                            for e in cutlass.range_constexpr(8):
                                rQ8[e] = sQ[col, s * 32 + quad * 8 + e]
                        for w in cutlass.range_constexpr(2):
                            pair = cvt.fp8x4_to_fp16x4(rQ8_u32[w])
                            rQ[w * 2, s, q] = pair[0]
                            rQ[w * 2 + 1, s, q] = pair[1]
                cute.arch.mbarrier_arrive(tma_empty_mbar)

                tma_stage = 1 % self.num_stages
                tma_parity = 0
                if tma_stage == 0:
                    tma_parity ^= 1

                rA = cute.make_rmem_tensor(8, Float16)
                rA_u32 = cute.recast_tensor(rA, Uint32)
                rB = cute.make_rmem_tensor(4, Float16)
                rB_u32 = cute.recast_tensor(rB, Uint32)

                for super_id in range(split_id, num_super, split_k):
                    if warp_id == 0:
                        cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_parity)
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

                    k_base = self.page_k_base(super_id, warp_id, rank)
                    rMax.fill(float("-inf"))
                    # One 16-row tile at a time: unrolled, the eight tiles of a
                    # P256 page at TP2 keep too much live across tiles.
                    for m in cutlass.range(ROWS // 16, unroll=1):
                        rC.fill(0.0)
                        for k in cutlass.range_constexpr(SPANS):
                            cute.copy(ldsm_atom, sK_ldsm[None, (m, k, tma_stage)], rK)
                            # Words: rows {r, r + 8} of the low, then high 16 B.
                            rK_u32 = cute.recast_tensor(rK, Uint32)
                            # {s, s} f16x2 scales: [2 * ri + h] is row r + 8 * ri's
                            # for slice 2k + h.
                            scales = cvt.e4m3x4_select_to_fp16x2x4(
                                sS_warp[m * 16 + row0, k, tma_stage],
                                sS_warp[m * 16 + 8 + row0, k, tma_stage],
                                scale_sel,
                            )
                            for h in cutlass.range_constexpr(2):
                                s = k * 2 + h
                                k0 = self.k_fp16x8(rK_u32[h * 2], scales[h])
                                k1 = self.k_fp16x8(rK_u32[h * 2 + 1], scales[2 + h])
                                # One m16n8k16 per four codes of each row.
                                for half in cutlass.range_constexpr(2):
                                    rA_u32[0] = k0[half * 2]
                                    rA_u32[1] = k1[half * 2]
                                    rA_u32[2] = k0[half * 2 + 1]
                                    rA_u32[3] = k1[half * 2 + 1]
                                    for n in cutlass.range_constexpr(Q_TILES):
                                        rB_u32[0] = rQ[half * 2, s, n]
                                        rB_u32[1] = rQ[half * 2 + 1, s, n]
                                        rC[None, n] = mma_sync(rA, rB, rC[None, n])
                        if m == ROWS // 16 - 1:
                            # The stage's last shared memory read is done.
                            cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)
                        self.fold_tile_maxima(rC, rMax, m, lane_id, k_base, q_start)

                    self.store_maxima(
                        rMax,
                        epi_buffer,
                        scores,
                        batch_id,
                        super_id,
                        warp_id,
                        lane_id,
                        decode_query_len,
                    )

                    tma_stage = (tma_stage + 1) % self.num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

    @cute.jit
    def k_fp16x8(self, codes, scale):
        """Eight codes of one row times an f16x2 {scale, scale}: four f16x2 words."""
        if cutlass.const_expr(self.round_k):
            k = cvt.nvfp4x8_to_e4m3_fp16x8(codes, scale)
        else:
            k = cvt.nvfp4x8_to_fp16x8(codes, scale)
        return k

    @cache
    @staticmethod
    def compile(
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
        world_size: int = 4,
        page_size: int = 128,
        use_pdl: bool = False,
        prefetch_history: bool = False,
        score_columns: int = MAX_K_TILES,
        round_k: bool = True,
    ):
        bs = cute.sym_int()
        total_tokens = cute.sym_int()
        rows = page_size // world_size

        q = make_fake_tensor(
            Float8E4M3FN, (total_tokens, num_heads, head_dim), divisibility=16
        )
        # Shard pages may sit inside larger compound pages; only the pitch varies.
        k_data = make_fake_tensor(
            Uint8,
            (cute.sym_int(), rows, IndexDecodeScoreShardedNvfp4Kernel.DATA_BYTES),
            divisibility=16,
        )
        k_scale = make_fake_tensor(
            Uint8, (cute.sym_int(), rows // 2, 16), divisibility=16
        )
        block_table = make_fake_tensor(Int32, (bs, cute.sym_int()), divisibility=1)
        scores = make_fake_tensor(
            Float32, (total_tokens, num_heads, score_columns), divisibility=4
        )
        seq_lens = make_fake_tensor(Int32, (bs,), divisibility=1)
        kernel = IndexDecodeScoreShardedNvfp4Kernel(
            num_heads,
            max_decode_query_len,
            split_k,
            head_dim,
            world_size,
            page_size,
            use_pdl,
            prefetch_history,
            round_k=round_k,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            q,
            k_data,
            k_scale,
            block_table,
            scores,
            seq_lens,
            Int32(0),
            stream,
            options="--enable-tvm-ffi",
        )


def nvfp4_shard_views(
    index_kv_shard: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``[pages, ROWS, 72]`` uint8 shard pages into code and scale views.

    A page is ``ROWS x 64`` code bytes, then ``ROWS x 8`` scale bytes; the page
    pitch may exceed ``ROWS * 72``. Returns ``[pages, ROWS, 64]`` codes and
    ``[pages, ROWS // 2, 16]`` scales, two token rows per 16 B.
    """
    pages, rows, width = index_kv_shard.shape
    data = IndexDecodeScoreShardedNvfp4Kernel.DATA_BYTES
    scale = IndexDecodeScoreShardedNvfp4Kernel.SCALE_BYTES
    pitch = index_kv_shard.stride(0)
    if (
        index_kv_shard.dtype != torch.uint8
        or width != data + scale
        or index_kv_shard.stride(2) != 1
        or index_kv_shard.stride(1) != width
        or pitch < rows * width
        or pitch % 16
        or index_kv_shard.data_ptr() % 16
    ):
        raise ValueError(
            "NVFP4 shard pages must be 16 B-aligned uint8 [pages, ROWS, 72] views"
        )
    offset = index_kv_shard.storage_offset()
    codes = index_kv_shard.as_strided((pages, rows, data), (pitch, data, 1), offset)
    scales = index_kv_shard.as_strided(
        (pages, rows // 2, 2 * scale), (pitch, 2 * scale, 1), offset + rows * data
    )
    return codes, scales


@cache
def nvfp4_k_rounds_to_e4m3(device_index: int) -> bool:
    """Whether the stock Q8KV4 decode indexer rounds NVFP4 K to e4m3 here.

    SM100/SM103 round each e2m1 x scale product once to e4m3 for an FP8 MMA;
    SM107 has no QMUL4 and keeps the exact FP16 product (MSA #41).
    """
    return not current_platform.is_device_capability(107, device_index)


def minimax_m3_index_decode_score_sharded_cutedsl(
    idx_q: torch.Tensor,
    index_kv_shard: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    rank: int,
    decode_query_len: int,
    max_decode_query_len: int,
    score_out: torch.Tensor,
    split_k: int = DEFAULT_SPLIT_K,
    *,
    page_size: int = 128,
    world_size: int | None = None,
    use_pdl: bool = False,
    prefetch_history: bool = False,
    nvfp4_round_k: bool | None = None,
) -> None:
    """Write B128 maxima for interleaved P128 or contiguous P256 shards.

    ``index_kv_shard`` is FP8 ``[pages, ROWS, 128]`` or, for NVFP4 index
    caches, uint8 ``[pages, ROWS, 72]`` pages (see ``nvfp4_shard_views``).
    History prefetch requires metadata fenced by a non-PDL cache writer.
    NVFP4 K is rounded to e4m3 where the stock decode indexer rounds it, unless
    ``nvfp4_round_k`` says otherwise.
    """
    nvfp4 = index_kv_shard.dtype == torch.uint8
    if idx_q.dtype != torch.float8_e4m3fn or not (
        nvfp4 or index_kv_shard.dtype == idx_q.dtype
    ):
        raise TypeError("sharded index decode score needs FP8 Q and FP8 or NVFP4 K")
    total_tokens, num_heads, head_dim = idx_q.shape
    batch = block_table.shape[0]
    assert page_size in (128, 256)
    if world_size is None:
        world_size = page_size // index_kv_shard.shape[1]
    assert world_size in SUPPORTED_TP
    assert index_kv_shard.shape[1] == page_size // world_size
    assert total_tokens == batch * decode_query_len
    assert 1 <= decode_query_len <= max_decode_query_len
    assert num_heads * max_decode_query_len <= 32
    assert 0 <= rank < world_size and num_heads % world_size == 0
    assert score_out.ndim == 3 and score_out.shape[:2] == (total_tokens, num_heads)
    score_columns = score_out.shape[-1]
    assert 0 < score_columns <= MAX_K_TILES
    assert score_columns % IndexDecodeScoreShardedKernel.PAGES == 0
    assert score_out.dtype == torch.float32 and score_out.is_contiguous()
    kernel_cls = (
        IndexDecodeScoreShardedNvfp4Kernel if nvfp4 else IndexDecodeScoreShardedKernel
    )
    # The NVFP4 kernel also compiles in how it rounds K.
    nvfp4_args: tuple[bool, ...] = ()
    if nvfp4:
        if nvfp4_round_k is None:
            nvfp4_round_k = nvfp4_k_rounds_to_e4m3(index_kv_shard.device.index)
        nvfp4_args = (nvfp4_round_k,)
    kernel = kernel_cls.compile(
        num_heads,
        max_decode_query_len,
        split_k,
        head_dim,
        world_size,
        page_size,
        use_pdl,
        prefetch_history,
        score_columns,
        *nvfp4_args,
    )
    if nvfp4:
        codes, scales = nvfp4_shard_views(index_kv_shard)
        kernel(idx_q, codes, scales, block_table, score_out, seq_lens, rank)
    else:
        kernel(idx_q, index_kv_shard, block_table, score_out, seq_lens, rank)


@triton.jit
def _index_k_load_tagged_word(ptr):
    return tl.inline_asm_elementwise(
        "ld.relaxed.sys.u64 $0, [$1];",
        "=l,l",
        [ptr],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _index_k_merge_candidates_kernel(
    recv_ptr,
    valid_ptr,
    out_ptr,
    out_stride_t,
    out_stride_h,
    valid_stride,
    err_ptr,
    recv_stride_src,
    recv_stride_h,
    recv_stride_t,
    expected_generations,
    tokens,
    TAGGED: tl.constexpr,
    PREFILL: tl.constexpr,
    ROWS: tl.constexpr,
    W: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
    MAX_TILES: tl.constexpr,
    SPIN: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if TAGGED and W == 2:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        end = tl.cdiv(tokens, ROWS)
        stride = tl.num_programs(0)
    else:
        end = tl.program_id(0) + 1
        stride = 1
    for tile in range(tl.program_id(0), end, stride):
        if ROWS > 1:
            t = tile * ROWS + tl.arange(0, ROWS)[:, None]
            k = tl.arange(0, W * 16)[None, :]
        else:
            t = tile
            k = tl.arange(0, W * 16)
        h = tl.program_id(1)
        row = recv_ptr + h * recv_stride_h + t * recv_stride_t
        source = row + k // 16 * recv_stride_src + k % 16
        if TAGGED and ROWS > 1:
            source = source - tl.maximum(t - (tokens - 1), 0) * recv_stride_t
        if PREFILL:
            ids = tl.load(source + 16, mask=t < tokens, other=-1)
            scores = tl.load(source, mask=t < tokens, other=0).to(
                tl.float32, bitcast=True
            )
            pending = 0
        elif TAGGED:
            counter = expected_generations + h * (recv_stride_h // recv_stride_t) + t
            generation = tl.load(counter, mask=t < tokens, other=0).to(tl.uint32) + 1
            generation = tl.where(generation == 0, 1, generation).to(tl.uint32)
            pending, spins = 1, 0
            score_words = tl.full(
                (ROWS, W * 16) if ROWS > 1 else (W * 16,), 0, tl.uint64
            )
            id_words = tl.full((ROWS, W * 16) if ROWS > 1 else (W * 16,), 0, tl.uint64)
            while (pending > 0) & (spins < SPIN):
                score_words = _index_k_load_tagged_word(source)
                id_words = _index_k_load_tagged_word(source + 16)
                matches = ((score_words >> 32) == generation) & (
                    (id_words >> 32) == generation
                )
                if ROWS > 1:
                    pending = tl.sum(
                        tl.sum((~matches & (t < tokens)).to(tl.int32), 1), 0
                    )
                else:
                    pending = tl.sum((~matches).to(tl.int32), 0)
                spins += 1
            ids = id_words.to(tl.int32)
            scores = score_words.to(tl.uint32).to(tl.float32, bitcast=True)
        else:
            ready = row + tl.arange(0, W) * recv_stride_src + 32
            seen = tl.atomic_add(ready, 0, sem="acquire", scope="sys")
            pending = tl.sum((seen != 1).to(tl.int32), 0)
            spins = 0
            while (pending > 0) & (spins < SPIN):
                seen = tl.atomic_add(ready, 0, sem="acquire", scope="sys")
                pending = tl.sum((seen != 1).to(tl.int32), 0)
                spins += 1
            ids = tl.load(source + 16)
            scores = tl.load(source).to(tl.float32, bitcast=True)
        if pending > 0:
            tl.atomic_add(err_ptr, 1)
            tl.inline_asm_elementwise(
                "trap; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
            )
        if ROWS > 1:
            nvalid = tl.minimum(
                tl.load(valid_ptr + t * valid_stride, mask=t < tokens, other=0),
                MAX_TILES,
            )
        else:
            nvalid = tl.minimum(tl.load(valid_ptr + t * valid_stride), MAX_TILES)
        valid = (ids >= 0) & (ids < nvalid)
        scores = tl.where(scores == scores, scores, -float("inf"))
        if W != 2:
            equal = (ids[:, None] == ids[None, :]) & valid[:, None] & valid[None, :]
            scores = tl.max(tl.where(equal, scores[None, :], -float("inf")), 1)
            first = tl.min(tl.where(equal, k[None, :], W * 16), 1)
            unique = valid & (k == first)
        forced = (ids < INIT) | (ids >= tl.maximum(0, nvalid - LOCAL))
        scores = tl.where(forced, float("inf"), scores)
        # Float ordering followed by inverse ID gives a stable cutoff tie break.
        bits = tl.where(scores == 0, 0.0, scores).to(tl.uint32, bitcast=True)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        if W == 2:
            # One candidate per lane; retain the maximum on the first matching lane.
            ordered, leader = tl.inline_asm_elementwise(
                """{
                    .reg .b32 peers, reversed, first, lane;
                    .reg .pred p;
                    match.any.sync.b32 peers, $2, 0xffffffff;
                    redux.sync.max.u32 $0, $3, peers;
                    brev.b32 reversed, peers;
                    clz.b32 first, reversed;
                    mov.u32 lane, %laneid;
                    setp.eq.u32 p, lane, first;
                    selp.u32 $1, 1, 0, p;
                }""",
                constraints="=r,=r,r,r",
                args=[ids, ordered],
                dtype=(tl.uint32, tl.int32),
                is_pure=True,
                pack=1,
            )
            unique = valid & (leader != 0)
        keys = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - ids.to(tl.uint32)).to(
            tl.uint64
        )
        keys = tl.where(unique, keys, 0)
        keys = tl.sort(keys, descending=True)
        if ROWS > 1:
            positions = tl.broadcast_to(tl.arange(0, 16)[None, :], (ROWS, 16))
            selected_keys = tl.gather(keys, positions, 1)
        else:
            positions = tl.arange(0, 16)
            selected_keys = tl.gather(keys, positions, 0)
        selected = (0xFFFFFFFF - selected_keys.to(tl.uint32)).to(tl.int32)
        selected = tl.where(selected_keys != 0, selected, 0x7FFFFFFF)
        selected = tl.sort(selected, descending=False)
        output = tl.where(selected == 0x7FFFFFFF, -1, selected)
        if PREFILL:
            # Keep graph-padding rows nonempty for downstream CSR construction.
            output = tl.where((nvalid == 0) & (positions == 0), 0, output)
        tl.store(
            out_ptr + t * out_stride_t + h * out_stride_h + positions,
            output,
            mask=t < tokens if PREFILL or ROWS > 1 else True,
        )
        if not PREFILL:
            tl.debug_barrier()
            if TAGGED:
                tl.store(counter, generation, mask=t < tokens)
            else:
                tl.atomic_xchg(ready, 0, sem="release", scope="sys")
    if launch_pdl:
        # Do not let the next layer reuse scratch before this publisher exits.
        tl.extra.cuda.gdc_wait()


def minimax_m3_index_select_publish_candidates(
    scores: torch.Tensor,
    num_valid_pages: torch.Tensor,
    recv_bases: torch.Tensor,
    slot: int,
    rank: int,
    t_max: int,
    errors: torch.Tensor,
    *,
    world_size: int,
    heads_per_rank: int,
    init_blocks: int,
    local_blocks: int,
    spin_limit: int = 1 << 26,
    use_pdl: bool = False,
    generations: torch.Tensor | None = None,
    index_scale: int = 1,
    index_offset: int = 0,
) -> None:
    """Select local Top16 directly into each head owner's receive slot."""
    from vllm.third_party.fmha_sm100.jit import get_sparse_topk_module

    get_sparse_topk_module().sparse_topk_select_publish(
        scores,
        num_valid_pages,
        recv_bases,
        errors,
        slot,
        rank,
        world_size,
        heads_per_rank,
        t_max,
        init_blocks,
        local_blocks,
        spin_limit,
        use_pdl,
        generations,
        index_scale,
        index_offset,
        torch.cuda.current_stream().cuda_stream,
    )


def minimax_m3_index_merge_candidates(
    recv_slot: torch.Tensor,
    num_valid_pages: torch.Tensor,
    output: torch.Tensor,
    errors: torch.Tensor,
    *,
    init_blocks: int,
    local_blocks: int,
    spin_limit: int = 1 << 26,
    use_pdl: bool = False,
    expected_generations: torch.Tensor | None = None,
    prefill: bool = False,
) -> None:
    """Merge duplicate blocks by max, force init/local blocks, and sort Top16 IDs."""
    world, heads, capacity, row_size = recv_slot.shape
    tokens, out_heads, topk = output.shape
    assert world in SUPPORTED_TP and out_heads == heads
    tagged = expected_generations is not None
    assert not prefill or (not tagged and not use_pdl)
    assert topk == TOPK and tokens <= capacity
    assert row_size == (2 * TOPK if tagged else CANDIDATE_ROW_SIZE)
    assert recv_slot.dtype == (torch.int64 if tagged else torch.int32)
    if expected_generations is not None:
        assert expected_generations.shape == (heads, capacity)
        assert expected_generations.dtype == torch.int32
        assert expected_generations.is_contiguous()
    assert recv_slot.stride(-1) == output.stride(-1) == 1
    assert num_valid_pages.shape == (tokens,)
    rows = 8 if world == 2 and (prefill or tagged) else 1
    grid = triton.cdiv(tokens, rows)
    if tagged and world == 2:
        # At most one 8-warp CTA per SM; larger batches loop over row tiles.
        grid = min(
            grid, max(1, num_compute_units(recv_slot.device.index or 0) // heads)
        )
    _index_k_merge_candidates_kernel[(grid, heads)](
        recv_slot,
        num_valid_pages,
        output,
        output.stride(0),
        output.stride(1),
        num_valid_pages.stride(0),
        errors,
        recv_slot.stride(0),
        recv_slot.stride(1),
        recv_slot.stride(2),
        expected_generations,
        tokens,
        TAGGED=tagged,
        PREFILL=prefill,
        ROWS=rows,
        W=world,
        INIT=init_blocks,
        LOCAL=local_blocks,
        MAX_TILES=MAX_K_TILES,
        SPIN=spin_limit,
        launch_pdl=use_pdl,
        # Tagged words must have one polling thread per logical candidate.
        num_warps=rows if world == 2 else 2 if tagged else 4,
    )
