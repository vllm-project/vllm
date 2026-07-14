# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL MiniMax M3 index decode score kernel.

The kernel computes decode-time index block scores with TMA + ``mma.sync``.
We use ``mma.sync`` instead of tcgen05 because this score GEMM has a very small
N dimension and benefits more from higher CTA occupancy than from a deeper
single-CTA tcgen05 pipeline.

The implementation should be portable to SM90/SM120 in principle, but it is
currently validated only for SM100.
"""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float16, Float32, Int32, Int64, Uint32, cute
from cutlass.cute.nvgpu import cpasync, warp
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import (
    _TORCH_TO_CUTE_DTYPE,
    EVICT_FIRST,
    cvt,
    mma_sync,
    simple_tma_copy,
)


@cute.jit
def _fp8_to_f16_mma_fragments(src: cute.Tensor):
    src_elems = cute.size(src)
    src_u32 = cute.recast_tensor(src, Uint32)
    src_f16 = cute.make_rmem_tensor(src_elems, Float16)
    src_f16_u32 = cute.recast_tensor(src_f16, Uint32)
    # This packed conversion is faster and emits fewer SASS instructions than
    # src.load().to(Float16).
    for i in cutlass.range_constexpr(src_elems // 4):
        converted = cvt.fp8x4_to_fp16x4(src_u32[i])
        src_f16_u32[i * 2] = converted[0]
        src_f16_u32[i * 2 + 1] = converted[1]
    lower = cute.make_rmem_tensor(src_elems // 2, Float16)
    upper = cute.make_rmem_tensor(src_elems // 2, Float16)

    # FP8 ldmatrix gives four consecutive values along K. Split each group
    # into the lower two and upper two values for two FP16 MMA k-fragments.
    for i in cutlass.range_constexpr(src_elems // 2):
        lower[i] = src_f16[(i // 2) * 4 + i % 2]
        upper[i] = src_f16[(i // 2) * 4 + 2 + i % 2]
    return lower, upper


class IndexDecodeScoreKernel:
    BLOCK_K = 128
    BAR_MMA = 1
    num_stages = 2

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
    ):
        self.dtype = dtype
        self.num_heads = num_heads
        self.max_decode_query_len = max_decode_query_len
        self.split_k = split_k
        self.head_dim = head_dim

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [bs * runtime_decode_query_len, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, page_size, head_dim]
        block_table: cute.Tensor,  # [bs, max_pages]
        score: cute.Tensor,  # [num_heads, bs * runtime_decode_query_len, max_pages]
        seq_lens: cute.Tensor,  # [bs]
        stream: CUstream,
    ):
        dtype = self.dtype
        num_heads = self.num_heads
        head_dim = self.head_dim
        BLOCK_K = self.BLOCK_K
        num_stages = self.num_stages
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = num_heads * MAX_DQL
        assert BLOCK_Q <= 32

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

        sK_layout = cute.make_layout(
            (1, BLOCK_K, (elems, head_dim // elems), num_stages),
            stride=(0, elems, (1, BLOCK_K * elems), BLOCK_K * head_dim),
        )
        sK_layout = cute.make_composed_layout(swizzle_128B, 0, sK_layout)
        K_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gK_cache, (None, None, elems)),
            sK_layout,
            cta_tiler=(1, BLOCK_K, head_dim),
        )

        self.kernel(
            Q_tma,
            K_tma,
            block_table,
            score,
            seq_lens,
            decode_query_len,
        ).launch(grid=grid, block=block, stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(
        self,
        Q_tma: cpasync.TmaInfo,
        K_tma: cpasync.TmaInfo,
        block_table: cute.Tensor,
        score: cute.Tensor,
        seq_lens: cute.Tensor,
        decode_query_len,
    ):
        tid, _, _ = cute.arch.thread_idx()
        batch_id, split_id, _ = cute.arch.block_idx()
        _, split_k, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        NUM_HEADS = self.num_heads
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = NUM_HEADS * MAX_DQL
        BLOCK_K = self.BLOCK_K
        head_dim = self.head_dim
        dtype = self.dtype
        MMA_N = 8
        num_stages = self.num_stages
        Q_TILES = cute.ceil_div(BLOCK_Q, MMA_N)
        EPI_Q = Q_TILES * MMA_N

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            dtype,
            K_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=K_tma.smem_layout.inner,
        )[0, None, None, None]
        # alias sQ with the 1st stage of sK
        sQ_tma = cute.make_tensor(
            sK[None, None, 0].iterator, layout=Q_tma.smem_layout.outer
        )
        # TMA sees Q as (query, head, dim), while ldmatrix consumes a
        # flattened Q column mode. The target profile keeps the rank-2 view even
        # for degenerate shapes like DQL1.
        q_tma_elems = 128 * 8 // dtype.width
        sQ = cute.coalesce(
            cute.group_modes(sQ_tma, 0, 2),
            target_profile=(BLOCK_Q, (q_tma_elems, head_dim // q_tma_elems)),
        )
        epi_buffer = smem.allocate_tensor(Float32, cute.make_layout((EPI_Q, 4)))

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        seqlen = seq_lens[batch_id]
        num_blocks = cute.ceil_div(seqlen, BLOCK_K)

        if split_id < num_blocks:
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

            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

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
                # TMA bounds-checks rows when runtime decode_query_len is smaller
                # than MAX_DQL; padded Q columns are masked before global stores.
                simple_tma_copy(Q_tma.atom, gQ_tile, sQ_tma, tma_full_mbar)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    tma_parity ^= 1

                for block_id in range(split_id, num_blocks, split_k):
                    page_id = block_table[batch_id, block_id]
                    gK_tile = K_tma.tma_tensor[page_id, None, None]
                    k_mbar = tma_full_mbar + tma_stage

                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
                    with cute.arch.elect_one():
                        K_size = BLOCK_K * head_dim * (dtype.width // 8)
                        cute.arch.mbarrier_arrive_and_expect_tx(k_mbar, K_size)
                    simple_tma_copy(
                        K_tma.atom,
                        gK_tile,
                        sK[None, None, tma_stage],
                        k_mbar,
                        cache_policy=EVICT_FIRST,
                    )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

            else:
                # MMA warps
                # each warp handles K[32, head_dim] @ Q[BLOCK_Q, head_dim].T
                sK_warp = cute.local_tile(
                    sK, (32, head_dim, num_stages), (warp_id, 0, 0)
                )
                q_start = seqlen - decode_query_len

                elems = 128 // dtype.width  # 16B
                MMA_K = 32 * 8 // dtype.width  # 32B

                # Pre-compute ldmatrix address.
                # sK loads a [16 x 16B] tile:
                #   ((16, (16B, 2), 1), (32 / 16, head_dim / 32B, num_stages))
                # sQ loads an [8 x 32B] tile:
                #   ((8, (16B, 4)), (BLOCK_Q / MMA_N, head_dim / 64B))
                sK_ldsm = cute.zipped_divide(
                    sK_warp, (16, cute.make_layout((elems, 2)), 1)
                )
                sQ_ldsm = cute.zipped_divide(sQ, (MMA_N, cute.make_layout((elems, 4))))

                # sK: (16B, (32 / 16, head_dim / 32B, num_stages))
                # sQ: (16B, (BLOCK_Q / MMA_N, head_dim / 64B))
                sK_ldsm = sK_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]
                sQ_ldsm = sQ_ldsm[(lane_id % MMA_N, (None, lane_id // 8)), None]

                ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
                ldsm_atom = cute.make_copy_atom(ldsm_op, dtype)

                rQ = cute.make_rmem_tensor(
                    ((elems // 2, 2), head_dim // (MMA_K * 2), Q_TILES), dtype
                )
                rK = cute.make_rmem_tensor((elems, 2, head_dim // MMA_K), dtype)
                rC = cute.make_rmem_tensor((4, 2, Q_TILES), Float32)

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

                # sm100 doesn't have native mma.sync.f8. ptxas lowers mma.sync.f8
                # to F2FP.F16.E4M3 + HMMA; doing the conversion explicitly gives
                # better codegen while keeping the two FP16 k-fragments visible.
                if cutlass.const_expr(dtype is Float8E4M3FN):
                    rQ_f16 = cute.make_rmem_tensor(
                        (4, head_dim // MMA_K, Q_TILES, 2), Float16
                    )
                    q_lower, q_upper = _fp8_to_f16_mma_fragments(rQ)
                    rQ_f16[None, None, None, 0].store(q_lower.load())
                    rQ_f16[None, None, None, 1].store(q_upper.load())

                for block_id in range(split_id, num_blocks, split_k):
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
                        for m in cutlass.range_constexpr(2):
                            if cutlass.const_expr(dtype is Float8E4M3FN):
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
                            else:
                                for n in cutlass.range_constexpr(Q_TILES):
                                    rC[None, m, n] = mma_sync(
                                        rK[None, m, k],
                                        rQ[(None, k % 2), k // 2, n],
                                        rC[None, m, n],
                                    )

                    cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                    k_start = block_id * BLOCK_K + warp_id * 32

                    # causal mask
                    for q in cutlass.range_constexpr(Q_TILES):
                        for i in cutlass.range_constexpr(4):
                            for j in cutlass.range_constexpr(2):
                                col = q * 8 + (lane_id % 4) * 2 + j
                                q_local_pos = col % MAX_DQL
                                q_pos = q_start + q_local_pos
                                k_pos = k_start + i * 8 + lane_id // 4
                                rC[q * 8 + i * 2 + j] = (
                                    rC[q * 8 + i * 2 + j]
                                    if q_pos >= k_pos
                                    else float("-inf")
                                )

                    for q in cutlass.range_constexpr(Q_TILES):
                        # thread-reduction along BLOCK_K dim
                        rScore = cute.make_rmem_tensor(2, Float32)
                        rScore.fill(float("-inf"))
                        for i in cutlass.range_constexpr(4):
                            rScore[0] = cute.arch.fmax(rScore[0], rC[i * 2 + 0 + q * 8])
                            rScore[1] = cute.arch.fmax(rScore[1], rC[i * 2 + 1 + q * 8])

                        # warp-reduction among lanes 0,4,8,12,...
                        for i in cutlass.range_constexpr(3):
                            offset = 4 << i
                            other0 = cute.arch.shuffle_sync_bfly(
                                rScore[0], offset=offset, mask=-1, mask_and_clamp=31
                            )
                            other1 = cute.arch.shuffle_sync_bfly(
                                rScore[1], offset=offset, mask=-1, mask_and_clamp=31
                            )
                            rScore[0] = cute.arch.fmax(rScore[0], other0)
                            rScore[1] = cute.arch.fmax(rScore[1], other1)

                        # store to smem for 4-warp reduction
                        if lane_id * 2 < MMA_N:
                            epi_buffer[q * MMA_N + lane_id * 2 + 0, warp_id] = rScore[0]
                            epi_buffer[q * MMA_N + lane_id * 2 + 1, warp_id] = rScore[1]
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

                    head_id = lane_id // MAX_DQL
                    q_local_pos = lane_id - head_id * MAX_DQL
                    valid_q = head_id < NUM_HEADS and q_local_pos < decode_query_len
                    if lane_id < BLOCK_Q and valid_q:
                        final_score = epi_buffer[lane_id, 0]
                        for i in cutlass.range_constexpr(1, 4):
                            final_score = cute.arch.fmax(
                                final_score, epi_buffer[lane_id, i]
                            )

                        t = batch_id * decode_query_len + q_local_pos
                        score[head_id, t, block_id] = final_score

                    tma_stage = (tma_stage + 1) % self.num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

    @cache
    @staticmethod
    def compile(
        dtype: type[cutlass.Numeric],
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
    ):
        bs = cute.sym_int()
        total_tokens = cute.sym_int()
        BLOCK_K = IndexDecodeScoreKernel.BLOCK_K

        q = make_fake_tensor(
            dtype, (total_tokens, num_heads, head_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            dtype, (cute.sym_int(), BLOCK_K, head_dim), divisibility=16
        )
        block_table = make_fake_tensor(Int32, (bs, cute.sym_int()), divisibility=1)
        score = make_fake_tensor(
            Float32, (num_heads, total_tokens, cute.sym_int()), divisibility=4
        )
        seq_lens = make_fake_tensor(Int32, (bs,), divisibility=1)
        kernel = IndexDecodeScoreKernel(
            dtype,
            num_heads,
            max_decode_query_len,
            split_k,
            head_dim,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            q,
            k_cache,
            block_table,
            score,
            seq_lens,
            stream,
            options="--enable-tvm-ffi",
        )


def minimax_m3_index_decode_score_cutedsl(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    score_out: torch.Tensor,
) -> torch.Tensor:
    if idx_q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("CuteDSL index decode score supports BF16 and FP8 E4M3 only")
    total_tokens, num_heads, head_dim = idx_q.shape
    batch = block_table.shape[0]
    assert index_kv_cache.shape[1] == IndexDecodeScoreKernel.BLOCK_K
    assert total_tokens == batch * decode_query_len
    assert 1 <= decode_query_len <= max_decode_query_len
    assert num_heads * max_decode_query_len <= 32
    dtype = _TORCH_TO_CUTE_DTYPE[idx_q.dtype]
    del max_seq_len, init_blocks, local_blocks, num_kv_heads
    score = score_out
    split_k = 256
    kernel = IndexDecodeScoreKernel.compile(
        dtype,
        num_heads,
        max_decode_query_len,
        split_k,
        head_dim,
    )
    kernel(idx_q, index_kv_cache, block_table, score, seq_lens)
    return score
