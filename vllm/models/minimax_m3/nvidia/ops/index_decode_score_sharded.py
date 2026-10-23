# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode index scores over TP-sharded FP8 keys and exchange top-k candidates."""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float16, Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, warp
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import EVICT_FIRST, mma_sync, simple_tma_copy
from vllm.models.minimax_m3.nvidia.ops.index_decode_score import (
    _fp8_to_f16_mma_fragments,
)
from vllm.triton_utils import tl, triton

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
        use_pdl: bool = False,
        prefetch_history: bool = False,
    ):
        self.dtype = Float8E4M3FN
        self.num_heads = num_heads
        self.max_decode_query_len = max_decode_query_len
        self.split_k = split_k
        self.head_dim = head_dim
        self.world_size = world_size
        self.use_pdl = use_pdl
        self.prefetch_history = prefetch_history

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [bs * runtime_decode_query_len, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, BLOCK // world_size, head_dim]
        block_table: cute.Tensor,  # [bs, max_pages]
        scores: cute.Tensor,  # [total_tokens, num_heads, MAX_K_TILES]
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
        ROWS = BLOCK // self.world_size
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
        ROWS = BLOCK // TP
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
        num_blocks = cute.ceil_div(seqlen, BLOCK)
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
                if (split_id + 1) * PAGES * BLOCK <= seqlen - decode_query_len:
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

                    tile = super_id * PAGES + warp_id
                    # Page row i holds the block's token i * TP + rank.
                    k_base = tile * BLOCK + rank

                    # causal mask
                    for q in cutlass.range_constexpr(Q_TILES):
                        rC_q = rC[None, None, q]
                        for i in cutlass.range_constexpr(ROWS // 8):
                            for j in cutlass.range_constexpr(2):
                                col = q * 8 + (lane_id % 4) * 2 + j
                                q_local_pos = col % MAX_DQL
                                q_pos = q_start + q_local_pos
                                k_pos = k_base + (i * 8 + lane_id // 4) * TP
                                rC_q[i * 2 + j] = (
                                    rC_q[i * 2 + j] if q_pos >= k_pos else float("-inf")
                                )

                    for q in cutlass.range_constexpr(Q_TILES):
                        rC_q = rC[None, None, q]
                        # thread-reduction along the page's local rows
                        rScore = cute.make_rmem_tensor(2, Float32)
                        rScore.fill(float("-inf"))
                        for i in cutlass.range_constexpr(ROWS // 8):
                            rScore[0] = cute.arch.fmax(rScore[0], rC_q[i * 2 + 0])
                            rScore[1] = cute.arch.fmax(rScore[1], rC_q[i * 2 + 1])

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

                        # lanes 0..3 hold columns q * 8 + 2 * lane + {0, 1}
                        if lane_id * 2 < MMA_N:
                            epi_buffer[q * MMA_N + lane_id * 2 + 0, warp_id] = rScore[0]
                            epi_buffer[q * MMA_N + lane_id * 2 + 1, warp_id] = rScore[1]
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

                    # Padded pages are -inf.
                    if warp_id == 0 and lane_id < BLOCK_Q:
                        head_id = lane_id // MAX_DQL
                        q_local_pos = lane_id - head_id * MAX_DQL
                        if q_local_pos < decode_query_len:
                            t = batch_id * decode_query_len + q_local_pos
                            for p in cutlass.range_constexpr(PAGES):
                                scores[t, head_id, super_id * PAGES + p] = epi_buffer[
                                    lane_id, p
                                ]

                    tma_stage = (tma_stage + 1) % self.num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

    @cache
    @staticmethod
    def compile(
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
        world_size: int = 4,
        use_pdl: bool = False,
        prefetch_history: bool = False,
    ):
        bs = cute.sym_int()
        total_tokens = cute.sym_int()
        dtype = Float8E4M3FN

        q = make_fake_tensor(
            dtype, (total_tokens, num_heads, head_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            dtype, (cute.sym_int(), BLOCK // world_size, head_dim), divisibility=16
        )
        block_table = make_fake_tensor(Int32, (bs, cute.sym_int()), divisibility=1)
        scores = make_fake_tensor(
            Float32, (total_tokens, num_heads, MAX_K_TILES), divisibility=4
        )
        seq_lens = make_fake_tensor(Int32, (bs,), divisibility=1)
        kernel = IndexDecodeScoreShardedKernel(
            num_heads,
            max_decode_query_len,
            split_k,
            head_dim,
            world_size,
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
    use_pdl: bool = False,
    prefetch_history: bool = False,
) -> None:
    """Write local maxima for all heads over interleaved token shards.

    History prefetch requires metadata fenced by a non-PDL cache writer.
    """
    if idx_q.dtype != torch.float8_e4m3fn or index_kv_shard.dtype != idx_q.dtype:
        raise TypeError("sharded index decode score supports FP8 E4M3 only")
    total_tokens, num_heads, head_dim = idx_q.shape
    batch = block_table.shape[0]
    assert index_kv_shard.shape[1] in (BLOCK // 2, BLOCK // 4)
    world_size = BLOCK // index_kv_shard.shape[1]
    assert total_tokens == batch * decode_query_len
    assert 1 <= decode_query_len <= max_decode_query_len
    assert num_heads * max_decode_query_len <= 32
    assert 0 <= rank < world_size and num_heads % world_size == 0
    assert score_out.shape == (total_tokens, num_heads, MAX_K_TILES)
    assert score_out.dtype == torch.float32 and score_out.is_contiguous()
    kernel = IndexDecodeScoreShardedKernel.compile(
        num_heads,
        max_decode_query_len,
        split_k,
        head_dim,
        world_size,
        use_pdl,
        prefetch_history,
    )
    kernel(idx_q, index_kv_shard, block_table, score_out, seq_lens, rank)


@triton.jit(do_not_specialize=["slot", "rank"])
def _index_k_publish_candidates_kernel(
    scores_ptr,
    indices_ptr,
    recv_bases,
    slot,
    rank,
    t_max,
    err_ptr,
    score_stride_t,
    score_stride_h,
    index_stride_t,
    index_stride_h,
    W: tl.constexpr,
    HPR: tl.constexpr,
    MAX_TILES: tl.constexpr,
    SPIN: tl.constexpr,
    PUBLISH: tl.constexpr = True,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    if PUBLISH:
        peer = tl.load(recv_bases + h // HPR).to(tl.pointer_type(tl.int32))
        row = peer + (((slot * W + rank) * HPR + h % HPR) * t_max + t) * 33
    else:
        row = recv_bases + (h * t_max + t) * 33
    ready = row + 32
    if PUBLISH:
        seen = tl.atomic_add(ready, 0, sem="acquire", scope="sys")
        spins = 0
        while (seen != 0) & (spins < SPIN):
            seen = tl.atomic_add(ready, 0, sem="acquire", scope="sys")
            spins += 1
        if seen != 0:
            tl.atomic_add(err_ptr, 1)
            tl.inline_asm_elementwise(
                "trap; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
            )
    k = tl.arange(0, 16)
    ids = tl.load(indices_ptr + t * index_stride_t + h * index_stride_h + k)
    valid = (ids >= 0) & (ids < MAX_TILES)
    scores = tl.load(
        scores_ptr + t * score_stride_t + h * score_stride_h + ids,
        mask=valid,
        other=-float("inf"),
    )
    tl.store(row + k, scores.to(tl.int32, bitcast=True))
    tl.store(row + 16 + k, tl.where(valid, ids, -1))
    if PUBLISH:
        tl.debug_barrier()
        tl.atomic_xchg(ready, 1, sem="release", scope="sys")
    else:
        tl.store(ready, 1)


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
    if ROWS > 1:
        t = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
        k = tl.arange(0, W * 16)[None, :]
    else:
        t = tl.program_id(0)
        k = tl.arange(0, W * 16)
    h = tl.program_id(1)
    row = recv_ptr + h * recv_stride_h + t * recv_stride_t
    source = row + k // 16 * recv_stride_src + k % 16
    if PREFILL:
        ids = tl.load(source + 16, mask=t < tokens, other=-1)
        scores = tl.load(source, mask=t < tokens, other=0).to(tl.float32, bitcast=True)
        pending = 0
    elif TAGGED:
        counter = expected_generations + h * (recv_stride_h // recv_stride_t) + t
        generation = tl.load(counter).to(tl.uint32) + 1
        generation = tl.where(generation == 0, 1, generation).to(tl.uint32)
        pending, spins = 1, 0
        score_words = tl.full((W * 16,), 0, tl.uint64)
        id_words = tl.full((W * 16,), 0, tl.uint64)
        while (pending > 0) & (spins < SPIN):
            score_words = _index_k_load_tagged_word(source)
            id_words = _index_k_load_tagged_word(source + 16)
            matches = ((score_words >> 32) == generation) & (
                (id_words >> 32) == generation
            )
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
    tl.store(
        out_ptr + t * out_stride_t + h * out_stride_h + positions,
        tl.where(selected == 0x7FFFFFFF, -1, selected),
        mask=t < tokens if PREFILL else True,
    )
    if not PREFILL:
        tl.debug_barrier()
        if TAGGED:
            tl.store(counter, generation)
        else:
            tl.atomic_xchg(ready, 0, sem="release", scope="sys")
    if launch_pdl:
        # Do not let the next layer reuse scratch before this publisher exits.
        tl.extra.cuda.gdc_wait()


def minimax_m3_index_publish_candidates(
    scores: torch.Tensor,
    local_indices: torch.Tensor,
    recv_bases: torch.Tensor,
    slot: int,
    rank: int,
    t_max: int,
    errors: torch.Tensor,
    *,
    world_size: int,
    heads_per_rank: int,
    spin_limit: int = 1 << 26,
) -> None:
    """Publish local Top16 into score/ID/ready rows on each head's owner."""
    tokens, heads, topk = local_indices.shape
    assert topk == TOPK and tokens <= t_max
    assert heads == world_size * heads_per_rank
    assert scores.shape == (tokens, heads, MAX_K_TILES)
    assert scores.stride(-1) == local_indices.stride(-1) == 1
    _index_k_publish_candidates_kernel[(tokens, heads)](
        scores,
        local_indices,
        recv_bases,
        slot,
        rank,
        t_max,
        errors,
        scores.stride(0),
        scores.stride(1),
        local_indices.stride(0),
        local_indices.stride(1),
        W=world_size,
        HPR=heads_per_rank,
        MAX_TILES=MAX_K_TILES,
        SPIN=spin_limit,
        num_warps=4,
    )


def minimax_m3_index_pack_candidates(
    scores: torch.Tensor, candidates: torch.Tensor
) -> None:
    """Fill score/ID/ready rows in a head-major prefill send buffer."""
    heads, tokens, row_size = candidates.shape
    assert row_size == CANDIDATE_ROW_SIZE and candidates.stride(-1) == 1
    assert scores.shape[:2] == (tokens, heads) and scores.stride(-1) == 1
    indices = candidates[..., TOPK : 2 * TOPK]
    _index_k_publish_candidates_kernel[(tokens, heads)](
        scores,
        indices,
        candidates,
        0,
        0,
        candidates.stride(0) // CANDIDATE_ROW_SIZE,
        None,
        scores.stride(0),
        scores.stride(1),
        indices.stride(1),
        indices.stride(0),
        W=1,
        HPR=heads,
        MAX_TILES=scores.shape[-1],
        SPIN=0,
        PUBLISH=False,
        num_warps=4,
    )


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
    rows = 8 if prefill and world == 2 else 1
    _index_k_merge_candidates_kernel[(triton.cdiv(tokens, rows), heads)](
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
        num_warps=rows if world == 2 else 4,
    )
