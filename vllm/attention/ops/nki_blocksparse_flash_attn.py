# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np
import numpy.typing as npt
from neuronxcc.nki.language import par_dim

from .nki_flash_attn import (_flash_attention_core, ceil_div, is_power_of_2,
                             load_block_tables, load_kv_tile_from_cache,
                             load_v_tile,
                             transform_block_tables_for_indirect_load)


@dataclass(frozen=True)
class BlockSparsePlan:
    tile_q_indices: npt.NDArray[np.int32]
    tile_block_table_offsets: npt.NDArray[np.int32]
    tile_q_seq_ids: npt.NDArray[np.int32]
    tile_kv_seq_ids: npt.NDArray[np.int32]
    block_size: int

    def __post_init__(self):
        for arg in [
                self.tile_q_indices,
                self.tile_block_table_offsets,
                self.tile_q_seq_ids,
                self.tile_kv_seq_ids,
        ]:
            assert isinstance(arg,
                              np.ndarray) and arg.dtype == np.int32, type(arg)
            assert arg.shape[0] == self.num_tiles, (arg.shape, self.num_tiles)

    @property
    def num_tiles(self):
        return len(self.tile_q_indices)

    def build_tile_masks(self):
        tile_kv_seq_ids = self.tile_kv_seq_ids
        B_P_SIZE = 128
        num_tiles, tile_size_kv = tile_kv_seq_ids.shape
        block_size = self.block_size
        num_tiled_blocks = max(B_P_SIZE, tile_size_kv // block_size)
        assert (
            tile_size_kv % B_P_SIZE == 0 and tile_size_kv % block_size == 0
        ), f"{tile_size_kv=} is not multiple of {B_P_SIZE=} and {block_size=}"
        tiled_block_size = tile_size_kv // num_tiled_blocks
        if tiled_block_size > 1:
            # reorder mask is needed somewhere as long as tiled_block_size > 1
            tile_kv_seq_ids = tile_kv_seq_ids.reshape((
                num_tiles,
                num_tiled_blocks // B_P_SIZE,
                B_P_SIZE,
                tiled_block_size,
            ))
            tile_kv_seq_ids = tile_kv_seq_ids.transpose(0, 1, 3, 2).reshape(
                (num_tiles, tile_size_kv))
        return np.expand_dims(self.tile_q_seq_ids,
                              2) == np.expand_dims(tile_kv_seq_ids, 1)

    def build_tile_block_tables(self, block_tables):
        tile_size_kv = self.tile_kv_seq_ids.shape[1]
        num_blocks_per_tile = tile_size_kv // self.block_size
        assert (tile_size_kv % self.block_size == 0
                ), f"{tile_size_kv=} is not multiple of {self.block_size=}"
        block_tables = block_tables.squeeze()
        in_tile_offset = np.arange(num_blocks_per_tile)
        indices = self.tile_block_table_offsets.reshape(
            -1, 1) + in_tile_offset.reshape(1, -1)
        return block_tables[indices]


def _check_np_int_array(*arrays):
    for a in arrays:
        if not isinstance(a, np.ndarray) or a.dtype not in (np.int32,
                                                            np.int64):
            return False
    return True


class FlashAttentionPlanner:

    def __init__(self, prompt_lens, context_lens, tile_size_q, tile_size_kv,
                 block_size):
        assert _check_np_int_array(prompt_lens, context_lens)
        assert len(prompt_lens) == len(
            context_lens
        ), "prompt_lens and context_lens must have the same length"
        self.num_seq = len(prompt_lens)
        assert self.num_seq > 0, "prompt_lens and context_lens can't be empty"
        self.prompt_lens = prompt_lens.astype(np.int32)
        self.context_lens = context_lens.astype(np.int32)
        self.tile_size_q = tile_size_q
        self.tile_size_kv = tile_size_kv
        self.block_size = block_size

    def plan(self):
        """
        Generate schedule for flash attention
        """
        num_context_blocks = ceil_div(self.context_lens, self.block_size)
        padded_context_lens = num_context_blocks * self.block_size

        def _get_seq_start_end(seqlens, padded_seqlens=None):
            if padded_seqlens is None:
                padded_seqlens = seqlens
            cu_seqlen = np.cumsum(padded_seqlens)
            seqlens_starts = np.concatenate(([0], cu_seqlen[:-1]))
            seqlens_ends = seqlens_starts + seqlens
            return seqlens_starts, seqlens_ends, cu_seqlen[-1]

        prompt_starts, prompt_ends, max_seqlen_q = _get_seq_start_end(
            self.prompt_lens)
        context_starts, context_ends, max_seqlen_kv = _get_seq_start_end(
            self.context_lens, padded_seqlens=padded_context_lens)

        # q dimension seq id
        tile_q_starts = np.arange(0, max_seqlen_q, self.tile_size_q)
        tile_q_ends = tile_q_starts + self.tile_size_q
        tile_q_seq_starts = np.searchsorted(prompt_ends,
                                            tile_q_starts,
                                            side="right")
        tile_q_seq_ends = np.searchsorted(prompt_starts,
                                          tile_q_ends,
                                          side="left")

        # kv dimension seq id
        tile_kv_starts = np.arange(0, max_seqlen_kv, self.tile_size_kv)
        tile_kv_ends = tile_kv_starts + self.tile_size_kv
        tile_kv_seq_starts = np.searchsorted(context_ends,
                                             tile_kv_starts,
                                             side="right")
        tile_kv_seq_ends = np.searchsorted(context_starts,
                                           tile_kv_ends,
                                           side="left")

        # tile_needed = max(q_id, kv_id) < min(q_id, kv_id)
        tile_seq_starts = np.maximum(tile_q_seq_starts.reshape(-1, 1),
                                     tile_kv_seq_starts.reshape(1, -1))
        tile_seq_ends = np.minimum(tile_q_seq_ends.reshape(-1, 1),
                                   tile_kv_seq_ends.reshape(1, -1))
        tile_needed = tile_seq_starts < tile_seq_ends
        tile_q_indices, tile_kv_indices = np.nonzero(tile_needed)

        num_q_tiles = len(tile_q_starts)
        num_kv_tiles = len(tile_kv_starts)
        q_seq_ids = np.repeat(
            np.arange(
                self.num_seq + 1,
                dtype=np.int32,
            ),  # use num_seq as padding value
            np.concatenate((
                self.prompt_lens,
                [num_q_tiles * self.tile_size_q - max_seqlen_q],
            )),
        ).reshape((num_q_tiles, self.tile_size_q))
        kv_seq_ids = np.repeat(
            np.stack((
                np.arange(self.num_seq, dtype=np.int32),
                np.full((self.num_seq, ), self.num_seq + 1,
                        dtype=np.int32),  # use num_seq + 1 as padding
            )).flatten("F"),
            np.stack((
                self.context_lens,
                padded_context_lens - self.context_lens,
            )).flatten("F"),
        )
        kv_seq_ids = np.concatenate((
            kv_seq_ids,
            np.full(
                (num_kv_tiles * self.tile_size_kv - max_seqlen_kv, ),
                self.num_seq + 1,
                dtype=np.int32,
            ),
        )).reshape((num_kv_tiles, self.tile_size_kv))

        tile_q_indices = tile_q_indices.astype(np.int32)
        tile_kv_offsets = tile_kv_indices.astype(np.int32) * self.tile_size_kv
        tile_bt_offsets = tile_kv_offsets // self.block_size
        tile_q_seq_ids = q_seq_ids[tile_q_indices]
        tile_kv_seq_ids = kv_seq_ids[tile_kv_indices]
        return BlockSparsePlan(
            tile_q_indices,
            tile_bt_offsets,
            tile_q_seq_ids,
            tile_kv_seq_ids,
            self.block_size,
        )


@nki.jit
def flash_paged_attention_with_schedule(
    query,
    key,
    value,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    softmax_scale=None,
    mixed_precision=True,
):
    """
    Flash PagedAttention Forward Kernel.
      - PagedAttention Paper: https://arxiv.org/abs/2309.06180
      - Chunked Prefill Paper: https://arxiv.org/abs/2403.02310

    IO tensor layouts:
      - query: shape (1, n_heads, d, seq_q)
      - key:   shape (1, n_kv_heads, d, seq_k)
      - value: shape (1, n_kv_heads, seq_v, d)
      - key_cache: (max_num_blocks, block_size, n_kv_heads, d)
      - value_cache: (max_num_blocks, block_size, n_kv_heads, d)
      - block_tables: (num_large_tile, num_block_per_large_tile)
      - tile_q_indices: (num_large_tiles,)
      - tile_bt_offsets: (num_large_tiles,)
      - tile_masks: (num_large_tiles, large_tile_size_q, large_tile_size_k)
      - active_mask: (seq_q, seq_q)

      - This kernel requires seq_k == seq_v
      - We use continuous batching by default, so the batch dimension is always
        1, and different requests are concatenated along sequence dimension.
      - We use paged cache blocks (key_cache, value_cache) to store KV cache.

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype except for
        block_tables (int32) and mask (int32)
      - If mixed_percision is True, then all Tensor Engine operation will be
        performed in bfloat16 and accumulation will be performed in float32.
        Otherwise the intermediates will be in the same type as the inputs.

    Compile-time Constants:
      - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32 precision, default
        is set to `true`, if false, we use same precision as input types

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of
      nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    B_F_SIZE = 512
    B_P_SIZE = 128

    NUM_LARGE_TILE, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape
    assert (NUM_LARGE_TILE
            > 0), f"At least 1 tile is needed, got {NUM_LARGE_TILE=}"
    b, h, d, seqlen_q = query.shape
    assert seqlen_q % LARGE_Q_TILE_SIZE == 0
    n_large_q_tile = seqlen_q // LARGE_Q_TILE_SIZE
    query = query.reshape((b, h, d, n_large_q_tile, LARGE_Q_TILE_SIZE))
    assert b == 1, f"Batch size must be 1 for Ragged Tensor, got {b}"
    assert (
        d >= 16 and d <= 128 and is_power_of_2(d)
    ), f" we head_dim must be power of 2 in range [16, 128], got head dim {d}"
    B_D_SIZE = d
    num_blocks, k_h, block_size, _ = key_cache.shape
    assert tuple(key_cache.shape) == (
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{key_cache.shape=} mismatch!"
    assert tuple(value_cache.shape) == (
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{value_cache.shape=} mismatch!"
    assert key is None or tuple(key.shape) == (
        1,
        k_h,
        d,
        seqlen_q,
    ), f"key shape {key.shape} mismatch!"
    assert value is None or tuple(value.shape) == (
        1,
        k_h,
        seqlen_q,
        d,
    ), f"value shape {value.shape} mismatch!"

    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    o = nl.ndarray(
        (b, h, seqlen_q, d),
        dtype=query.dtype,
        buffer=nl.shared_hbm,
    )

    assert (
        nl.program_ndim() == 2
    ), f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"

    batch_id = nl.program_id(axis=0)  # equals 0
    head_id = nl.program_id(axis=1)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    assert LARGE_Q_TILE_SIZE % B_P_SIZE == 0
    n_small_in_large_q_tile = LARGE_Q_TILE_SIZE // B_P_SIZE
    assert (LARGE_KV_TILE_SIZE % B_F_SIZE == 0
            ), f"Need {LARGE_KV_TILE_SIZE=} to be divisible by {B_F_SIZE=}"

    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size
    assert is_power_of_2(
        num_blocks_per_large_tile
    ), f"{num_blocks_per_large_tile=} is expected of be power of 2"
    if seqlen_q > B_F_SIZE:
        MAX_REDUCTION_TILE = 2048
        if seqlen_q // 2 > MAX_REDUCTION_TILE:
            assert (
                seqlen_q % MAX_REDUCTION_TILE == 0
            ), f"{seqlen_q=} should be divisible by {MAX_REDUCTION_TILE=}"
        else:
            assert (seqlen_q % B_F_SIZE == 0
                    ), f"{seqlen_q=} should be divisible by {B_F_SIZE=})"

    tile_q_indices_sbuf = nl.load(
        tile_q_indices.reshape((1, NUM_LARGE_TILE)),
        dtype=nl.int32,
    )
    block_tables_sbuf = load_block_tables(
        block_tables_hbm=tile_block_tables,
        num_tiles=NUM_LARGE_TILE,
        num_blocks_per_tile=num_blocks_per_large_tile,
    )
    # We need B_P_SIZE=128 blocks to make DMA efficient
    if num_blocks_per_large_tile < B_P_SIZE:
        # we checked num_blocks_per_tile is a power of 2
        assert B_P_SIZE % num_blocks_per_large_tile == 0
        block_size_tiling_factor = B_P_SIZE // num_blocks_per_large_tile
        assert block_size % block_size_tiling_factor == 0
    else:
        block_size_tiling_factor = 1
    tiled_block_size = block_size // block_size_tiling_factor

    # Indirect DMA load must be placed along Partition dimension
    block_tables_sbuf = transform_block_tables_for_indirect_load(
        block_tables_sbuf,
        block_size_tiling_factor=block_size_tiling_factor,
        num_head=k_h,
        head_id=head_id,
    )

    # flatten KV cache to be 2D for loading into SBUF
    new_cache_shape = (
        num_blocks * k_h * block_size_tiling_factor,
        tiled_block_size * d,
    )
    key_cache = key_cache.reshape(new_cache_shape)
    value_cache = value_cache.reshape(new_cache_shape)

    NEG_INF = -9984.0  # Magic number to replace -inf
    q_h_per_k_h = h // k_h
    # =============== Global Flash Attention accumulators ==================== #
    o_buffer = nl.ndarray(
        (B_P_SIZE, n_large_q_tile, n_small_in_large_q_tile * q_h_per_k_h * d),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    m_buffer = nl.ndarray(
        (B_P_SIZE, n_large_q_tile, n_small_in_large_q_tile * q_h_per_k_h * 1),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    # L buffer stores LSE + M
    # L_0 = LSE_0 + M_0 = log(sum([])) + max([]) = -inf + -inf
    # TODO: since we target inference, we only need to save SumExp instead of
    #       LSE + M
    l_buffer = nl.ndarray(
        (B_P_SIZE, n_large_q_tile, n_small_in_large_q_tile * q_h_per_k_h * 1),
        dtype=acc_type,
        buffer=nl.hbm,
    )

    for large_q_idx in nl.affine_range(n_large_q_tile):
        nl.store(dst=o_buffer[:, large_q_idx], value=0.0)
        nl.store(dst=m_buffer[:, large_q_idx], value=NEG_INF)
        nl.store(dst=l_buffer[:, large_q_idx], value=NEG_INF + NEG_INF)

    for large_tile_idx in nl.sequential_range(0, NUM_LARGE_TILE):
        num_loads = ceil_div(num_blocks_per_large_tile, B_P_SIZE)
        cur_k_tile = nl.ndarray(
            (par_dim(B_D_SIZE), LARGE_KV_TILE_SIZE),
            dtype=kernel_dtype,
        )
        cur_v_tile = nl.ndarray(
            (par_dim(B_P_SIZE), num_loads * tiled_block_size * B_D_SIZE),
            dtype=kernel_dtype,
        )
        load_kv_tile_from_cache(
            cur_k_tile=cur_k_tile,
            cur_v_tile=cur_v_tile,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables_sbuf,
            large_k_tile_idx=large_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            tiled_block_size=tiled_block_size,
            B_P_SIZE=B_P_SIZE,
            B_D_SIZE=B_D_SIZE,
        )

        large_q_idx = tile_q_indices_sbuf[0, large_tile_idx]

        # load aggregation buffer from HBM to SBUF
        m_sbuf_tile = nl.ndarray(
            (par_dim(B_P_SIZE), n_small_in_large_q_tile, q_h_per_k_h, 1),
            dtype=acc_type,
            buffer=nl.sbuf,
        )
        l_sbuf_tile = nl.ndarray(
            (par_dim(B_P_SIZE), n_small_in_large_q_tile, q_h_per_k_h, 1),
            dtype=acc_type,
            buffer=nl.sbuf,
        )
        o_sbuf_tile = nl.ndarray(
            (par_dim(B_P_SIZE), n_small_in_large_q_tile, q_h_per_k_h, d),
            dtype=acc_type,
            buffer=nl.sbuf,
        )
        m_sbuf_tile_flattened = m_sbuf_tile.reshape(
            (B_P_SIZE, n_small_in_large_q_tile * q_h_per_k_h * 1))
        l_sbuf_tile_flattened = l_sbuf_tile.reshape(
            (B_P_SIZE, n_small_in_large_q_tile * q_h_per_k_h * 1))
        o_sbuf_tile_flattened = o_sbuf_tile.reshape(
            (B_P_SIZE, n_small_in_large_q_tile * q_h_per_k_h * d))
        m_sbuf_tile_flattened[...] = nl.load(m_buffer[:, large_q_idx])
        l_sbuf_tile_flattened[...] = nl.load(l_buffer[:, large_q_idx])
        o_sbuf_tile_flattened[...] = nl.load(o_buffer[:, large_q_idx])

        # load query
        q_sbuf_tile = nl.ndarray(
            (q_h_per_k_h, par_dim(B_D_SIZE), LARGE_Q_TILE_SIZE),
            dtype=kernel_dtype,
        )
        for i_q_h in nl.affine_range(q_h_per_k_h):
            q_hbm_tile = nl.load(query[
                batch_id,
                head_id * q_h_per_k_h + i_q_h,
                :,
                large_q_idx,
                :,
            ])
            if kernel_dtype != query.dtype:
                q_hbm_tile = nl.copy(q_hbm_tile, dtype=kernel_dtype)
            q_sbuf_tile[i_q_h, :, :] = q_hbm_tile
        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            cur_mask = nl.load(
                tile_masks[large_tile_idx,
                           nl.ds(small_q_idx * B_P_SIZE, B_P_SIZE), :],
                dtype=tile_masks.dtype,
            )
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_tile = (
                    q_sbuf_tile[i_q_h, :,
                                nl.ds(small_q_idx * B_P_SIZE, B_P_SIZE)] *
                    softmax_scale)

                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    o_buffer=o_sbuf_tile[:, small_q_idx, i_q_h],
                    l_buffer=l_sbuf_tile[:, small_q_idx, i_q_h],
                    m_buffer=m_sbuf_tile[:, small_q_idx, i_q_h],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=cur_mask,
                    use_causal_mask=False,
                    q_tile_idx=None,
                    initialize=False,
                    LARGE_TILE_SZ=LARGE_KV_TILE_SIZE,
                    B_P_SIZE=B_P_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                )

        # write aggregation buffer from SBUF back to HBM
        nl.store(m_buffer[:, large_q_idx], m_sbuf_tile_flattened)
        nl.store(l_buffer[:, large_q_idx], l_sbuf_tile_flattened)
        nl.store(o_buffer[:, large_q_idx], o_sbuf_tile_flattened)

    # ------- Load l, m, o back to SBUF for attention on active tokens ------- #
    o_buffer_sbuf = nl.ndarray(
        (
            n_large_q_tile,
            n_small_in_large_q_tile,
            q_h_per_k_h,
            par_dim(B_P_SIZE),
            d,
        ),
        dtype=acc_type,
    )
    m_buffer_sbuf = nl.ndarray(
        (
            n_large_q_tile,
            n_small_in_large_q_tile,
            q_h_per_k_h,
            par_dim(B_P_SIZE),
            1,
        ),
        dtype=acc_type,
    )
    l_buffer_sbuf = nl.ndarray(
        (
            n_large_q_tile,
            n_small_in_large_q_tile,
            q_h_per_k_h,
            par_dim(B_P_SIZE),
            1,
        ),
        dtype=acc_type,
    )
    for i0 in nl.affine_range(n_large_q_tile):
        for i1 in nl.affine_range(n_small_in_large_q_tile):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                offset = i1 * q_h_per_k_h + i_q_h
                o_buffer_sbuf[i0, i1, i_q_h] = nl.load(
                    o_buffer[:, i0, nl.ds(offset * B_D_SIZE, B_D_SIZE)])
                l_buffer_sbuf[i0, i1,
                              i_q_h] = nl.load(l_buffer[:, i0,
                                                        nl.ds(offset, 1)])
                m_buffer_sbuf[i0, i1,
                              i_q_h] = nl.load(m_buffer[:, i0,
                                                        nl.ds(offset, 1)])

    # compute attention between input query, key and value
    if key is not None and value is not None:
        B_F_SIZE = min(seqlen_q, B_F_SIZE)
        LARGE_Q_TILE_SIZE = seqlen_q
        cur_k_tile = nl.ndarray(
            (par_dim(B_D_SIZE), LARGE_Q_TILE_SIZE),
            dtype=kernel_dtype,
        )
        cur_v_tile = nl.ndarray(
            (par_dim(B_P_SIZE), LARGE_Q_TILE_SIZE // B_P_SIZE * B_D_SIZE),
            dtype=kernel_dtype,
        )

        cur_k_tile[:, :] = nl.load(
            key[batch_id, head_id, :, :],
            dtype=cur_k_tile.dtype,
        )

        v_hbm_tile = value[batch_id, head_id]
        # load at granularity of B_P_SIZE
        for v_i in nl.affine_range(LARGE_Q_TILE_SIZE // B_P_SIZE):
            load_v_tile(
                v_hbm_tile=v_hbm_tile,
                cur_v_tile=cur_v_tile,
                v_i=v_i,
            )

        for i0 in nl.affine_range(n_large_q_tile):
            for i1 in nl.affine_range(n_small_in_large_q_tile):
                i = i0 * n_small_in_large_q_tile + i1
                cur_mask = nl.load(
                    active_mask[
                        nl.ds(i * B_P_SIZE, B_P_SIZE),
                        nl.ds(0, LARGE_Q_TILE_SIZE),
                    ],
                    dtype=active_mask.dtype,
                )
                for i_q_h in nl.affine_range(q_h_per_k_h):
                    q_tile = nl.ndarray(
                        (B_D_SIZE, B_P_SIZE),
                        dtype=kernel_dtype,
                    )
                    q_hbm_tile = query[
                        batch_id,
                        head_id * q_h_per_k_h + i_q_h,
                        :,
                        i0,
                        nl.ds(i1 * B_P_SIZE, B_P_SIZE),
                    ]
                    q_sbuf_tile = nl.load(
                        q_hbm_tile,
                        dtype=kernel_dtype,
                    )  # load (d, 128) tile in SBUF
                    q_tile[:, :] = q_sbuf_tile * softmax_scale
                    _flash_attention_core(
                        q_local_tile=q_tile,
                        k=cur_k_tile,
                        v=cur_v_tile,
                        o_buffer=o_buffer_sbuf[i0, i1, i_q_h],
                        l_buffer=l_buffer_sbuf[i0, i1, i_q_h],
                        m_buffer=m_buffer_sbuf[i0, i1, i_q_h],
                        kernel_dtype=kernel_dtype,
                        acc_type=acc_type,
                        tile_mask=cur_mask,
                        use_causal_mask=True,
                        q_tile_idx=i,
                        initialize=False,
                        LARGE_TILE_SZ=LARGE_Q_TILE_SIZE,
                        B_P_SIZE=B_P_SIZE,
                        B_F_SIZE=B_F_SIZE,
                        B_D_SIZE=B_D_SIZE,
                    )

    # -------- write output to buffer on HBM ------------ #
    for i_q_h in nl.affine_range(q_h_per_k_h):
        for i0 in nl.affine_range(n_large_q_tile):
            for i1 in nl.affine_range(n_small_in_large_q_tile):
                i = i0 * n_small_in_large_q_tile + i1
                out = nl.multiply(
                    o_buffer_sbuf[i0, i1, i_q_h],
                    nl.exp(m_buffer_sbuf[i0, i1, i_q_h] -
                           l_buffer_sbuf[i0, i1, i_q_h]),
                    dtype=kernel_dtype,
                )

                nl.store(
                    o[
                        batch_id,
                        head_id * q_h_per_k_h + i_q_h,
                        nl.ds(i * B_P_SIZE, B_P_SIZE),
                        :,
                    ],
                    out,
                )
    return o


def flash_attn_varlen_blocksparse_nkifunc(
    query,
    key,
    value,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    n_kv_head=None,
    head_size=None,
    mixed_precision=True,
):
    if n_kv_head is None:
        n_kv_head = key_cache.shape[1]
    assert key_cache.shape[1] == n_kv_head
    if head_size is None:
        head_size = key_cache.shape[-1]
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        tile_q_indices=tile_q_indices,
        tile_block_tables=tile_block_tables,
        tile_masks=tile_masks,
        active_mask=active_mask,
        softmax_scale=1.0 / (head_size**0.5),
        mixed_precision=mixed_precision,
    )

    o = flash_paged_attention_with_schedule[1, n_kv_head](**kwargs)
    return o
