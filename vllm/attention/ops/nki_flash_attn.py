# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
import torch
from neuronxcc import nki
from neuronxcc.nki.language import par_dim


def ceil_div(a, b):
    return (a + b - 1) // b


def is_power_of_2(x):
    return x > 0 and (x & (x - 1)) == 0


@nki.jit
def load_block_tables(block_tables_hbm, num_tiles, num_blocks_per_tile):
    """
    Load block tables from HBM into SRAM

    `block_tables_hbm` has shape `(num_tiles * num_blocks_per_tile, )`.
    In case `num_tiles > B_P_SIZE`, we need further tile `num_tile` dimension.
    """
    B_P_SIZE = 128

    # reshape as `(num_tiles, num_blocks_per_tile)`
    assert len(block_tables_hbm.shape) == 1
    (num_total_blocks, ) = block_tables_hbm.shape
    assert num_blocks_per_tile * num_tiles == num_total_blocks
    block_tables_hbm = block_tables_hbm.reshape(
        (num_tiles, num_blocks_per_tile))

    block_tables_sbuf = nl.zeros(
        (ceil_div(num_tiles,
                  B_P_SIZE), par_dim(B_P_SIZE), num_blocks_per_tile),
        dtype=nl.int32,
    )
    for i in nl.affine_range(ceil_div(num_tiles, B_P_SIZE)):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(num_blocks_per_tile)[None, :]
        block_tables_sbuf[i, i_p, i_f] = nl.load(
            block_tables_hbm[i_p + i * B_P_SIZE, i_f],
            dtype=nl.int32,
            mask=(i_p + i * B_P_SIZE < num_tiles),
        )
    return block_tables_sbuf


@nki.jit
def transform_block_tables_for_indirect_load(
    block_tables,
    block_size_tiling_factor,
    num_head,
    head_id,
):
    """
    This function does two things:
    1. calculate new `block_tables` for a `head_id` after flattening
    `num_block`, `num_head`, and `block_size_tiling_factor` dimensions
    2. transpose the result so that `block_table` for each tile is mapped to
    SBUF Partition dimension for vectorized DMA

    Tiling trick to further improve DMA performance:
    Given KV cache shape `(num_block, num_head, block_size, D)`, when loading M
    blocks of a given `head_id` from HBM, the load `cache[block_tables,
    head_id]` has shape `(M, block_size, D)`. If M < B_P_SIZE = 128, DMA may not
    fully utilize hardware parallelization. The solution is to tile `block_size`
    into `(block_size_tiling_factor, tiled_block_size)` s.t. `M *
    block_size_tiling_factor = B_P_SIZE`. After tiling, KV cache has shape
    `(num_block, num_head, block_size_tiling_factor, tiled_block_size, D)`. 

    Note:
    We don't further tile D dimension as small DMA size also hurts performance.
    """
    B_P_SIZE = 128
    num_partitions, num_tiles_per_partition, num_blocks_per_tile = (
        block_tables.shape)
    assert num_tiles_per_partition == B_P_SIZE
    assert is_power_of_2(
        num_blocks_per_tile), f"{num_blocks_per_tile=} is not power of 2"

    num_loads = ceil_div(num_blocks_per_tile, B_P_SIZE)
    block_tables_transposed = nl.ndarray(
        (
            num_loads,
            par_dim(B_P_SIZE),
            num_partitions * num_tiles_per_partition,
        ),
        dtype=nl.int32,
    )

    # prepare iota ahead of time to avoid repeatedly using Gpsimd
    if num_head > 1:
        head_id = nisa.iota(head_id, dtype=nl.int32).reshape((1, 1))
        head_id = nl.transpose(
            head_id.broadcast_to((1, num_tiles_per_partition)))
        if num_blocks_per_tile > 1:
            head_id = head_id.broadcast_to(
                (num_tiles_per_partition, num_blocks_per_tile))

    if block_size_tiling_factor > 1:
        broadcast_shape = (
            num_tiles_per_partition,
            num_blocks_per_tile,
            block_size_tiling_factor,
        )
        offset = nisa.iota(nl.arange(block_size_tiling_factor)[None, None, :],
                           dtype=nl.int32).broadcast_to(broadcast_shape)

    for partition_id in nl.affine_range(num_partitions):
        block_tables_partition = block_tables[partition_id]
        if num_head > 1:
            # fuse num_block and num_head dimension
            block_tables_partition = block_tables_partition * num_head + head_id

        if block_size_tiling_factor > 1:
            # need to apply block size tiling trick
            assert num_blocks_per_tile * block_size_tiling_factor == B_P_SIZE
            block_tables_partition = ((block_tables_partition *
                                       block_size_tiling_factor).reshape(
                                           (num_tiles_per_partition,
                                            num_blocks_per_tile,
                                            1)).broadcast_to(broadcast_shape))
            new_block_tables = block_tables_partition + offset
            new_block_tables = new_block_tables.reshape(
                (num_tiles_per_partition, B_P_SIZE))
        else:
            new_block_tables = block_tables_partition

        # transpose the block table so that it can be used by vector DGE
        for i in nl.affine_range(num_loads):
            i_p = nl.arange(B_P_SIZE)[:, None]
            i_f = (partition_id * num_tiles_per_partition +
                   nl.arange(num_tiles_per_partition)[None, :])
            block_tables_transposed[i, i_p, i_f] = nl.transpose(
                new_block_tables[:, nl.ds(i * B_P_SIZE, B_P_SIZE)])
    return block_tables_transposed


@nki.jit
def load_kv_tile_from_cache(
    cur_k_tile,
    cur_v_tile,
    key_cache,
    value_cache,
    block_tables,
    large_k_tile_idx,
    num_blocks_per_large_tile,
    tiled_block_size,
    B_P_SIZE,
    B_D_SIZE,
):
    """
    Load KV cache and transform Key and Value into layout required by Matmul

    Vectorized DMA Load layout:
    Key and Value: (par_dim(B_P_SIZE), seqlen_kv // B_P_SIZE * B_D_SIZE)

    Layout used by attention matmuls:
    Key: (par_dim(B_D_SIZE), seqlen_kv)
    Value: (seqlen_kv // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE)
           equivalent to (par_dim(B_P_SIZE), seqlen_kv // B_P_SIZE * B_D_SIZE)
    """
    # load key cache
    num_loads = ceil_div(num_blocks_per_large_tile, B_P_SIZE)
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(tiled_block_size * B_D_SIZE)[None, :]
        loaded = nl.load(key_cache[block_tables[load_idx, i_p,
                                                large_k_tile_idx], i_f])
        if cur_k_tile.dtype != loaded.dtype:
            loaded = nl.copy(loaded, dtype=cur_k_tile.dtype)
        # Transpose SBUF tensor using PE
        for tb_i in nl.affine_range(tiled_block_size):
            cur_k_tile[
                :,
                nl.ds(
                    load_idx * B_P_SIZE * tiled_block_size + tb_i * B_P_SIZE,
                    B_P_SIZE,
                ),
            ] = nl.transpose(loaded[:, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)])

    # load value cache
    for load_idx in nl.affine_range(num_loads):
        loaded = nl.load(value_cache[block_tables[load_idx, i_p,
                                                  large_k_tile_idx], i_f])
        if cur_v_tile.dtype != loaded.dtype:
            loaded = nl.copy(loaded, dtype=cur_v_tile.dtype)
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(tiled_block_size * B_D_SIZE)[None, :]
        cur_v_tile[
            :,
            nl.ds(
                load_idx * tiled_block_size * B_D_SIZE,
                tiled_block_size * B_D_SIZE,
            ),
        ] = loaded


@nki.jit
def transpose_p_local(p_local_transposed,
                      p_local,
                      LARGE_TILE_SZ,
                      B_F_SIZE=512):
    for i in nl.affine_range(LARGE_TILE_SZ // B_F_SIZE):
        if nisa.get_nc_version() == nisa.nc_version.gen3:
            p_local_t_tmp = nl.ndarray((par_dim(128), B_F_SIZE),
                                       buffer=nl.sbuf,
                                       dtype=p_local.dtype)
        else:
            p_local_t_tmp = nl.ndarray((par_dim(128), B_F_SIZE),
                                       buffer=nl.psum,
                                       dtype=np.float32)

        for j in nl.affine_range(B_F_SIZE // 128):
            j_128_slice = nl.ds(j * 128, 128)
            i_j_128_slice = nl.ds(i * B_F_SIZE + j * 128, 128)

            if nisa.get_nc_version() == nisa.nc_version.gen3:
                p_local_t_tmp[:, j_128_slice] = nisa.dma_transpose(
                    p_local[:, i_j_128_slice])
            else:
                p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(
                    p_local[:, i_j_128_slice])

        p_local_transposed[:, nl.ds(i * B_F_SIZE, B_F_SIZE)] = nl.copy(
            p_local_t_tmp, dtype=p_local_transposed.dtype)


@nki.jit
def _flash_attention_core(
    q_local_tile,
    k,
    v,
    o_buffer,
    l_buffer,
    m_buffer,
    kernel_dtype,
    acc_type,
    tile_mask,
    use_causal_mask,
    q_tile_idx=None,
    initialize=False,
    LARGE_TILE_SZ=2048,
    B_P_SIZE=128,
    B_F_SIZE=512,
    B_D_SIZE=128,
    qk_res_buffer=None,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    The q_local_tile has (B_P_SIZE, B_D_SIZE)
    The K and V have shape (B_D_SIZE, LARGE_TILE_SZ), whose free dimension will
    be split into size B_F_SIZE tiles

    The results are stored in the following three buffers
    o_buffer: (B_P_SIZE, d)
    l_buffer: (B_P_SIZE, 1)
    m_buffer: (B_P_SIZE, 1)

    All IO buffers are in SBUF.
    """
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE

    qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ),
                            buffer=nl.sbuf,
                            dtype=acc_type)
    max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile),
                           dtype=acc_type)
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        if use_causal_mask:
            # mask are used to only apply computation to the lower half of the
            # matrix, which reduce the arithmetic intensity by up to 50%
            multiplication_required_selection = (q_tile_idx * B_P_SIZE
                                                 >= k_i * B_F_SIZE)
        else:
            multiplication_required_selection = True

        if multiplication_required_selection:
            qk_psum = nl.ndarray((par_dim(B_P_SIZE), B_F_SIZE),
                                 dtype=np.float32,
                                 buffer=nl.psum)  # (128, 512)
            qk_psum[:, :] = nl.matmul(q_local_tile,
                                      k[:, k_i_b_f_slice],
                                      transpose_x=True)  # (p(128), 512)
            qk_res_buf[:, k_i_b_f_slice] = nl.where(
                tile_mask[:, k_i_b_f_slice],
                qk_psum[:, nl.ds(0, B_F_SIZE)],
                -9984.0,
                dtype=acc_type,
            )
        else:
            qk_res_buf[:, k_i_b_f_slice] = -9984.0

        # Calculate max of the current tile
        max_local[:, k_i] = nisa.tensor_reduce(
            np.max,
            qk_res_buf[:, k_i_b_f_slice],
            axis=(1, ),
            dtype=acc_type,
            negate=False,
        )

    if qk_res_buffer is not None:
        qk_res_buffer[:, :] = nl.copy(qk_res_buf[:, :])

    max_ = nisa.tensor_reduce(
        np.max,
        max_local[:, :],
        axis=(1, ),
        dtype=acc_type,
        negate=False,
    )

    o_previous_scaled = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE),
                                   dtype=o_buffer.dtype)

    if initialize:
        m_buffer[:, 0] = nl.copy(max_)
        m_current = max_
    else:
        m_previous = nl.copy(m_buffer[:, 0])
        m_buffer[:, 0] = nl.maximum(m_previous, max_)  # (128,1)

        m_current = m_buffer[:, 0]
        # Compute scaling factor
        alpha = nisa.activation(
            np.exp,
            m_previous,
            bias=-1 * m_current,
            scale=1.0,
        )
        o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha)

    p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ),
                         dtype=kernel_dtype)
    REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)

    p_partial_sum = nl.ndarray(
        (par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE),
        dtype=acc_type,
    )

    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

        # compute exp(qk - max)
        # Compute partial row - tile sum of exp(qk - max))
        # FIXME : Use activation accumulate to accumulate over k_r_i loop ?
        p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
            np.exp,
            qk_res_buf[:, k_r_i_reduce_slice],
            bias=-1 * m_current,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=p_partial_sum[:, k_r_i],
            dtype=kernel_dtype,
        )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    p_local_transposed = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ),
                                    dtype=kernel_dtype)
    transpose_p_local(
        p_local_transposed=p_local_transposed,
        p_local=p_local,
        LARGE_TILE_SZ=LARGE_TILE_SZ,
        B_F_SIZE=B_F_SIZE,
    )

    pv_psum = nl.zeros(
        (par_dim(B_P_SIZE), B_D_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)],
            v[:, nl.ds(k_i * B_D_SIZE, B_D_SIZE)],
            transpose_x=True,
        )  # (128, 128) (p(Br), d)

    if initialize:
        o_buffer[:, :] = nl.copy(pv_psum[:, :])
        l_buffer[:, 0] = nl.add(nl.log(ps), max_)
    else:
        o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum)

        l_prev = l_buffer[:, 0]
        l_exp = nl.add(
            nl.exp(nl.subtract(l_prev, m_current)),
            ps,
        )
        l_buffer[:, 0] = nl.add(m_current, nl.log(l_exp))


@nki.jit
def load_v_tile(v_hbm_tile, cur_v_tile, large_tile_idx, v_i, LARGE_TILE_SZ):
    B_P_SIZE = 128
    B_D_SIZE = v_hbm_tile.shape[-1]
    loaded = nl.load(v_hbm_tile[
        nl.ds(large_tile_idx * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE),
        :,
    ])
    if cur_v_tile.dtype != loaded.dtype:
        loaded = nl.copy(loaded, dtype=cur_v_tile.dtype)
    cur_v_tile[:, nl.ds(v_i * B_D_SIZE, B_D_SIZE)] = loaded


@nki.jit
def flash_paged_attention(
    query,
    key,
    value,
    key_cache,
    value_cache,
    block_tables,
    mask,
    softmax_scale=None,
    mixed_precision=True,
    LARGE_TILE_SZ=2048,
    return_debug_tensors=False,
):
    """
    Flash PagedAttention Forward Kernel.

    IO tensor layouts:
      - query: shape   (1, n_heads, d, seq_q)
      - key:   shape   (1, n_kv_heads, d, seq_k)
      - value: shape   (1, n_kv_heads, seq_v, d)
      - key_cache: (num_blocks, n_kv_heads, block_size, d)
      - value_cache: (num_blocks, n_kv_heads, block_size, d)
      - block_tables: (num_active_blocks, )
      - mask: (seq_q, num_active_blocks * block_size + seq_q)
      - o: shape (1, n_heads, seq_q, d)

      - This kernel requires seq_k == seq_v
      - We use continuous batching by default, so the batch dimension is
        always 1, and different requests are concatenated along sequence
        dimension.
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
      - LARGE_TILE_SZ: `default=2048`, size of the kv tile size for attention
        computation reduction

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
    b, h, d, seqlen_q = query.shape
    B_D_SIZE = d
    n_tile_q = seqlen_q // B_P_SIZE  # since q will be loaded on tensor engine
    num_blocks, k_h, block_size, _ = key_cache.shape
    q_h_per_k_h = h // k_h
    assert b == 1, f"invalid batch size {b=}"
    assert d <= 128, f" we do not support head_dim > 128, got head dim {d=}"
    cache_shape = (num_blocks, k_h, block_size, d)
    assert (tuple(key_cache.shape) == cache_shape
            ), f"{key_cache.shape=} mismatch, expect {cache_shape}"
    assert (tuple(value_cache.shape) == cache_shape
            ), f"{value_cache.shape=} mismatch, expect {cache_shape}"
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

    assert (
        nl.program_ndim() == 2
    ), f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"
    batch_id = nl.program_id(axis=0)
    head_id = nl.program_id(axis=1)

    (num_active_blocks, ) = block_tables.shape
    context_kv_len = num_active_blocks * block_size
    assert (
        LARGE_TILE_SZ % B_F_SIZE == 0
    ), f"Need {LARGE_TILE_SZ=} to be divisible by {B_F_SIZE=} in transpose_p"
    assert (context_kv_len % LARGE_TILE_SZ == 0
            ), f"Need {context_kv_len=} to be divisible by {LARGE_TILE_SZ=}"

    num_blocks_per_large_tile = LARGE_TILE_SZ // block_size
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

    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    softmax_scale = softmax_scale or (1.0 / (d**0.5))
    num_large_k_tile = context_kv_len // LARGE_TILE_SZ

    o = nl.ndarray((b, h, seqlen_q, d),
                   dtype=query.dtype,
                   buffer=nl.shared_hbm)
    hbm_l_buffer, hbm_m_buffer, hbm_qk_res, qk_res_buffer = (
        None,
        None,
        None,
        None,
    )
    if return_debug_tensors:
        hbm_l_buffer = nl.ndarray((b, h, seqlen_q),
                                  dtype=acc_type,
                                  buffer=nl.shared_hbm)
        hbm_m_buffer = nl.ndarray((b, h, seqlen_q),
                                  dtype=acc_type,
                                  buffer=nl.shared_hbm)
        hbm_qk_res = nl.ndarray((b, h, B_P_SIZE, seqlen_q),
                                dtype=acc_type,
                                buffer=nl.shared_hbm)
        qk_res_buffer = nl.zeros(
            (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), seqlen_q),
            dtype=acc_type,
            buffer=nl.sbuf,
            lazy_initialization=True,
        )
    block_tables_sbuf = load_block_tables(
        block_tables_hbm=block_tables,
        num_tiles=num_large_k_tile,
        num_blocks_per_tile=num_blocks_per_large_tile,
    )

    # On Neuron, we need B_P_SIZE = 128 blocks to make DMA efficient
    if num_blocks_per_large_tile < B_P_SIZE:
        # we checked num_blocks_per_tile is a power of 2
        assert B_P_SIZE % num_blocks_per_large_tile == 0
        block_size_tiling_factor = B_P_SIZE // num_blocks_per_large_tile
        # We assume block_size >= block_size_tiling_factor
        assert block_size % block_size_tiling_factor == 0
    else:
        block_size_tiling_factor = 1
    tiled_block_size = block_size // block_size_tiling_factor

    # Indirect DMA load must be placed along Partition Dimension
    block_tables_sbuf = transform_block_tables_for_indirect_load(
        block_tables_sbuf,
        block_size_tiling_factor=block_size_tiling_factor,
        num_head=k_h,
        head_id=head_id,
    )

    # Flatten KV cache to be 2D for loading into SBUF
    new_cache_shape = (
        num_blocks * k_h * block_size_tiling_factor,
        tiled_block_size * d,
    )
    key_cache = key_cache.reshape(new_cache_shape)
    value_cache = value_cache.reshape(new_cache_shape)

    # Global Flash Attention accumulators
    o_buffer = nl.zeros(
        (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), d),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )
    l_buffer = nl.zeros(
        (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), 1),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )
    m_buffer = nl.zeros(
        (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), 1),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )

    for large_k_tile_idx in nl.sequential_range(0, num_large_k_tile):
        num_loads = ceil_div(num_blocks_per_large_tile, B_P_SIZE)
        cur_k_tile = nl.ndarray(
            (par_dim(B_D_SIZE), LARGE_TILE_SZ),
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
            large_k_tile_idx=large_k_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            tiled_block_size=tiled_block_size,
            B_P_SIZE=B_P_SIZE,
            B_D_SIZE=B_D_SIZE,
        )

        for i in nl.affine_range(n_tile_q):
            cur_mask = nl.load(mask[
                nl.ds(i * B_P_SIZE, B_P_SIZE),
                nl.ds(large_k_tile_idx * LARGE_TILE_SZ, LARGE_TILE_SZ),
            ])
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                q_hbm_tile = query[batch_id, head_id * q_h_per_k_h + i_q_h]
                q_sbuf_tile = nl.load(q_hbm_tile[:,
                                                 nl.ds(i *
                                                       B_P_SIZE, B_P_SIZE)])
                if q_sbuf_tile.dtype != kernel_dtype:
                    q_sbuf_tile = nl.copy(q_sbuf_tile, dtype=kernel_dtype)
                q_tile[:, :] = q_sbuf_tile * softmax_scale

                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    o_buffer=o_buffer[i, i_q_h],
                    l_buffer=l_buffer[i, i_q_h],
                    m_buffer=m_buffer[i, i_q_h],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=cur_mask,
                    use_causal_mask=False,
                    q_tile_idx=i,
                    initialize=large_k_tile_idx == 0,
                    LARGE_TILE_SZ=LARGE_TILE_SZ,
                    B_P_SIZE=B_P_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                )

    # compute attention between input query, key and value
    if key is not None and value is not None:
        B_F_SIZE = min(seqlen_q, B_F_SIZE)
        LARGE_TILE_SZ = seqlen_q

        cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ),
                                dtype=kernel_dtype)
        cur_v_tile = nl.ndarray(
            (par_dim(B_P_SIZE), LARGE_TILE_SZ // B_P_SIZE * B_D_SIZE),
            dtype=kernel_dtype,
        )

        loaded = nl.load(key[batch_id, head_id, :, :])
        if loaded.dtype != kernel_dtype:
            loaded = nl.copy(loaded, dtype=kernel_dtype)
        cur_k_tile[:, :] = loaded

        v_hbm_tile = value[batch_id, head_id]
        for v_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
            load_v_tile(
                v_hbm_tile=v_hbm_tile,
                cur_v_tile=cur_v_tile,
                large_tile_idx=0,
                v_i=v_i,
                LARGE_TILE_SZ=LARGE_TILE_SZ,
            )

        for i in nl.affine_range(n_tile_q):
            cur_mask = nl.load(mask[
                nl.ds(i * B_P_SIZE, B_P_SIZE),
                nl.ds(context_kv_len, LARGE_TILE_SZ),
            ])
            for i_q_h in nl.affine_range(q_h_per_k_h):

                q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                q_hbm_tile = query[batch_id, head_id * q_h_per_k_h + i_q_h]
                q_sbuf_tile = nl.load(q_hbm_tile[:,
                                                 nl.ds(i *
                                                       B_P_SIZE, B_P_SIZE)])
                if q_sbuf_tile.dtype != kernel_dtype:
                    q_sbuf_tile = nl.copy(q_sbuf_tile, dtype=kernel_dtype)
                q_tile[:, :] = q_sbuf_tile * softmax_scale
                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    o_buffer=o_buffer[i, i_q_h],
                    l_buffer=l_buffer[i, i_q_h],
                    m_buffer=m_buffer[i, i_q_h],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=cur_mask,
                    use_causal_mask=True,
                    q_tile_idx=i,
                    initialize=False,
                    LARGE_TILE_SZ=LARGE_TILE_SZ,
                    B_P_SIZE=B_P_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                    qk_res_buffer=(qk_res_buffer[i, i_q_h]
                                   if qk_res_buffer is not None else None),
                )

    # -- -- -- -- write output to buffer on HBM -- -- -- -- -- -- #
    for i_q_h in nl.affine_range(q_h_per_k_h):
        for i in nl.affine_range(n_tile_q):
            out = nl.multiply(
                o_buffer[i, i_q_h],
                nl.exp(m_buffer[i, i_q_h] - l_buffer[i, i_q_h]),
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
            # maximum and summation statistics
            if return_debug_tensors:
                nl.store(
                    hbm_m_buffer[
                        batch_id,
                        head_id * q_h_per_k_h + i_q_h,
                        nl.ds(i * B_P_SIZE, B_P_SIZE),
                    ],
                    m_buffer[i, i_q_h, :, :],
                )
                nl.store(
                    hbm_l_buffer[
                        batch_id,
                        head_id * q_h_per_k_h + i_q_h,
                        nl.ds(i * B_P_SIZE, B_P_SIZE),
                    ],
                    l_buffer[i, i_q_h],
                )
                nl.store(
                    hbm_qk_res[batch_id, head_id * q_h_per_k_h + i_q_h, :, :],
                    qk_res_buffer[batch_id, i_q_h, :, :],
                )

    if return_debug_tensors:
        return o, hbm_m_buffer, hbm_l_buffer, hbm_qk_res
    return o


def reorder_context_mask(mask, LARGE_TILE_SZ, block_size):
    """
    Reorder the mask to make it compatible with the flash attention kernel.

    We vectorize KV cache read to improve DMA utilization. However, the layout
    that maximizes DMA bandwidth changes the order tokens are consumed.
    
    The token layout (inner 2 dimensions) after vectorized load is (B_P_SIZE,
    tiled_block_size) in a tile of `B_P_SIZE * tiled_block_size` tokens. And
    each step the engine consumes a column (rather than a row) of B_P_SIZE
    tokens. Therefore, the tokens are visited in a strided way.

    To make sure mask matches the order tokens are consumed, we need to properly
    transpose mask.
    """
    total_query_len, total_seq_len = mask.shape
    context_kv_len = total_seq_len - total_query_len

    B_P_SIZE = 128
    assert (LARGE_TILE_SZ
            >= B_P_SIZE), f"{LARGE_TILE_SZ=} must be larger than {B_P_SIZE=}"
    num_tiled_blocks = max(B_P_SIZE, LARGE_TILE_SZ // block_size)
    tiled_block_size = LARGE_TILE_SZ // num_tiled_blocks
    if tiled_block_size > 1:
        # Mask reordering is needed when tiled_block_size > 1
        device = mask.device
        mask = mask.cpu()
        context_mask = mask[:, :context_kv_len]
        context_mask = context_mask.view(
            total_query_len,
            context_kv_len // LARGE_TILE_SZ,
            num_tiled_blocks // B_P_SIZE,
            B_P_SIZE,
            tiled_block_size,
        )
        context_mask = context_mask.transpose(3, 4).reshape(
            total_query_len, context_kv_len)
        new_mask = mask[:, context_kv_len:]
        return torch.concat([context_mask, new_mask], dim=1).to(device)
    else:
        return mask


def flash_attn_varlen_nkifunc(
    query,
    key,
    value,
    key_cache,
    value_cache,
    block_table,
    attn_mask,
    n_kv_head=None,
    head_size=None,
    LARGE_TILE_SZ=2048,
    mixed_precision=True,
):
    """
    Compute flash paged attention for variable length sequences.

    This function is a wrapper around the flash attention NKI kernel. It takes
    in the following arguments:
      - query: (1, n_heads, d, seq_q)
      - key:   (1, n_kv_heads, d, seq_k)
      - value: (1, n_kv_heads, seq_v, d)
      - key_cache:   (n_blocks, n_kv_heads, block_size, d)
      - value_cache: (n_blocks, n_kv_heads, block_size, d)
      - block_tables: (n_active_blocks, )
      - attn_mask: (seq_q, n_active_blocks * block_size + seq_q)

    Notes:
      - attn_mask must be reordered outside using `reorder_context_mask`
      - Key/value cache layout must be (n_blocks, n_kv_heads, block_size, d) 
        for better DMA throughput
    """
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
        block_tables=block_table,
        mask=attn_mask,
        softmax_scale=1.0 / (head_size**0.5),
        mixed_precision=mixed_precision,
        LARGE_TILE_SZ=LARGE_TILE_SZ,
    )

    o = flash_paged_attention[1, n_kv_head](**kwargs)
    return o
