# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.platforms.rocm import use_rocm_custom_paged_attention
from vllm.triton_utils import tl, triton

from .prefix_prefill import context_attention_fwd


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def kernel_paged_attention_2d(
        output_ptr,  # [num_tokens, num_query_heads, head_size]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
        value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        alibi_slopes_ptr,  # [num_query_heads]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        num_queries_per_kv_padded: tl.constexpr,  # int
        block_table_stride: tl.int64,  # int
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        output_stride_0: tl.int64,  # int
        output_stride_1: tl.int64,  # int, should be equal to head_size
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        x: tl.constexpr,  # int
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_k_cache_3: tl.constexpr,  # int
        stride_k_cache_4: tl.constexpr,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_v_cache_3: tl.constexpr,  # int
        filter_by_query_len: tl.constexpr,  # bool
        query_start_len_ptr,  # [num_seqs+1]
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx +
                                              1)
        cur_batch_query_len = cur_batch_in_all_stop_index \
            - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded)

    query_offset = (cur_batch_in_all_start_index * query_stride_0 +
                    query_head_idx[:, None] * query_stride_1)

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                        0).to(tl.int1)

    # Q : (num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.full([num_queries_per_kv_padded], 1.0, dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED],
                   dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx,
                              mask=head_mask,
                              other=0.0)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)

        v_offset = (physical_block_idx * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_1 +
                    offs_d[None, :] * stride_v_cache_2 +
                    offs_n[:, None] * stride_v_cache_3)

        k_offset = (physical_block_idx * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_1 +
                    (offs_d[:, None] // x) * stride_k_cache_2 +
                    offs_n[None, :] * stride_k_cache_3 +
                    (offs_d[:, None] % x) * stride_k_cache_4)

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(key_cache_ptr + k_offset,
                         mask=dim_mask[:, None],
                         other=0.0)

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (BLOCK_SIZE, HEAD_SIZE)
        V_load = tl.load(value_cache_ptr + v_offset,
                         mask=dim_mask[None, :],
                         other=0.0)

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        # S : (num_queries_per_kv, BLOCK_SIZE,)
        S = tl.where(head_mask[:, None] & seq_mask, 0.0,
                     float("-inf")).to(tl.float32)
        S += scale * tl.dot(Q, K)

        context_len = seq_len - 1

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S,
                         -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # P : (num_queries_per_kv, BLOCK_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (num_queries_per_kv,)
        l_j = tl.sum(P, axis=1)

        # alpha : (num_queries_per_kv, )
        alpha = tl.exp(M - m_j)

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]

    output_offset = (cur_batch_in_all_start_index * output_stride_0 +
                     query_head_idx * output_stride_1)

    tl.store(
        output_ptr + output_offset[:, None] +
        tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )


@triton.jit
def kernel_paged_attention_3d(
        segm_output_ptr,
        # [num_seqs, num_query_heads, num_segments, head_size_padded]
        segm_max_ptr,  # [num_seqs, num_query_heads, num_segments]
        segm_expsum_ptr,  # [num_seqs, num_query_heads, num_segments]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
        value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        alibi_slopes_ptr,  # [num_query_heads]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        num_seqs,  # int
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        num_queries_per_kv_padded: tl.constexpr,  # int
        block_table_stride: tl.int64,
        query_stride_0: tl.int64,
        query_stride_1: tl.int64,
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        x: tl.constexpr,  # int
        stride_k_cache_0: tl.int64,
        stride_k_cache_1: tl.int64,
        stride_k_cache_2: tl.int64,
        stride_k_cache_3: tl.constexpr,
        stride_k_cache_4: tl.constexpr,
        stride_v_cache_0: tl.int64,
        stride_v_cache_1: tl.int64,
        stride_v_cache_2: tl.int64,
        stride_v_cache_3: tl.constexpr,
        filter_by_query_len: tl.constexpr,  # bool
        query_start_len_ptr,  # [num_seqs+1]
        MIN_NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
        MAX_NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx +
                                              1)
        cur_batch_query_len = cur_batch_in_all_stop_index \
            - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments and segment size for this particular sequence
    num_segments = (MAX_NUM_SEGMENTS_PER_SEQ
                    if num_seqs <= 64 else MIN_NUM_SEGMENTS_PER_SEQ)
    blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)

    if segm_idx * blocks_per_segment * BLOCK_SIZE >= seq_len:
        return

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded)

    query_offset = (cur_batch_in_all_start_index * query_stride_0 +
                    query_head_idx[:, None] * query_stride_1)

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                        0).to(tl.int1)

    # Q : (num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.full([num_queries_per_kv_padded], 1.0, dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED],
                   dtype=tl.float32)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx,
                              mask=head_mask,
                              other=0.0)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # iterate through tiles within current segment
    for j in range(
            segm_idx * blocks_per_segment,
            min((segm_idx + 1) * blocks_per_segment, num_blocks),
    ):
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)

        v_offset = (physical_block_idx * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_1 +
                    offs_d[None, :] * stride_v_cache_2 +
                    offs_n[:, None] * stride_v_cache_3)

        k_offset = (physical_block_idx * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_1 +
                    (offs_d[:, None] // x) * stride_k_cache_2 +
                    offs_n[None, :] * stride_k_cache_3 +
                    (offs_d[:, None] % x) * stride_k_cache_4)

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(key_cache_ptr + k_offset,
                         mask=dim_mask[:, None],
                         other=0.0)

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (BLOCK_SIZE, HEAD_SIZE)
        V_load = tl.load(value_cache_ptr + v_offset,
                         mask=dim_mask[None, :],
                         other=0.0)

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        # S : (num_queries_per_kv, BLOCK_SIZE,)
        S = tl.where(head_mask[:, None] & seq_mask, 0.0,
                     float("-inf")).to(tl.float32)
        S += scale * tl.dot(Q, K)

        context_len = seq_len - 1

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S,
                         -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # P : (num_queries_per_kv, BLOCK_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (num_queries_per_kv,)
        l_j = tl.sum(P, axis=1)

        # alpha : (num_queries_per_kv, )
        alpha = tl.exp(M - m_j)

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ \
            * HEAD_SIZE_PADDED)
        + query_head_idx[:, None] * (MAX_NUM_SEGMENTS_PER_SEQ \
            * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )

    segm_offset = (seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ) +
                   query_head_idx * MAX_NUM_SEGMENTS_PER_SEQ + segm_idx)
    tl.store(segm_max_ptr + segm_offset, M, mask=head_mask)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=head_mask)


@triton.jit
def reduce_segments(
        output_ptr,  # [num_seqs, num_query_heads, head_size]
        segm_output_ptr,
        # [num_seqs, num_query_heads, max_num_segments, head_size_padded]
        segm_max_ptr,  # [num_seqs, num_query_heads, max_num_segments]
        segm_expsum_ptr,  # [num_seqs, num_query_heads, max_num_segments]
        seq_lens_ptr,  # [num_seqs]
        num_seqs,  # int
        num_query_heads: tl.constexpr,  # int
        output_stride_0: tl.int64,
        output_stride_1: tl.int64,
        block_table_stride: tl.int64,
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int, must be power of 2
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        filter_by_query_len: tl.constexpr,  # bool
        query_start_len_ptr,  # [num_seqs+1]
        MIN_NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
        MAX_NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx +
                                              1)
        cur_batch_query_len = cur_batch_in_all_stop_index \
            - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments and segment size for this particular sequence
    num_segments = (MAX_NUM_SEGMENTS_PER_SEQ
                    if num_seqs <= 64 else MIN_NUM_SEGMENTS_PER_SEQ)
    blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, blocks_per_segment * BLOCK_SIZE)
    segm_mask = tl.arange(0, MAX_NUM_SEGMENTS_PER_SEQ) < tl.full(
        [MAX_NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                        0).to(tl.int1)

    # load segment maxima
    segm_offset = (seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ) +
                   query_head_idx * MAX_NUM_SEGMENTS_PER_SEQ +
                   tl.arange(0, MAX_NUM_SEGMENTS_PER_SEQ))
    segm_max = tl.load(segm_max_ptr + segm_offset,
                       mask=segm_mask,
                       other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset,
                          mask=segm_mask,
                          other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ \
            * HEAD_SIZE_PADDED)
        + query_head_idx * (MAX_NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, MAX_NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc = tl.sum(segm_output, axis=0) / overall_expsum

    # write result
    output_offset = (cur_batch_in_all_start_index * output_stride_0 +
                     query_head_idx * output_stride_1 +
                     tl.arange(0, HEAD_SIZE_PADDED))
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def chunked_prefill_paged_decode(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
):

    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[1]**0.5)

    use_alibi_slopes = alibi_slopes is not None

    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if max_query_len > 1:
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            kv_cache_dtype=kv_cache_dtype,
            k_cache=key_cache,
            v_cache=value_cache,
            b_loc=block_table,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_seq_len=max_seq_len,
            max_input_len=max_query_len,
            k_scale=k_scale,
            v_scale=v_scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            skip_decode=True,
        )

    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    num_queries_per_kv = query.shape[1] // key.shape[1]
    head_size = query.shape[2]

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert key_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]
        assert value_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = current_platform.fp8_dtype()
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)

    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv),
                                    16)

    use_custom = use_rocm_custom_paged_attention(query.dtype, head_size,
                                                 block_size,
                                                 num_queries_per_kv,
                                                 max_seq_len, sliding_window,
                                                 kv_cache_dtype, alibi_slopes)
    if use_custom:
        _PARTITION_SIZE_ROCM = 256
        max_num_partitions = ((max_seq_len + _PARTITION_SIZE_ROCM - 1) //
                              _PARTITION_SIZE_ROCM)
        assert _PARTITION_SIZE_ROCM % block_size == 0
        total_num_seq = block_table.shape[0]
        tmp_output = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions,
                  head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

        ops.paged_attention_rocm(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale=sm_scale,
            block_tables=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_size=block_size,
            max_seq_len=max_seq_len,
            alibi_slopes=alibi_slopes,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    elif max_seq_len < 256:
        # use 2d kernel to avoid reduction overhead for short sequences
        kernel_paged_attention_2d[(
            num_seqs,
            num_kv_heads,
        )](
            output_ptr=output,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            block_tables_ptr=block_table,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=alibi_slopes,
            scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=block_table.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            SLIDING_WINDOW=sliding_window,
            x=key_cache.shape[4],
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_k_cache_4=key_cache.stride(4),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            filter_by_query_len=True,
            query_start_len_ptr=query_start_loc,
        )
    else:
        # minimum and maximum number of segments to be processed in parallel
        # in the sequence (key/value) length dimension
        MIN_NUM_SEGMENTS = 4
        MAX_NUM_SEGMENTS = 16

        segm_output = torch.empty(
            num_seqs,
            num_query_heads,
            MAX_NUM_SEGMENTS,
            triton.next_power_of_2(head_size),
            dtype=torch.float32,
            device=query.device,
        )
        segm_max = torch.empty(
            num_seqs,
            num_query_heads,
            MAX_NUM_SEGMENTS,
            dtype=torch.float32,
            device=query.device,
        )
        segm_expsum = torch.empty(
            num_seqs,
            num_query_heads,
            MAX_NUM_SEGMENTS,
            dtype=torch.float32,
            device=query.device,
        )

        kernel_paged_attention_3d[(num_seqs, num_kv_heads, MAX_NUM_SEGMENTS)](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            block_tables_ptr=block_table,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=alibi_slopes,
            scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=block_table.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            SLIDING_WINDOW=sliding_window,
            x=key_cache.shape[4],
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_k_cache_4=key_cache.stride(4),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            filter_by_query_len=True,
            query_start_len_ptr=query_start_loc,
            MIN_NUM_SEGMENTS_PER_SEQ=MIN_NUM_SEGMENTS,
            MAX_NUM_SEGMENTS_PER_SEQ=MAX_NUM_SEGMENTS,
        )

        reduce_segments[(num_seqs, num_query_heads)](
            output_ptr=output,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seq_lens,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            block_table_stride=block_table.stride(0),
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            filter_by_query_len=True,
            query_start_len_ptr=query_start_loc,
            MIN_NUM_SEGMENTS_PER_SEQ=MIN_NUM_SEGMENTS,
            MAX_NUM_SEGMENTS_PER_SEQ=MAX_NUM_SEGMENTS,
        )
