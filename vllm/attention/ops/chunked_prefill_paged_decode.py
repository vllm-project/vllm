# SPDX-License-Identifier: Apache-2.0

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch
import triton
import triton.language as tl
import triton_dejavu

from vllm import _custom_ops as ops
from vllm.platforms.rocm import use_rocm_custom_paged_attention

from .prefix_prefill import context_attention_fwd


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


import time
t_prefix = 0.0
t_paged = 0.0

@triton_dejavu.jitcache(
    check_keys=["USE_ALIBI_SLOPES", "SLIDING_WINDOW"],
    cache_launch_grid=False,
)
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
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_k_cache_3: tl.int64,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_v_cache_3: tl.int64,  # int
        query_start_len_ptr,  # [num_seqs+1]
        BLOCK_Q: tl.constexpr, # int
        num_seqs: tl.int32,
):

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        mid_val = tl.load(query_start_len_ptr + mid) // BLOCK_Q + mid
        if mid_val <= q_block_global_idx:
            left = mid + 1
        else:
            right = mid

    seq_idx = left - 1
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    #print("seq_idx: %d, q_block_idx: %d, kv_head_idx: %d" % (seq_idx, q_block_idx, kv_head_idx))

    #tl.device_print("%d %d %d " % (max_num_q_blocks, seq_idx, q_block_idx))

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    #print("q_block_idx*BLOCK_Q: %d, cur_batch_query_len: %d" % (q_block_idx*BLOCK_Q, cur_batch_query_len))

    if q_block_local_idx*BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_Q * num_queries_per_kv)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)

    #print("offs_m: ", offs_m)
    #print("offs_d: ", offs_d)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    #print("query_offset_0: ", query_offset_0)
    #print("query_offset_1: ", query_offset_1)

    query_offset = (query_offset_0[:, None] * query_stride_0 +
                    query_offset_1[:, None] * query_stride_1 +
                    offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    #print("query_mask_0: ", query_mask_0)
    #print("query_mask_1: ", query_mask_1)

    # Q : (BLOCK_Q * num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_Q * num_queries_per_kv], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_Q * num_queries_per_kv], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q * num_queries_per_kv, HEAD_SIZE_PADDED],
                   dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)

        v_offset = (physical_block_idx * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_2 +
                    offs_d[None, :] * stride_v_cache_3 +
                    offs_n[:, None] * stride_v_cache_1)

        k_offset = (physical_block_idx * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_2 +
                    offs_d[:, None] * stride_k_cache_3 +
                    offs_n[None, :] * stride_k_cache_1)

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

        #print("seq_offset: ", seq_offset)
        #print("context_len: ", context_len + query_pos)

        #boundary = tl.full([BLOCK_SIZE], context_len+query_lke, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_Q * num_queries_per_kv, BLOCK_SIZE,)
        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, 0.0,
                     float("-inf")).to(tl.float32)
        S += scale * tl.dot(Q, K)

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW, S,
                         -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (BLOCK_Q * num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # P : (BLOCK_Q * num_queries_per_kv, BLOCK_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_Q * num_queries_per_kv,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_Q * num_queries_per_kv, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_Q * num_queries_per_kv, BLOCK_SIZE,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_Q * num_queries_per_kv, BLOCK_SIZE,)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]

    output_offset = (query_offset_0[:, None] * output_stride_0 +
                     query_offset_1[:, None] * output_stride_1 +
                     offs_d[None, :])

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def chunked_prefill_paged_decode(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
):

    use_alibi_slopes = alibi_slopes is not None

    #print("q.shape: ", q.shape)
    #print("k.shape: ", k.shape)
    #print("v.shape: ", v.shape)
    #print("seqused_k: ", seqused_k)
    #print("window_size: ", window_size)
    #print("cu_seqlens_q: ", cu_seqlens_q)

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv),
                                    16)

    #print("block_size: ", block_size)
    #print("num_seqs: ", num_seqs)
    #print("num_query_heads: ", num_query_heads)
    #print("num_kv_heads: ", num_kv_heads)
    #print("head_size: ", head_size)
    #print("max_seqlen_q: ", max_seqlen_q)
    #print("seqused_k: ", seqused_k)

    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    '''
    t0 = time.time()
    torch.cuda.synchronize()

    q_block_start_idx = torch.empty(size=(num_seqs+1,), device=q.device, dtype=torch.int32)
    q_block_start_idx[0] = 0
    for i in range(num_seqs):
        this_q_len = cu_seqlens_q[i+1]-cu_seqlens_q[i]
        this_n_q_blocks = triton.cdiv(this_q_len, BLOCK_Q)
        q_block_start_idx[i+1] = q_block_start_idx[i] + this_n_q_blocks

    tot_num_q_blocks = q_block_start_idx[num_seqs]

    q_block_seq_idx = torch.empty(size=(tot_num_q_blocks,), device=q.device, dtype=torch.int32)
    for i in range(num_seqs):
        start_idx, stop_idx = q_block_start_idx[i], q_block_start_idx[i+1]
        q_block_seq_idx[start_idx:stop_idx] = i

    torch.cuda.synchronize()
    global t_prefix
    t_prefix += time.time()-t0
    '''

    #t0 = time.time()

    #print("num_queries_per_kv: ", num_queries_per_kv)
    #print("BLOCK_Q:            ", BLOCK_Q)
    #print("num_query_blocks:   ", num_query_blocks)
    #print("num_seqs:           ", num_seqs)
    #print("max_num_query_blocks: ", max_num_query_blocks)


    total_num_q_blocks = cu_seqlens_q[num_seqs].to(device="cpu", non_blocking=False).item() // BLOCK_Q + num_seqs
    

    kernel_paged_attention_2d[(
        total_num_q_blocks,
        num_kv_heads,
    )](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=num_queries_per_kv_padded,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        SLIDING_WINDOW=(1+window_size[0]),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
    )

    #torch.cuda.synchronize()
    #global t_paged
    #t_paged += time.time()-t0

    #print("t_prefix: %.2f seconds, t_paged: %.2f seconds" % (t_prefix, t_paged))
