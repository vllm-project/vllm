import math
from typing import Optional

import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform

#This code is adapted from this FlashNN project
#(https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py)

_SEQ_PARTITION_SIZE = 512 if not current_platform.is_rocm() else 1024


def paged_attn_decode_v1(
        output: torch.Tensor,  #[num_seqs, num_kv_heads*query_grp_sz, head_sz]
        query: torch.Tensor,  #[num_seqs, num_kv_heads*query_grp_sz, head_sz]
        key_cache: torch.Tensor,  #[num_seqs, num_kv_heads, kv_blk_sz, head_sz]
        value_cache: torch.
    Tensor,  #[num_seqs, num_kv_heads, kv_blk_sz, head_sz]
        block_tables: torch.Tensor,  #[num_seqs, max_num_blks_per_seq]
        seq_lens: torch.Tensor,  #[num_seqs]
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale,
        v_scale,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0):
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = query.shape[1] // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert (key_cache.dtype == torch.uint8)
        assert (value_cache.dtype == torch.uint8)

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = (torch.float8_e4m3fn
                            if not current_platform.is_rocm() else
                            torch.float8_e4m3fnuz)
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)

    if (key_cache.dtype == torch.uint8
            or value_cache.dtype == torch.uint8 and kv_cache_dtype == "auto"):
        raise ValueError("kv_cache_dtype='auto' unsupported for\
                FP8 KV Cache Decode kernel")

    #MHA- Multi-Head Attention
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v1_wo_dot_kernel[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            query.stride(0),
            query.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            MAX_SEQ_LEN_POW2=max_seq_len)
    #GQA - Grouped Query Attention
    else:
        grid = (num_seqs, num_kv_heads, 1)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v1_w_dot_kernel[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz)


@triton.jit
def _paged_attn_decode_v1_wo_dot_kernel(
    out,  #[num_seqs, num_kv_heads * query_grp_sz, head_sz]
    q_ptr,  #[num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,  #[num_blks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr,  #[num_blks, num_kv_heads, kv_blk_sz, head_sz]
    blk_tables_ptr,  #[num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  #[num_seqs]
    alibi_slopes_ptr,  #[num_q_heads]
    scale,
    k_scale,
    v_scale,
    stride_q_s,
    stride_q_h,
    stride_o_s,
    stride_o_nh,
    stride_o_hs,
    stride_k_b,
    stride_k_nh,
    stride_k_kb,
    stride_bt_s,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    MAX_SEQ_LEN_POW2: tl.constexpr,
):
    head_idx = tl.program_id(axis=0)
    seq_idx = tl.program_id(axis=1)
    kv_head_idx = head_idx // QUERY_GRP_SZ

    log2e: tl.constexpr = 1.4426950408889634

    dtype = q_ptr.dtype.element_ty

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if seq_idx >= seq_len:
        return

    num_kv_blks = tl.cdiv(seq_len, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)

    #load alibi slopes [1]
    if alibi_slopes_ptr is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes_ptr + head_idx)

    #load q [1, HEAD_SZ_POW2]
    q_offs = seq_idx * stride_q_s + head_idx * stride_q_h + head_sz_offs
    q = tl.load(q_ptr + q_offs, mask=head_sz_offs < HEAD_SZ)
    q = (q * scale).to(dtype)

    acc = tl.zeros([KV_BLK_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = float("-inf")
    exp_sum = 0.0

    kv_offs = (kv_head_idx * stride_k_nh + blk_offs[:, None] * stride_k_kb +
               head_sz_offs[None, :])
    blk_tbl_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_nums = tl.load(blk_tbl_start_ptr + b)
        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = b * KV_BLK_SZ + blk_offs
        kv_mask = ((blk_seq_offs[:, None] < seq_len) &
                   (blk_offs[:, None] < KV_BLK_SZ) &
                   (head_sz_offs[None, :] < HEAD_SZ))

        #load k [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = (k_0.to(tl.float32) * k_scale) if k_0.dtype.is_fp8() else k_0
        k = k.to(dtype)

        #qk #[KV_BLK_SZ_POW2]
        qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        if alibi_slopes_ptr is not None:
            qk += (alibi_slope * (blk_seq_offs - seq_len + 1)).to(tl.float32)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        max_logit_new = tl.maximum(tl.max(qk, axis=0), max_logit)

        # p: [KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        #load v [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask)
        v = (v_0.to(tl.float32) * v_scale) if v_0.dtype.is_fp8() else v_0
        v = v.to(dtype)

        acc += p[:, None] * v

        exp_sum = exp_sum * alpha + tl.sum(p, axis=0)
        max_logit = max_logit_new

    acc = acc / exp_sum

    offs_out = (seq_idx * stride_o_s + head_idx * stride_o_nh + head_sz_offs)
    out_mask = head_sz_offs < HEAD_SZ
    tl.store(out + offs_out, tl.sum(acc, axis=0), mask=out_mask)


@triton.jit
def _paged_attn_decode_v1_w_dot_kernel(
    out_ptr,  #[num_seqs, num_kv_heads * query_grp_sz, head_sz]
    q_ptr,  #[num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,  #[num_blocks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr,  #[num_blocks, num_kv_heads, kv_blk_sz, head_sz]
    blk_tables_ptr,  #[num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,  #[num_seqs]
    alibi_slopes,  #[num_kv_heads*query_grp_sz]
    scale,
    k_scale,
    v_scale,
    stride_o_s,
    stride_o_nh,
    stride_o_hs,
    stride_q_s,
    stride_q_nh,
    stride_q_hs,
    stride_k_b,
    stride_k_nh,
    stride_k_kb,
    stride_k_hs,
    stride_bt_s,
    stride_bt_nb,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    log2e: tl.constexpr = 1.4426950408889634

    dtype = q_ptr.dtype.element_ty

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if seq_idx >= seq_len:
        return

    num_kv_blks = tl.cdiv(seq_len, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)

    #load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(alibi_slopes + kv_head_idx * QUERY_GRP_SZ +
                              q_grp_offs,
                              mask=q_grp_offs < QUERY_GRP_SZ,
                              other=0.0)

    q_offs = (
        seq_idx * stride_q_s +
        (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh +
        head_sz_offs[None, :] * stride_q_hs)

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] <
                                                     HEAD_SZ)
    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(dtype)

    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)

    kv_offs = (kv_head_idx * stride_k_nh + blk_offs[:, None] * stride_k_kb +
               head_sz_offs[None, :] * stride_k_hs)
    blk_tbl_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_nums = tl.load(blk_tbl_start_ptr + b)
        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = b * KV_BLK_SZ + blk_offs
        kv_mask = ((blk_seq_offs[:, None] < seq_len) &
                   (blk_offs[:, None] < KV_BLK_SZ) &
                   (head_sz_offs[None, :] < HEAD_SZ))

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
        k = k.to(dtype)

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = tl.where((q_grp_offs[:, None] < QUERY_GRP_SZ) &
                      (blk_seq_offs[None, :] < seq_len), qk, float("-inf"))

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] *
                   (blk_seq_offs - seq_len + 1)[None, :]).to(tl.float32)

        qk = tl.where((q_grp_offs[:, None] < QUERY_GRP_SZ) &
                      (blk_seq_offs[None, :] < seq_len), qk, float("-inf"))
        max_logit_new = tl.maximum(tl.max(qk, axis=1), max_logit)

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ, HEAD_SZ]
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask,
                      other=0.0).to(dtype)
        v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
        v = v.to(dtype)

        p = p.to(v.dtype)
        acc += tl.dot(p, v, out_dtype=tl.float32)

        exp_sum = exp_sum * alpha + tl.sum(p, axis=1)
        max_logit = max_logit_new

    acc = acc / exp_sum[:, None]

    out_offs = (
        seq_idx * stride_o_s +
        (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_o_nh +
        head_sz_offs[None, :])

    out_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] <
                                                       HEAD_SZ)
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def paged_attn_decode_v2(
        output: torch.Tensor,  #[num_seqs, num_kv_heads*query_grp_sz, head_sz],
        query: torch.Tensor,  #[num_seqs, num_kv_heads*query_grp_sz, head_sz],
        key_cache: torch.
    Tensor,  #[num_seqs, num_kv_heads, kv_blk_sz, head_sz] ,
        value_cache: torch.
    Tensor,  #[num_seqs, num_kv_heads, kv_blk_sz, head_sz] ,
        block_tables: torch.Tensor,  #[num_seqs, max_num_blks_per_seq],
        seq_lens: torch.Tensor,  #[num_seqs],
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale,
        v_scale,
        max_num_partitions: int,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0):
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)

    if "fp8" in kv_cache_dtype:
        assert (key_cache.dtype == torch.uint8)
        assert (value_cache.dtype == torch.uint8)

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = (torch.float8_e4m3fn
                            if not current_platform.is_rocm() else
                            torch.float8_e4m3fnuz)
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)

    if (key_cache.dtype == torch.uint8
            or value_cache.dtype == torch.uint8 and kv_cache_dtype == "auto"):
        raise ValueError("kv_cache_dtype='auto' unsupported for\
                FP8 KV Cache Decode kernel")

    #Note: There is a bug in triton.next_power_of_2 function which causes it
    #to update the passed in arg, so that's why we have a workaround here
    #max_num_partitions_pow2 = triton.next_power_of_2(max_num_partitions)
    if max_num_partitions == 0:
        max_num_partitions_pow2 = 1
    else:
        max_num_partitions_pow2 = 2**math.ceil(math.log2(max_num_partitions))

    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    #MHA
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, max_num_partitions)
        shape_info = (num_seqs, num_q_heads, max_num_partitions)
        exp_sums = torch.empty(size=shape_info,
                               dtype=torch.float32,
                               device=output.device)
        max_logits = torch.empty(size=shape_info,
                                 dtype=torch.float32,
                                 device=output.device)
        tmp_output = torch.empty((*shape_info, head_sz),
                                 dtype=output.dtype,
                                 device=output.device)
        _paged_attn_decode_v2_wo_dot_kernel[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            block_tables.stride(1),
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_BLKS_PER_SEQ=block_tables.shape[1],
            MAX_SEQ_LEN_POW2=max_seq_len)
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v2_wo_dot_reduce_kernel[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS=int(max_num_partitions),
            MAX_NUM_SEQ_PARTITIONS_POW2=int(max_num_partitions_pow2))
    #GQA
    else:
        grid = (num_seqs, num_kv_heads, max_num_partitions)
        shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
        max_logits = torch.empty(shape_info,
                                 dtype=torch.float32,
                                 device=output.device)
        exp_sums = torch.empty(shape_info,
                               dtype=torch.float32,
                               device=output.device)
        tmp_output = torch.empty(*shape_info,
                                 head_sz,
                                 dtype=output.dtype,
                                 device=output.device)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v2_w_dot_kernel[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE)
        grid = (num_seqs, num_kv_heads, 1)
        _paged_attn_decode_v2_w_dot_reduce_kernel[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS=int(max_num_partitions),
            MAX_NUM_SEQ_PARTITIONS_POW2=int(
                triton.next_power_of_2(max_num_partitions)))


@triton.jit
def _paged_attn_decode_v2_wo_dot_kernel(
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    blk_tables_ptr,
    seq_lens_ptr,
    scale,
    k_scale,
    v_scale,
    alibi_slopes,
    stride_exp_s,
    stride_exp_h,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_q_s,
    stride_q_h,
    stride_k_b,
    stride_k_nh,
    stride_k_kb,
    stride_bt_s,
    stride_bt_nb,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_BLKS_PER_SEQ: tl.constexpr,
    MAX_SEQ_LEN_POW2: tl.constexpr,
):
    head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    seq_part_idx = tl.program_id(2)
    kv_head_idx = head_idx // QUERY_GRP_SZ

    log2e: tl.constexpr = 1.4426950408889634

    dtype = q_ptr.dtype.element_ty

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if seq_part_idx * SEQ_PARTITION_SZ >= seq_len:
        return

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)

    # load alibi slopes
    if alibi_slopes is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes + head_idx)

    # load q[HEAD_SZ]
    q_offs = (seq_idx * stride_q_s + head_idx * stride_q_h + head_sz_offs)
    q = tl.load(q_ptr + q_offs, mask=head_sz_offs < HEAD_SZ)
    q = (q * scale).to(dtype)

    acc = tl.zeros([KV_BLK_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = float("-inf")
    exp_sum = 0.0

    kv_offs = (kv_head_idx * stride_k_nh + blk_offs[:, None] * stride_k_kb +
               head_sz_offs[None, :])
    kv_blk_start = seq_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    blk_tables_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s

    for b in range(num_kv_blks):
        kv_blk_idx = kv_blk_start + b
        kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_idx * stride_bt_nb)

        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = kv_blk_idx * KV_BLK_SZ + blk_offs
        kv_mask = ((blk_seq_offs[:, None] < seq_len) &
                   (blk_offs[:, None] < KV_BLK_SZ) &
                   (head_sz_offs[None, :] < HEAD_SZ))

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
        k = k.to(dtype)

        # qk: [KV_BLK_SZ_POW2]
        qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        if alibi_slopes is not None:
            qk += (alibi_slope * (blk_seq_offs - seq_len + 1)).to(tl.float32)
        qk = tl.where(blk_seq_offs < seq_len, qk, float("-inf"))

        max_logit_new = tl.maximum(max_logit, tl.max(qk, axis=0))

        # p: [KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
        v = v.to(dtype)

        # acc: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        acc += p[:, None] * v

        exp_sum = exp_sum * alpha + tl.sum(p, axis=0)
        max_logit = max_logit_new

    acc = acc / exp_sum

    max_logits_offs = (seq_idx * stride_exp_s + head_idx * stride_exp_h +
                       seq_part_idx)

    tl.store(max_logits_ptr + max_logits_offs, max_logit)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum)

    logits_offs = (seq_idx * stride_logits_s + head_idx * stride_logits_h +
                   seq_part_idx * stride_logits_p + head_sz_offs)
    logits_mask = head_sz_offs < HEAD_SZ
    tl.store(logits_ptr + logits_offs, tl.sum(acc, axis=0), mask=logits_mask)


@triton.jit
def _paged_attn_decode_v2_wo_dot_reduce_kernel(
    out,
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptr,
    seq_lens,
    stride_out_n,
    stride_out_h,
    stride_exp_sums_n,
    stride_exp_sums_h,
    stride_logits_n,
    stride_logits_h,
    stride_logits_b,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr,
):
    #get seq_idx, head_idx, seq_len
    head_idx = tl.program_id(axis=0)
    seq_idx = tl.program_id(axis=1)

    seq_len = tl.load(seq_lens + seq_idx)
    num_partitions = tl.cdiv(seq_len, SEQ_PARTITION_SZ)

    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    seq_part_offs = tl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2)

    max_logit = float("-inf")
    acc = tl.zeros([HEAD_SZ_POW2], dtype=tl.float32)
    global_exp_sum = tl.zeros([1], dtype=tl.float32)

    #load max_logits [MAX_NUM_SEQ_PARTITIONS_POW2]
    max_logits_offs = seq_idx * stride_exp_sums_n + \
        head_idx * stride_exp_sums_h + seq_part_offs
    max_logits_mask = seq_part_offs < num_partitions
    max_logits = tl.load(
        max_logits_ptr + max_logits_offs,
        mask=max_logits_mask,
        other=float("-inf"),
    )

    #find max_logit
    max_logit = tl.max(max_logits, axis=0)

    #load exp_sum [MAX_NUM_SEQ_PARTITIONS_POW2]
    exp_sums_offs = seq_idx * stride_exp_sums_n + \
        head_idx * stride_exp_sums_h + seq_part_offs
    exp_sums_mask = seq_part_offs < num_partitions
    exp_sums = tl.load(
        exp_sums_ptr + exp_sums_offs,
        mask=exp_sums_mask,
        other=0.0,
    )

    #rescaled_exp_sum and global_exp_sum
    # [MAX_NUM_SEQ_PARTITIONS_POW2]
    rescaled_exp_sum = exp_sums * tl.exp(max_logits - max_logit)
    global_exp_sum += tl.sum(rescaled_exp_sum, axis=0)
    rescaled_exp_sum /= global_exp_sum

    #load logits
    logits_offs = (seq_idx * stride_logits_n + head_idx * stride_logits_h +
                   seq_part_offs[:, None] * stride_logits_b +
                   head_sz_offs[None, :])
    logits_mask = (seq_part_offs[:, None] <
                   num_partitions) & (head_sz_offs[None, :] < HEAD_SZ)

    logits = tl.load(logits_ptr + logits_offs, mask=logits_mask, other=0.0)
    acc += tl.sum(logits * rescaled_exp_sum[:, None], axis=0)

    #store the final output
    out_ptr = seq_idx * stride_out_n + head_idx * stride_out_h + head_sz_offs
    out_mask = (head_sz_offs < HEAD_SZ)
    tl.store(out + out_ptr, acc, mask=out_mask)


@triton.jit
def _paged_attn_decode_v2_w_dot_kernel(
        exp_sums_ptr,  #[num_seqs, num_kv_heads, max_parts, q_grp_sz]
        max_logits_ptr,  #[num_seqs, num_kv_heads, max_parts, q_grp_sz]
        logits_ptr,  #[num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
        q_ptr,  #[num_seqs, num_kv_heads * query_grp_sz, head_sz]
        k_cache_ptr,  #[num_seqs, num_kv_heads, kv_blk_sz, head_sz]
        v_cache_ptr,  #[num_seqs, num_kv_heads, kv_blk_sz, head_sz]
        blk_tables_ptrs,  #[num_seqs, max_num_blks_per_seq]
        seq_lens_ptr,  #[num_seqs]
        scale,
        k_scale,
        v_scale,
        alibi_slopes,
        stride_max_logits_s,
        stride_max_logits_nh,
        stride_max_logits_p,
        stride_logits_s,
        stride_logits_nh,
        stride_logits_p,
        stride_logits_g,
        stride_q_s,
        stride_q_nh,
        stride_k_b,
        stride_k_nh,
        stride_k_kb,
        stride_bt_s,
        HEAD_SZ: tl.constexpr,
        HEAD_SZ_POW2: tl.constexpr,
        QUERY_GRP_SZ: tl.constexpr,
        QUERY_GRP_SZ_POW2: tl.constexpr,
        KV_BLK_SZ: tl.constexpr,
        KV_BLK_SZ_POW2: tl.constexpr,
        SEQ_PARTITION_SZ: tl.constexpr):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_part_idx = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634
    dtype = q_ptr.dtype.element_ty

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if seq_part_idx * SEQ_PARTITION_SZ >= seq_len:
        return

    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= seq_len:
        return

    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)

    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)

    #load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(alibi_slopes + kv_head_idx * QUERY_GRP_SZ +
                              q_grp_offs,
                              mask=q_grp_offs < QUERY_GRP_SZ,
                              other=0.0)

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * stride_q_s +
        (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh +
        head_sz_offs[None, :])
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] <
                                                     HEAD_SZ)
    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * scale).to(dtype)

    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    max_logit = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)

    kv_offs = (kv_head_idx * stride_k_nh + blk_offs[:, None] * stride_k_kb +
               head_sz_offs[None, :])
    kv_blk_start = seq_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    for b in range(num_kv_blks):
        kv_blk_idx = kv_blk_start + b
        kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_idx)

        kv_blk_offs = kv_blk_nums * stride_k_b + kv_offs
        blk_seq_offs = kv_blk_idx * KV_BLK_SZ + blk_offs
        kv_mask = ((blk_seq_offs[:, None] < seq_len) &
                   (blk_offs[:, None] < KV_BLK_SZ) &
                   (head_sz_offs[None, :] < HEAD_SZ))

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_0 = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)
        k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
        k = k.to(dtype)

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = tl.where((q_grp_offs[:, None] < QUERY_GRP_SZ) &
                      (blk_seq_offs[None, :] < seq_len), qk, float("-inf"))

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] *
                   (blk_seq_offs - seq_len + 1)[None, :]).to(tl.float32)
        qk = tl.where((q_grp_offs[:, None] < QUERY_GRP_SZ) &
                      (blk_seq_offs[None, :] < seq_len), qk, float("-inf"))

        max_logit_new = tl.maximum(max_logit, tl.max(qk, axis=1))

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v_0 = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask,
                      other=0.0).to(dtype)
        v = v_0.to(tl.float32) * v_scale if v_0.dtype.is_fp8() else v_0
        v = v.to(dtype)

        p = p.to(v.dtype)
        acc += tl.dot(p, v, out_dtype=tl.float32)

        exp_sum = exp_sum * alpha + tl.sum(p, axis=1)
        max_logit = max_logit_new

    acc = acc / exp_sum[:, None]

    max_logits_offs = (seq_idx * stride_max_logits_s +
                       kv_head_idx * stride_max_logits_nh +
                       seq_part_idx * stride_max_logits_p + q_grp_offs)
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs, max_logit, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)

    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (seq_part_idx * stride_logits_p +
                    q_grp_offs[:, None] * stride_logits_g +
                    head_sz_offs[None, :])

    tl.store(logits_ptr + logits_offs, acc, mask=q_mask)


@triton.jit
def _paged_attn_decode_v2_w_dot_reduce_kernel(
        out_ptr,  # [num_seqs, num_kv_heads, q_grp_sz, head_sz]
        exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
        max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
        logits_ptrs,  # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
        seq_lens_ptr,  #[num_seqs]
        stride_o_s,
        stride_o_h,
        stride_exp_sums_s,
        stride_exp_sums_h,
        stride_exp_sums_p,
        stride_logits_s,
        stride_logits_h,
        stride_logits_p,
        stride_logits_g,
        HEAD_SZ: tl.constexpr,
        HEAD_SZ_POW2: tl.constexpr,
        QUERY_GRP_SZ: tl.constexpr,
        QUERY_GRP_SZ_POW2: tl.constexpr,
        SEQ_PARTITION_SZ: tl.constexpr,
        MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
        MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_partitions = tl.cdiv(seq_len, SEQ_PARTITION_SZ)

    part_offs = tl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    head_offs = tl.arange(0, HEAD_SZ_POW2)

    #get global max logit
    exp_sums_offs = (seq_idx * stride_exp_sums_s +
                     kv_head_idx * stride_exp_sums_h +
                     part_offs[:, None] * stride_exp_sums_p +
                     q_grp_offs[None, :])
    exp_sums_mask = (part_offs[:, None] <
                     num_partitions) & (q_grp_offs[None, :] < QUERY_GRP_SZ)

    # max_logits: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    max_logits = tl.load(max_logits_ptr + exp_sums_offs,
                         mask=exp_sums_mask,
                         other=float("-inf"))
    # max_logit: [QUERY_GRP_SZ_POW2]
    ml = tl.max(max_logits, axis=0)

    #Rescale the exp sums and compute the global sum
    # exp_sums: [MAX_NUM_SEQ_PARTITIONS, QUERY_GRP_SZ_POW2]
    exp_sums = tl.load(exp_sums_ptr + exp_sums_offs,
                       mask=exp_sums_mask,
                       other=0.0)
    exp_sums *= tl.exp(max_logits - ml[None, :])

    # exp_sum: [QUERY_GRP_SZ_POW2]
    exp_sum = tl.sum(exp_sums, axis=0)

    # p: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    p = exp_sums / exp_sum[None, :]
    p = tl.reshape(p, (MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2, 1))

    #logits_offset
    logits_offset = (seq_idx * stride_logits_s +
                     kv_head_idx * stride_logits_h +
                     part_offs[:, None, None] * stride_logits_p +
                     q_grp_offs[None, :, None] * stride_logits_g +
                     head_offs[None, None, :])
    #load logits
    logits_mask = (part_offs[:, None] < num_partitions) & (q_grp_offs[None, :]
                                                           < QUERY_GRP_SZ)
    logits = tl.load(logits_ptrs + logits_offset,
                     mask=logits_mask[:, :, None],
                     other=0.0)

    #out: [QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    out = tl.sum((logits * p).to(tl.float32), axis=0)

    #store output
    out_offs = (
        seq_idx * stride_o_s +
        (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_o_h +
        head_offs[None, :])
    tl.store(out_ptr + out_offs,
             out,
             mask=(q_grp_offs[:, None] < QUERY_GRP_SZ) &
             (head_offs[None, :] < HEAD_SZ))
