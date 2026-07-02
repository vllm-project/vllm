# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>
import math

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .prefix_prefill import context_attention_fwd

logger = init_logger(__name__)

float8_info = torch.finfo(current_platform.fp8_dtype())

_MAX_SPLITS = 16
_DEFAULT_COMPUTE_BLOCK_SIZE = 32


# The split-kv kernel has the best performance when the
# compute block size is 32.
def _choose_compute_block_size(physical_block_size: int) -> int:
    """Choose the logical attention tile size inside a physical KV block."""
    for block_size in (32, 16, 8, 4, 2):
        if physical_block_size % block_size == 0:
            return min(block_size, _DEFAULT_COMPUTE_BLOCK_SIZE)
    return 1


def has_native_kv_cache_layout(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> bool:
    """Return whether KV cache blocks can use the native ROCm pairing.

    The native reshape_and_cache writer assumes packed blocks. If cache update
    needs reshape_and_cache_flash for a stride-padded hybrid layout, decode
    should use the matching Triton path too.
    """
    return (
        key_cache.stride(0) == key_cache.shape[1:].numel()
        and value_cache.stride(0) == value_cache.shape[1:].numel()
    )


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def kernel_paged_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale_inv,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    num_queries_per_kv_padded: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    PHYSICAL_BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    x: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.int64,  # int
    stride_k_cache_4: tl.int64,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.int64,  # int
    filter_by_query_len: tl.constexpr,  # bool
    query_start_len_ptr,  # [num_seqs+1]
    USE_SINKS: tl.constexpr,  # bool
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # Q : (num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
        L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_head_idx,
            mask=head_mask,
            other=float("-inf"),
        ).to(dtype=tl.float32)
        L = tl.where(float("-inf") < M, 1.0, 0.0)

    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_head_idx, mask=head_mask, other=0.0
        )

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    # iterate through tiles
    for j in range(0, num_blocks):
        start_n = j * BLOCK_SIZE
        # Calculate the logical location within a non-standard physical block,
        # such as 544 in Qwen/Qwen3-Next-80B-A3B-Thinking.
        # Supports non-contiguous mapping
        # from logical blocks to physical blocks
        abs_token_idx = start_n + offs_n
        # Slots >= seq_len are unwritten KV cache and may hold NaN/garbage
        # (e.g. the tail of the last partial block). They are score-masked
        # below, but 0 * NaN = NaN would still poison the output, so exclude
        # them from the K/V loads too.
        kv_load_mask = abs_token_idx < seq_len
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        # Vectorized loading of physical block IDs
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE

        # 5D addressing logic of K
        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )

        # 4D addressing logic of V (Slot is innermost)
        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & kv_load_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (BLOCK_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & kv_load_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        # First calculate the dot, then apply the mask.
        qk = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))

        context_len = seq_len - 1

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # P : (num_queries_per_kv, BLOCK_SIZE,)
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)

        # l_j : (num_queries_per_kv,)
        l_j = tl.sum(p, axis=1)

        # alpha : (num_queries_per_kv, )
        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc += tl.dot(p.to(V.dtype), V)

    # epilogue
    acc = acc / (L[:, None] + 1e-10)
    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
    )

    tl.store(
        output_ptr + output_offset[:, None] + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )


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
    output_scale=None,
    # Optional tensor for sinks
    sinks=None,
    is_block_table_ptr: bool = False,
    causal: bool = True,
):
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[2] ** 0.5)

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
            fp8_out_scale=output_scale,
            sinks=sinks,
            causal=causal,
        )

    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    # key may be None in cross-attention decode (already cached from encoder)
    num_kv_heads = key.shape[1] if key is not None else key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
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
            raise ValueError(
                f"Unsupported FP8 kv_cache_dtype {kv_cache_dtype}: "
                f"should be one of 'fp8', 'fp8_e4m3', 'fp8_e5m2'."
            )

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)

    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)

    from vllm.platforms.rocm import use_rocm_custom_paged_attention

    use_custom = use_rocm_custom_paged_attention(
        query.dtype,
        head_size,
        block_size,
        num_queries_per_kv,
        max_seq_len,
        sliding_window,
        kv_cache_dtype,
        alibi_slopes,
        sinks,
    )
    has_native_layout = has_native_kv_cache_layout(key_cache, value_cache)
    # Force Triton for non-standard blocks like Qwen3's 544 and for
    # stride-padded hybrid layouts. The latter use reshape_and_cache_flash
    # during cache update, so keep decode on the matching stride-aware path.
    is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)
    if not is_pow2 or not has_native_layout:
        use_custom = False

    if use_custom:
        _PARTITION_SIZE_ROCM = 256
        max_num_partitions = (
            max_seq_len + _PARTITION_SIZE_ROCM - 1
        ) // _PARTITION_SIZE_ROCM
        assert _PARTITION_SIZE_ROCM % block_size == 0
        total_num_seq = block_table.shape[0]
        tmp_output = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions, head_size),
            dtype=query.dtype,
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
            fp8_out_scale=output_scale,
        )
    else:
        logger.warning_once(
            "Cannot use ROCm custom paged attention kernel,"
            " falling back to Triton implementation."
        )
        real_block_size = value_cache.shape[3]
        # The standard model directly uses the original block_size.
        # Non-standard 544 uses 32 to accommodate integer division logic.
        # Cap at 128 to avoid exceeding GPU shared memory limits
        # (e.g. hybrid Mamba models inflate block_size to 2048).
        # The kernel handles TRITON_BLOCK_SIZE != PHYSICAL_BLOCK_SIZE
        # via the l_block_idx/internal_offsets addressing logic.
        MAX_TRITON_BLOCK_SIZE = 128
        TRITON_BLOCK_SIZE = min(block_size, MAX_TRITON_BLOCK_SIZE) if is_pow2 else 32
        if is_block_table_ptr:
            # Using the physical base address of tensors
            kv_element_size = key_cache.element_size()
            block_byte_stride = key_cache.stride(0) * kv_element_size
            # Get the starting physical address of the KV Cache
            base_addr = key_cache.data_ptr()

            # Normalization: Directly calculate the block offset
            # of the pointer relative to the base address
            processed_block_table = ((block_table - base_addr) // block_byte_stride).to(
                torch.int32
            )
        else:
            processed_block_table = block_table.to(torch.int32)

        from vllm.platforms.rocm import on_gfx12x

        # Split kv is currently only tuned for gfx12x with head dim 256.
        use_splitkv_decode = (
            on_gfx12x()
            and query.dtype in (torch.float16, torch.bfloat16)
            and head_size == 256
            and not use_alibi_slopes
            and sliding_window == 0
            and sinks is None
            and output_scale is None
            and "fp8" not in kv_cache_dtype
        )
        if use_splitkv_decode:
            paged_attention_2d_splitkv_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=processed_block_table,
                seq_lens=seq_lens,
                scale=sm_scale,
                output=output,
                max_seq_len=max_seq_len,
                max_num_splits=_MAX_SPLITS,
                query_start_loc=query_start_loc,
                filter_by_query_len=True,
            )
        else:
            kernel_paged_attention_2d[
                (
                    num_seqs,
                    num_kv_heads,
                )
            ](
                output_ptr=output,
                query_ptr=query,
                key_cache_ptr=key_cache,
                value_cache_ptr=value_cache,
                sink_ptr=sinks,
                block_tables_ptr=processed_block_table,
                seq_lens_ptr=seq_lens,
                alibi_slopes_ptr=alibi_slopes,
                scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                out_scale_inv=1.0 / output_scale if output_scale is not None else 1.0,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                num_queries_per_kv_padded=num_queries_per_kv_padded,
                block_table_stride=processed_block_table.stride(0),
                query_stride_0=query.stride(0),
                query_stride_1=query.stride(1),
                output_stride_0=output.stride(0),
                output_stride_1=output.stride(1),
                BLOCK_SIZE=TRITON_BLOCK_SIZE,
                PHYSICAL_BLOCK_SIZE=real_block_size,
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
                USE_SINKS=sinks is not None,
                USE_FP8=output_scale is not None,
            )


@triton.jit
def kernel_paged_attention_2d_splitkv(
    mid_out_ptr,  # [num_seqs, num_query_heads, max_num_splits, head_size]
    mid_lse_ptr,  # [num_seqs, num_query_heads, max_num_splits]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    mid_out_stride_0: tl.int64,
    mid_out_stride_1: tl.int64,
    mid_out_stride_2: tl.int64,
    mid_lse_stride_0: tl.int64,
    mid_lse_stride_1: tl.int64,
    mid_lse_stride_2: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    x: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_k_cache_4: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_splits = tl.num_programs(2)

    split_len = cdiv_fn(cdiv_fn(seq_len, num_splits), BLOCK_SIZE) * BLOCK_SIZE
    split_start = split_idx * split_len
    split_end = tl.minimum(split_start + split_len, seq_len)

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )
    Q = tl.load(
        query_ptr + query_offset + offs_d[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    block_table_offset = seq_idx * block_table_stride
    offs_n = tl.arange(0, BLOCK_SIZE)

    for start_n in tl.range(split_start, split_end, BLOCK_SIZE):
        abs_token_idx = start_n + offs_n
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE
        token_mask = abs_token_idx < split_end

        # Should use stride_k_cache_4 = 1 and stride_k_cache_3 = x here
        # to make triton compiler happy.
        # However benchmark show the compiler correctly generates
        # 128bit memory access instruction,
        # but there is no obvious performance difference.
        # So we keep the original stride for better readability.
        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )
        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & token_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )
        V = tl.load(
            value_cache_ptr + v_offset,
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        S = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & token_mask[None, :], S, float("-inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)
        l_j = tl.sum(p, axis=1)

        # Previous partial sums are expressed in exp(x - M); rescale them when
        # the running maximum increases before adding this block.
        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(p.to(V.dtype), V)

    mid_out_offset = (
        seq_idx * mid_out_stride_0
        + query_head_idx[:, None] * mid_out_stride_1
        + split_idx * mid_out_stride_2
        + offs_d[None, :]
    )
    mid_lse_offset = (
        seq_idx * mid_lse_stride_0
        + query_head_idx * mid_lse_stride_1
        + split_idx * mid_lse_stride_2
    )

    has_tokens = split_end > split_start
    out = acc / (L[:, None] + 1e-10)
    lse = M + tl.log(L)

    tl.store(
        mid_out_ptr + mid_out_offset,
        out,
        mask=has_tokens & head_mask[:, None] & dim_mask[None, :],
    )
    tl.store(
        mid_lse_ptr + mid_lse_offset,
        lse,
        mask=has_tokens & head_mask,
    )


@triton.jit
def kernel_paged_attention_2d_splitkv_reduce(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    mid_out_ptr,  # [num_seqs, num_query_heads, max_num_splits, head_size]
    mid_lse_ptr,  # [num_seqs, num_query_heads, max_num_splits]
    seq_lens_ptr,  # [num_seqs]
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    mid_out_stride_0: tl.int64,
    mid_out_stride_1: tl.int64,
    mid_out_stride_2: tl.int64,
    mid_lse_stride_0: tl.int64,
    mid_lse_stride_1: tl.int64,
    mid_lse_stride_2: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    MAX_NUM_SPLITS: tl.constexpr,
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    split_len = cdiv_fn(cdiv_fn(seq_len, MAX_NUM_SPLITS), BLOCK_SIZE) * BLOCK_SIZE

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE

    M = -float("inf")
    L = 0.0
    acc = tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32)

    for split_idx in tl.range(0, MAX_NUM_SPLITS, num_stages=2):
        split_start = split_idx * split_len
        split_end = tl.minimum(split_start + split_len, seq_len)

        if split_end > split_start:
            lse = tl.load(
                mid_lse_ptr
                + seq_idx * mid_lse_stride_0
                + query_head_idx * mid_lse_stride_1
                + split_idx * mid_lse_stride_2
            )
            partial = tl.load(
                mid_out_ptr
                + seq_idx * mid_out_stride_0
                + query_head_idx * mid_out_stride_1
                + split_idx * mid_out_stride_2
                + offs_d,
                mask=dim_mask,
                other=0.0,
            )

            m_j = tl.maximum(M, lse)
            alpha = tl.exp(M - m_j)
            beta = tl.exp(lse - m_j)
            acc = acc * alpha + partial * beta
            L = L * alpha + beta
            M = m_j

    out = acc / (L + 1e-10)
    tl.store(
        output_ptr
        + cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
        + offs_d,
        out,
        mask=dim_mask,
    )


def _num_splits_heuristic(
    batch_nheads_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    """Choose split count for small-batch decode occupancy.

    Use FlashAttention's wave-efficiency heuristic: pick the smallest eligible
    split whose wave efficiency (n_waves / ceil(n_waves)) is within 85% of the
    maximum achievable.  On gfx12 torch reports WGPs while rocprof reports CUs,
    so target two workgroups per reported processor.
    """
    target_workgroups = 2 * num_sms
    if batch_nheads_mblocks >= 0.8 * target_workgroups:
        return 1

    max_splits = min(max_splits, num_sms, num_n_blocks)
    if max_splits <= 1:
        return 1

    def is_split_eligible(num_splits: int) -> bool:
        return num_splits == 1 or _cdiv(num_n_blocks, num_splits) != _cdiv(
            num_n_blocks, num_splits - 1
        )

    max_efficiency = 0.0
    efficiency = []
    for num_splits in range(1, max_splits + 1):
        if not is_split_eligible(num_splits):
            efficiency.append(0.0)
            continue

        n_waves = batch_nheads_mblocks * num_splits / target_workgroups
        eff = n_waves / math.ceil(n_waves)
        max_efficiency = max(max_efficiency, eff)
        efficiency.append(eff)

    for num_splits, eff in enumerate(efficiency, start=1):
        if is_split_eligible(num_splits) and eff >= 0.85 * max_efficiency:
            return num_splits

    return 1


def _choose_decode_num_splits(
    batch_size: int,
    num_kv_heads: int,
    max_seq_len: int,
    block_size: int,
    max_num_splits: int,
    num_sms: int,
) -> int:
    if max_seq_len <= block_size:
        return 1

    batch_nheads = batch_size * num_kv_heads
    num_n_blocks = _cdiv(max_seq_len, block_size)

    # Not enough KV blocks to keep each split busy on the GPU.
    if num_n_blocks < 2 * num_sms:
        return 1

    max_splits = min(max_num_splits, num_sms, num_n_blocks)
    return _num_splits_heuristic(batch_nheads, num_sms, num_n_blocks, max_splits)


def _get_num_splits(
    batch_size: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    max_seq_len: int,
    max_num_splits: int = _MAX_SPLITS,
    num_sms: int | None = None,
) -> int:
    """Heuristic for decode split-KV.

    Decode means one query token per sequence. max_seq_len is the static
    KV/context length bound used to choose one split count for the whole launch.
    """
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        ).multi_processor_count

    compute_block_size = _choose_compute_block_size(block_size)

    if head_size <= 64 and max_seq_len < 4096:
        return 1

    # Match FlashAttention's 128-thread split-KV occupancy model. The model is
    # based on compute tiles, not physical cache pages; a 528-token page should
    # behave like 33 smaller 16-token tiles for split-KV scheduling.
    return _choose_decode_num_splits(
        batch_size,
        num_kv_heads,
        max_seq_len,
        compute_block_size,
        max_num_splits,
        num_sms,
    )


def paged_attention_2d_splitkv_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    output: torch.Tensor | None = None,
    actual_max_splits: int | None = None,
    max_seq_len: int | None = None,
    mid_out: torch.Tensor | None = None,
    mid_lse: torch.Tensor | None = None,
    max_num_splits: int = _MAX_SPLITS,
    query_start_loc: torch.Tensor | None = None,
    filter_by_query_len: bool = False,
) -> torch.Tensor:
    """Decode-only paged attention launcher with split-KV.

    query is one token per sequence: [batch, num_query_heads, head_size].
    seq_lens is the current KV/context length per sequence, not query length.
    """
    if output is None:
        output = torch.empty_like(query)

    batch_size = seq_lens.shape[0] if filter_by_query_len else query.shape[0]
    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    physical_block_size = key_cache.shape[3]
    block_size = _choose_compute_block_size(physical_block_size)
    if block_size != 32:
        logger.warning_once(
            f"Chosen block size {block_size} may lead to suboptimal performance. "
            f"Consider using block size = 32 if possible."
        )
    x = key_cache.shape[4]
    num_queries_per_kv = num_query_heads // num_kv_heads
    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)
    head_size_padded = triton.next_power_of_2(head_size)

    if max_seq_len is None:
        max_seq_len = block_tables.shape[1] * physical_block_size

    if actual_max_splits is None:
        actual_max_splits = _get_num_splits(
            batch_size,
            num_kv_heads,
            head_size,
            block_size,
            max_seq_len,
            max_num_splits,
        )

    if actual_max_splits > max_num_splits:
        raise ValueError(
            f"actual_max_splits ({actual_max_splits}) must be <= "
            f"max_num_splits ({max_num_splits})."
        )

    if actual_max_splits == 1:
        kernel_paged_attention_2d[(batch_size, num_kv_heads)](
            output_ptr=output,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            sink_ptr=None,
            block_tables_ptr=block_tables,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=None,
            scale=scale,
            k_scale=1.0,
            v_scale=1.0,
            out_scale_inv=1.0,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=block_tables.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            BLOCK_SIZE=block_size,
            PHYSICAL_BLOCK_SIZE=physical_block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            USE_ALIBI_SLOPES=False,
            SLIDING_WINDOW=0,
            x=x,
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_k_cache_4=key_cache.stride(4),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            filter_by_query_len=filter_by_query_len,
            query_start_len_ptr=query_start_loc,
            USE_SINKS=False,
            USE_FP8=False,
        )
        return output

    # In practice the intermediate buffers should be pre-allocated,
    # however the calls to this function is deeply coupled with CUDA FlashAttention,
    # so we allocate them here for simplicity.
    if mid_out is None:
        mid_out = torch.empty(
            (batch_size, num_query_heads, actual_max_splits, head_size),
            device=query.device,
            dtype=torch.float32,
        )
    if mid_lse is None:
        mid_lse = torch.empty(
            (batch_size, num_query_heads, actual_max_splits),
            device=query.device,
            dtype=torch.float32,
        )

    kernel_paged_attention_2d_splitkv[(batch_size, num_kv_heads, actual_max_splits)](
        mid_out,
        mid_lse,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=num_queries_per_kv_padded,
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        mid_out_stride_0=mid_out.stride(0),
        mid_out_stride_1=mid_out.stride(1),
        mid_out_stride_2=mid_out.stride(2),
        mid_lse_stride_0=mid_lse.stride(0),
        mid_lse_stride_1=mid_lse.stride(1),
        mid_lse_stride_2=mid_lse.stride(2),
        BLOCK_SIZE=block_size,
        PHYSICAL_BLOCK_SIZE=physical_block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        x=x,
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=filter_by_query_len,
        query_start_len_ptr=query_start_loc,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    kernel_paged_attention_2d_splitkv_reduce[(batch_size, num_query_heads)](
        output,
        mid_out,
        mid_lse,
        seq_lens,
        output.stride(0),
        output.stride(1),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        MAX_NUM_SPLITS=actual_max_splits,
        filter_by_query_len=filter_by_query_len,
        query_start_len_ptr=query_start_loc,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    return output
