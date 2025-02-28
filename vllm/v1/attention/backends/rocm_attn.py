# SPDX-License-Identifier: Apache-2.0
"""Attention layer with PagedAttention on rocm"""
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.ops.prefix_prefill import context_attention_fwd
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata, FlashAttentionMetadataBuilder

logger = init_logger(__name__)

import os
import torch
import triton
import triton.language as tl

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


debug_flag = False

@triton.jit
def kernel_paged_attention_2d(
    output_ptr,  # [num_seqs, num_query_heads, head_size]
    query_ptr,  # [num_seqs, num_query_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    cu_q_len_ptr, # [num_seqs+1]
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    cache_block_stride: tl.constexpr,  # int
    block_table_stride: tl.constexpr,  # int, should be equal to max_num_blocks_per_seq
    query_stride_0: tl.constexpr,  # int
    query_stride_1: tl.constexpr,  # int, should be equal to head_size
    output_stride_0: tl.constexpr,  # int
    output_stride_1: tl.constexpr,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    x: tl.constexpr,
    stride_k_cache_0: tl.constexpr,
    stride_k_cache_1: tl.constexpr,
    stride_k_cache_2: tl.constexpr,
    stride_k_cache_3: tl.constexpr,
    stride_k_cache_4: tl.constexpr,
    stride_v_cache_0: tl.constexpr,
    stride_v_cache_1: tl.constexpr,
    stride_v_cache_2: tl.constexpr,
    stride_v_cache_3: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    kv_head_idx = query_head_idx // num_queries_per_kv

    cur_batch_in_all_start_index = tl.load(cu_q_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(cu_q_len_ptr + seq_idx + 1)
    cur_batch_query_len = (cur_batch_in_all_stop_index -
                           cur_batch_in_all_start_index)

    if cur_batch_query_len > 1:
        return

    query_offset = seq_idx * query_stride_0 + query_head_idx * query_stride_1

    # Q : (HEAD_SIZE,)
    Q = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))

    block_table_offset = seq_idx * block_table_stride

    m = tl.full([1], float("-inf"), dtype=tl.float32)
    l = tl.full([1], 1.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)

    # context len for this particualr sequence
    context_len = tl.load(context_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx)

    num_blocks = cdiv_fn(context_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE)

        v_offset = (physical_block_idx * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_1 +
                    offs_d[:, None] * stride_v_cache_2 +
                    offs_n[None, :] * stride_v_cache_3)

        k_offset = (physical_block_idx * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_1 +
                    (offs_d[:, None] // x) * stride_k_cache_2 +
                    offs_n[None, :] * stride_k_cache_3 +
                    (offs_d[:, None] % x) * stride_k_cache_4)

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K = tl.load(key_cache_ptr + k_offset)

        # V : (HEAD_SIZE, BLOCK_SIZE)
        V = tl.load(value_cache_ptr + v_offset)

        tmp = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], context_len, dtype=tl.int32)
        mask_new = tmp < boundary
        # S : (BLOCK_SIZE,)
        S = tl.where(mask_new, 0.0, float("-inf")).to(tl.float32)
        S += scale * tl.sum(K * Q[:, None], axis=0)

        if USE_ALIBI_SLOPES:
            S += alibi_slope * (tmp - context_len + 1)

        # compute running maximum
        # m_j : (1,)
        m_j = tl.maximum(m, tl.max(S, axis=0))

        # P : (BLOCK_SIZE,)
        P = tl.exp(S - m_j)

        # l_j : (1,)
        l_j = tl.sum(P, axis=0)

        # alpha : (1, )
        alpha = tl.exp(m - m_j)

        # acc : (BLOCK_SIZE,)
        acc = acc * alpha

        # update constants
        l = l * alpha + l_j
        m = m_j

        # acc : (BLOCK_SIZE,)
        acc += tl.sum(V * P[None, :], axis=1)

    # epilogue
    acc = acc / l

    output_offset = seq_idx * output_stride_0 + query_head_idx * output_stride_1

    tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE), acc)


def paged_attention_triton_2d(
    output,
    query,
    key_cache,
    value_cache,
    scale,
    block_tables,
    context_lens,
    alibi_slopes,
    block_size,
    num_seqs,
    num_query_heads,
    num_queries_per_kv,
    head_size,
    cu_q_len,
):
    use_alibi_slopes = alibi_slopes is not None

    #if len(key_cache.shape) == 5 and key_cache.shape[4] != 1:
    #    raise RuntimeError("5d kv cache not supported")

    if debug_flag and not torch.cuda.is_current_stream_capturing():
        torch.set_printoptions(threshold=10_000)
        print("\nnum_seqs: ", num_seqs)
        print("query shape: ", query.shape)
        print("num query heads: ", num_query_heads)
        print("context_lens: ", context_lens)
        print("block_tables.shape: ", block_tables.shape)
        print("key_cache.shape: ", key_cache.shape)
        print("value_cache.shape: ", value_cache.shape)
        print(block_tables)
        print("query strides: ", query.stride(0), query.stride(1), query.stride(2))
        print("block_tables strides: ", block_tables.stride(0), block_tables.stride(1))
        print(
            "key_cache strides: ",
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
        )
        print("output strides: ", output.stride(0), output.stride(1), output.stride(2))
        print(
            "value_cache strides: ",
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
        )
        print("context_lens stride: ", context_lens.stride(0))
        if alibi_slopes is not None:
            print("alibi_slobes stride: ", alibi_slopes.stride(0))

    kernel_paged_attention_2d[
        (
            num_seqs,
            num_query_heads,
        )
    ](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        context_lens_ptr=context_lens,
        alibi_slopes_ptr=alibi_slopes,
        scale=scale,
        cu_q_len_ptr=cu_q_len,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        cache_block_stride=key_cache.stride(0),
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        USE_ALIBI_SLOPES=use_alibi_slopes,
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
    )


class ROCmAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "ROCM_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> Type["ROCmAttentionImpl"]:
        return ROCmAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


class ROCmAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "ROCmAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = ROCmAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by ROCmAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "ROCmAttentionImpl")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size)

        # Reshape the input keys and values and store them in the cache.
        PagedAttention.write_to_paged_cache(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        num_queries_per_kv = (query.shape[1] // key.shape[1])

        '''
        print("num_actual_tokens:  ", num_actual_tokens)
        print("query.shape:        ", query.shape)
        print("key.shape:          ", key.shape)
        print("value.shape:        ", value.shape)
        print("output.shape:       ", output.shape)
        print("key_cache.shape:    ", key_cache.shape)
        print("value_cache.shape:  ", value_cache.shape)
        print("query_start_loc:    ", attn_metadata.query_start_loc)
        print("seq_lens:           ", attn_metadata.seq_lens)
        print("num_seqs:           ", len(attn_metadata.seq_lens))
        print("num_queries_per_kv: ", num_queries_per_kv)
        print("block_table.shape:  ", attn_metadata.block_table.shape)
        print("block_table.stride: ", attn_metadata.block_table.stride())
        print("output.stride:      ", output.stride())
        print("seq_lens.stride:    ", attn_metadata.seq_lens.stride())
        print("alibi_slopes:       ", self.alibi_slopes)
        print("sliding_window:     ", self.sliding_window[0])
        '''

        # Compute attention and update output up to `num_actual_tokens`.
        context_attention_fwd(q=query[:num_actual_tokens],
                              k=key[:num_actual_tokens],
                              v=value[:num_actual_tokens],
                              o=output[:num_actual_tokens],
                              kv_cache_dtype=self.kv_cache_dtype,
                              k_cache=key_cache,
                              v_cache=value_cache,
                              b_loc=attn_metadata.block_table,
                              b_start_loc=attn_metadata.query_start_loc,
                              b_seq_len=attn_metadata.seq_lens,
                              max_input_len=attn_metadata.max_query_len,
                              k_scale=layer._k_scale,
                              v_scale=layer._v_scale,
                              alibi_slopes=self.alibi_slopes,
                              sliding_window=self.sliding_window[0],
                              sm_scale=self.scale)

        paged_attention_triton_2d(
            output=output[:num_actual_tokens],
            query=query[:num_actual_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            scale=self.scale,
            cu_q_len=attn_metadata.query_start_loc,
            block_tables=attn_metadata.block_table,
            context_lens=attn_metadata.seq_lens,
            alibi_slopes=self.alibi_slopes,
            block_size=16,
            num_seqs=len(attn_metadata.seq_lens),
            num_query_heads=query.shape[1],
            num_queries_per_kv=num_queries_per_kv,
            head_size=query.shape[2]
        )

        return output
