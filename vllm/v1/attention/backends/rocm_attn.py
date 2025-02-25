# SPDX-License-Identifier: Apache-2.0
"""Attention layer with PagedAttention on rocm"""
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.ops.prefix_prefill import context_attention_fwd
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

logger = init_logger(__name__)


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
        return output
