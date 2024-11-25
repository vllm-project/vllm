"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.forward_context import get_forward_context
from vllm.utils import direct_register_custom_op
from vllm.vllm_flash_attn import flash_attn_varlen_func


class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

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


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_start_loc: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class FlashAttentionImpl(AttentionImpl):

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
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
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
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = FlashAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashAttentionImpl")

        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in FlashAttention.")

        output = torch.empty_like(query)
        torch.ops.vllm.unified_v1_flash_attention(
            output,
            query,
            key,
            value,
            self.num_heads,
            self.head_size,
            self.num_kv_heads,
            kv_cache,
            self.kv_cache_dtype,
            k_scale,
            v_scale,
            self.scale,
            self.sliding_window,
            self.alibi_slopes,
            self.logits_soft_cap,
        )
        return output


def unified_v1_flash_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    softmax_scale: float,
    window_size: Optional[List[int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    logits_soft_cap: Optional[float] = None,
) -> None:
    context = get_forward_context()
    current_metadata = context.dynamic_forward_context
    if current_metadata is None:
        # Profiling run.
        return

    assert current_metadata is not None
    assert isinstance(current_metadata, FlashAttentionMetadata)
    attn_metadata: FlashAttentionMetadata = current_metadata
    num_actual_tokens = attn_metadata.num_actual_tokens

    # Reshape the query, key, and value tensors.
    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)

    # Reshape the input keys and values and store them in the cache.
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key[:num_actual_tokens],
        value[:num_actual_tokens],
        key_cache,
        value_cache,
        attn_metadata.slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    attn_output = flash_attn_varlen_func(
        q=query[:num_actual_tokens],
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=attn_metadata.query_start_loc,
        max_seqlen_q=attn_metadata.max_query_len,
        cu_seqlens_k=attn_metadata.seq_start_loc,
        max_seqlen_k=attn_metadata.max_seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        alibi_slopes=alibi_slopes,
        window_size=window_size,
        block_table=attn_metadata.block_table,
        softcap=logits_soft_cap,
    )
    attn_output = attn_output.view(num_actual_tokens, -1)
    # TODO(woosuk): Optimize this.
    output[:num_actual_tokens].copy_(attn_output)


def unified_v1_flash_attention_fake(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    softmax_scale: float,
    window_size: Optional[List[int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    logits_soft_cap: Optional[float] = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_v1_flash_attention",
    op_func=unified_v1_flash_attention,
    mutates_args=["kv_cache", "output"],
    fake_impl=unified_v1_flash_attention_fake,
)
