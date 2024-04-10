"""Attention layer with Pallas FlashAttention and PagedAttention."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch_xla.experimental.custom_kernel import flash_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)


class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["PallasAttentionImpl"]:
        return PallasAttentionImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "PallasAttentionMetadata":
        return PallasAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_kv_heads, num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        raise NotImplementedError(
            "Swapping blocks is not supported on TPU backend.")

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        raise NotImplementedError(
            "Copying blocks is not supported on TPU backend.")


@dataclass
class PallasAttentionMetadata(AttentionMetadata):
    """Metadata for PallasAttentionBackend."""
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor


class PallasAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        if sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is not supported on TPU backend.")
        if alibi_slopes is not None:
            raise NotImplementedError(
                "Alibi slopes are not supported on TPU backend.")

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        # TODO(woosuk): Check supported head sizes.

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: PallasAttentionMetadata,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache = [2, num_kv_heads, num_blocks, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        value = value.view(batch_size, seq_len, self.num_kv_heads,
                           self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = kv_cache[0], kv_cache[1]

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            key_cache.index_copy_(dim=2,
                                  index=attn_metadata.slot_mapping,
                                  source=key)
            value_cache.index_copy_(dim=2,
                                    index=attn_metadata.slot_mapping,
                                    source=value)

        if attn_metadata.is_prompt:
            # Prompt run.
            if kv_cache is None or attn_metadata.block_tables.numel() == 0:
                # normal attention
                output = flash_attention(
                    query.permute(0, 2, 1, 3),
                    key.permute(0, 2, 1, 3),
                    value.permute(0, 2, 1, 3),
                    causal=True,
                )
                output = output.permute(0, 2, 1, 3)
                output = output.reshape(batch_size, seq_len, hidden_size)
            else:
                # prefix-enabled attention
                raise NotImplementedError(
                    "Prefix-enabled attention is not supported on TPU backend."
                )
        else:
            # Decoding run.
            pass

        return output
