from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_xla.experimental.custom_kernel  # Required to register custom ops.
import torch_xla.experimental.dynamo_set_buffer_donor

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)


class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "PallasMetadata":
        return PallasMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_kv_heads, num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        raise NotImplementedError("swap_blocks is not implemented.")

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        # TODO(woosuk): Implement this.
        raise NotImplementedError("copy_blocks is not implemented.")


@dataclass
class PallasMetadata(AttentionMetadata):
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    block_tables: Optional[torch.Tensor]
    context_lens: Optional[torch.Tensor]


class PallasAttentionBackendImpl(AttentionImpl):

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

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not implemented.")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window is not implemented.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Any,
        attn_metadata: AttentionMetadata[PallasMetadata],  # type: ignore
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache = [num_kv_heads, num_blocks, block_size, head_size]
            value_cache = [num_kv_heads, num_blocks, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        assert kv_scale == 1.0
        batch_size, seq_len, hidden_size = query.shape
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        value = value.view(batch_size, seq_len, self.num_kv_heads,
                           self.head_size)

        if kv_cache is not None:
            slot_mapping = attn_metadata.slot_mapping
            key_cache, value_cache = kv_cache
            write_to_kv_cache(key, value, key_cache, value_cache, slot_mapping)

        query = query * self.scale
        if attn_metadata.is_prompt:
            # assert attn_metadata.block_tables is None
            assert seq_len % 16 == 0

            # Handle GQA/MQA.
            if self.num_kv_heads != self.num_heads:
                key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
                key = key.view(batch_size, seq_len, self.num_heads, self.head_size)
                value = value.repeat_interleave(self.num_queries_per_kv, dim=1)
                value = value.view(batch_size, seq_len, self.num_heads, self.head_size)
            output = torch.ops.xla.flash_attention(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                True,
            )
            output = output.permute(0, 2, 1, 3)
        else:
            # Decoding run.
            assert kv_cache is not None
            output = torch.ops.xla.paged_attention(
                query.squeeze(dim=1),
                key_cache,
                value_cache,
                attn_metadata.context_lens,
                attn_metadata.block_tables,
                16,  # pages_per_compute_block. TODO(woosuk): Tune this value.
                megacore_mode="kv_head",  # FIXME(woosuk): Must be None for TPUv5e.
            )

        # Reshape the output tensor.
        return output.reshape(batch_size, seq_len, hidden_size)


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)
    torch.ops.xla.dynamo_set_buffer_donor_(value_cache, True)

    key = key.flatten(0, 2)
    value = value.flatten(0, 2)
    key_cache = key_cache.flatten(0, 2)
    value_cache = value_cache.flatten(0, 2)
    key_cache.index_copy_(0, slot_mapping, key)
    value_cache.index_copy_(0, slot_mapping, value)
