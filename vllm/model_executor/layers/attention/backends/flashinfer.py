"""Attention layer with Flash and PagedAttention."""
from typing import List, Optional

# NOTE(woosuk): This imports flash_attn under vllm/thirdparty_files/.
from flash_attn import flash_attn_func
import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention.ops.flash_attn import (
    FlashAttentionImpl)
from vllm.block import KVCache

class FlashInferBackend:
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
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        suppored_head_sizes = [32, 64, 128, 256]
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        self.sliding_window = (-1, -1)
        if sliding_window is not None:        
            raise RuntimeError("FlashInfer does not support sliding window attention")
        if alibi_slopes is not None:
            raise RuntimeError("FlashInfer does not support alibi slopes")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        decoder_wrapper: BatchDecodeWithPagedKVCacheWrapper = input_metadata.decoder_wrapper
        assert decoder_wrapper is not None, "attempt to call FlashInferBackend.forward without initializing decoder_wrapper"

        assert kv_cache is None or isinstance(kv_cache, torch.Tensor), "unsupported KV cache layout"

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if kv_cache is not None:
            FlashAttentionImpl.reshape_and_cache(key, value, kv_cache[:, 0],
                                                 kv_cache[:, 1], input_metadata)

        if input_metadata.is_prompt:
            # Prompt run.
            if (kv_cache is None
                    or input_metadata.block_tables.numel() == 0):
                # normal attention
                query = query.unflatten(0, (batch_size, seq_len))
                key = key.unflatten(0, (batch_size, seq_len))
                value = value.unflatten(0, (batch_size, seq_len))
                output = flash_attn_func(
                    query,
                    key,
                    value,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
            else:
                # prefix-enabled attention
                raise NotImplementedError
        else:
            # Decoding run.
            output = decoder_wrapper.forward(query.contiguous(), kv_cache)

        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)
