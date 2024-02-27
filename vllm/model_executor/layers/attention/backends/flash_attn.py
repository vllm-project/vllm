"""Attention layer with Flash and PagedAttention."""
from typing import List, Optional

from flash_attn import flash_attn_func, flash_attn_with_kvcache
import torch

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention.ops.paged_attn import (
    PagedAttentionImpl)


class FlashAttentionBackend:

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
        suppored_head_sizes = PagedAttentionImpl.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        self.sliding_window = ((self.sliding_window, self.sliding_window) if
                               self.sliding_window is not None else (-1, -1))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
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

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            # Update kv-cache using tensor indexing. We don't use the kernel
            # `flash_attn_with_kvcache` for kv-cache updating as it submitted
            # many small kernels for each key/value and is slow.
            flatten_slot_mapping = input_metadata.slot_mapping.flatten()
            slot_block_index = flatten_slot_mapping // key_cache.shape[1]
            slot_block_offset = flatten_slot_mapping % key_cache.shape[1]
            key_cache[slot_block_index, slot_block_offset, :, :] = key
            value_cache[slot_block_index, slot_block_offset, :, :] = value

        if input_metadata.is_prompt:
            # normal attention
            query = query.unflatten(0, (batch_size, seq_len))
            key = key.unflatten(0, (batch_size, seq_len))
            value = value.unflatten(0, (batch_size, seq_len))
            if (key_cache is None or value_cache is None
                    or not input_metadata.context_lens.any()):
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
                output = flash_attn_with_kvcache(
                    q=query,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    cache_seqlens=input_metadata.context_lens + seq_len,
                    block_table=input_metadata.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
        else:
            # Decoding run.

            # NOTE: in `_prepare_prompt` and `_prepare_decode` is filled in
            # different manner (which may needs to be fixed in the future): in
            # the former `context_lens` is the length of contexts whose kv-cache
            # has been stored in previous rounds, (e.g., with prefix cache).
            # However, in the later `context_lens` is the length of current
            # attention context (includes the token whose kv-cache will be
            # computed and filled in this round).
            #
            # - The kernel `flash_attn_with_kvcache` expects `cache_seqlens` to
            #   be the length of the context whose kv-cache has been stored.
            # - The kernel `context_attention_fwd` expects it to be the length
            #   of already computed query-key-values in previous rounds.
            # - The kernel `paged_attention_v1/v2` expect it to be the length of
            #   current attention context., same as flash-attn (without k & v).
            #
            # The flash-attn kernel can also be used with the k/v in current
            # round as argument as they will be stored into the key-value cache
            # inside the kernel. In which case, the `cache_seqlens` is expected
            # to be the context length of tokens whose k/v has already been
            # stored into kv-cache. However, it is found inefficient (especially
            # for prompting) due to too many calls of cudaMemcpy kernels.

            # see also: https://github.com/Dao-AILab/flash-attention/commit/54e80a3829c6d2337570d01e78ebd9529c02d342
            output = flash_attn_with_kvcache(
                q=query.reshape(batch_size, -1, *query.shape[1:]),
                k_cache=key_cache,
                v_cache=value_cache,
                cache_seqlens=input_metadata.context_lens,
                block_table=input_metadata.block_tables,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)
