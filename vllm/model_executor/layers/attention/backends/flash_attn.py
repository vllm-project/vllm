"""Attention layer with Flash and PagedAttention."""
from typing import List, Optional

from flash_attn import flash_attn_varlen_func
import torch

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention.ops.paged_attn import (
    PagedAttentionImpl)


class FlashAttentionBackend:
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prompt_tokens -------------->|	
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|

    Otherwise, the layout is as follows:	
    |<------------------ num_generation_tokens (M) ----------------->|	
    |<--generation_0-->|..........|<--generation_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.
    """

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
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            PagedAttentionImpl.reshape_and_cache(key, value, key_cache,
                                                 value_cache, input_metadata)

        if input_metadata.is_prompt:
            # Prompt run.
            if (key_cache is None or value_cache is None
                    or input_metadata.block_tables.numel() == 0):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                output = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=input_metadata.seq_start_loc,
                    cu_seqlens_k=input_metadata.seq_start_loc,
                    max_seqlen_q=input_metadata.max_seq_len,
                    max_seqlen_k=input_metadata.max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
            else:
                # prefix-enabled attention
                output = PagedAttentionImpl.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    input_metadata,
                    self.alibi_slopes,
                )
        else:
            # Decoding run.
            output = PagedAttentionImpl.forward_decode(
                query,
                key_cache,
                value_cache,
                input_metadata,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)
