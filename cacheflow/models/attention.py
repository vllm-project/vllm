from typing import List, Optional

from flash_attn.flash_attention import FlashAttention
import torch
import torch.nn as nn

from cacheflow import attention_ops
from cacheflow import cache_ops
from cacheflow.models import InputMetadata


class OPTCacheFlowAttention(nn.Module):

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

        self.flash_attn = FlashAttention(softmax_scale=self.scale)

    def _masked_attention(
        self,
        query: torch.Tensor,                        # [num_queries, num_heads, head_size]
        key: torch.Tensor,                          # [num_keys, num_heads, head_size]
        value: torch.Tensor,                        # [num_keys, num_heads, head_size]
        attn_mask: Optional[torch.Tensor] = None,   # [num_queries, num_keys]
    ) -> torch.Tensor:                              # [num_queries, num_heads, head_size]
        query = query * self.scale
        attn = torch.einsum('qhd,khd->hqk', query, key)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('hqk,khd->qhd', attn, value)
        return out

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,       # [num_prompt_tokens, num_heads, head_size]
        query: torch.Tensor,        # [num_prompt_tokens, num_heads, head_size]
        key: torch.Tensor,          # [num_prompt_tokens, num_heads, head_size]
        value: torch.Tensor,        # [num_prompt_tokens, num_heads, head_size]
        prompt_lens: List[int],
    ) -> None:
        device = query.device
        prefix_sum = [0]
        for prompt_len in prompt_lens:
            prefix_sum.append(prefix_sum[-1] + prompt_len)
        prefix_sum = torch.tensor(prefix_sum, dtype=torch.int, device=device)
        max_prompt_len = max(prompt_lens)

        # FIXME(woosuk): Unnecessary copy. Optimize this.
        qkv = torch.stack([query, key, value], dim=1)
        out = self.flash_attn(
            qkv,
            cu_seqlens=prefix_sum,
            max_s=max_prompt_len,
            causal=True,
        )[0]
        num_tokens = prefix_sum[-1]
        # FIXME(woosuk): Unnecessary copy. Optimize this.
        output[:num_tokens].copy_(out, non_blocking=True)

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,           # [num_generation_tokens, num_heads, head_size]
        query: torch.Tensor,            # [num_generation_tokens, num_heads, head_size]
        key_cache: torch.Tensor,        # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: torch.Tensor,      # [num_blocks, num_heads, head_size, block_size]
        input_metadata: InputMetadata,
    ) -> None:
        block_size = value_cache.shape[3]
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
        )

    def forward(
        self,
        query: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key: torch.Tensor,                      # [num_tokens, num_heads * head_size]
        value: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key_cache: torch.Tensor,                # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: torch.Tensor,              # [num_blocks, num_heads, head_size, block_size]
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:                          # [num_tokens, num_heads * head_size]
        # Pre-allocate the output tensor.
        output = torch.empty_like(query)

        # Prune out paddings if any.
        query = query[:input_metadata.num_valid_tokens]
        key = key[:input_metadata.num_valid_tokens]
        value = value[:input_metadata.num_valid_tokens]

        # Reshape the input tensors.
        num_heads = value_cache.shape[1]
        head_size = value_cache.shape[2]
        query = query.view(-1, num_heads, head_size)
        key = key.view(-1, num_heads, head_size)
        value = value.view(-1, num_heads, head_size)
        output = output.view(-1, num_heads, head_size)

        # Compute the attention op for prompts.
        if input_metadata.num_prompts > 0:
            self.multi_query_kv_attention(
                output, query, key, value, input_metadata.prompt_lens)

        # Wait until the cache op is done.
        if cache_event is not None:
            cache_event.wait()

        # Reshape the keys and values and store them in the cache.
        cache_ops.reshape_and_cache(
            key, value, key_cache, value_cache, input_metadata.slot_mapping)

        if input_metadata.num_generation_tokens > 0:
            # Compute the attention op for generation tokens.
            start_idx = sum(input_metadata.prompt_lens)
            self.single_query_cached_kv_attention(
                output[start_idx:],
                query[start_idx:],
                key_cache,
                value_cache,
                input_metadata)

        # Reshape the output tensor.
        # NOTE(woosuk): The output tensor may include paddings.
        return output.view(-1, num_heads * head_size)
