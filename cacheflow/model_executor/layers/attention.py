from typing import Optional

import torch
import torch.nn as nn
from xformers import ops as xops

from cacheflow import attention_ops
from cacheflow import cache_ops
from cacheflow import pos_encoding_ops
from cacheflow.model_executor.input_metadata import InputMetadata


class GPTCacheFlowAttention(nn.Module):

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.attn_op = xops.fmha.cutlass.FwOp()

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,                   # [num_prompt_tokens, num_heads, head_size]
        query: torch.Tensor,                    # [num_prompt_tokens, num_heads, head_size]
        key: torch.Tensor,                      # [num_prompt_tokens, num_heads, head_size]
        value: torch.Tensor,                    # [num_prompt_tokens, num_heads, head_size]
        attn_bias: xops.AttentionBias,
    ) -> None:
        # TODO(woosuk): The unsqueeze op may incur some CPU overhead. Optimize.
        out = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=attn_bias,
            p=0.0,
            scale=self.scale,
            op=self.attn_op,
        )
        # TODO(woosuk): Unnecessary copy. Optimize.
        output.copy_(out.squeeze(0))
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,           # [num_generation_tokens, num_heads, head_size]
        query: torch.Tensor,            # [num_generation_tokens, num_heads, head_size]
        key_cache: torch.Tensor,        # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: torch.Tensor,      # [num_blocks, num_heads, head_size, block_size]
        input_metadata: InputMetadata,
    ) -> None:
        head_size = value_cache.shape[2]
        supported_head_sizes = [32, 64, 80, 96, 128, 160, 192, 256]
        if head_size not in supported_head_sizes:
            raise ValueError(f'head_size ({head_size}) is not supported by '
                             'the single_query_cached_kv_attention kernel. '
                             'Use one of the following head sizes: '
                             f'{supported_head_sizes}.')

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
        # NOTE: The query, key, and value tensors must be sliced from a qkv
        # tensor of shape [num_tokens, 3 * num_heads * head_size].

        # Reshape the query, key, and value tensors.
        num_heads = value_cache.shape[1]
        head_size = value_cache.shape[2]
        query = query.view(-1, num_heads, head_size)
        key = key.view(-1, num_heads, head_size)
        value = value.view(-1, num_heads, head_size)

        # Pre-allocate the output tensor.
        output = torch.empty_like(query)

        # Compute the attention op for prompts.
        num_prompt_tokens = input_metadata.num_prompt_tokens
        if num_prompt_tokens > 0:
            self.multi_query_kv_attention(
                output[:num_prompt_tokens],
                query[:num_prompt_tokens],
                key[:num_prompt_tokens],
                value[:num_prompt_tokens],
                input_metadata.attn_bias,
            )

        # Wait until the cache op is done.
        if cache_event is not None:
            cache_event.wait()

        # Reshape the keys and values and store them in the cache.
        num_valid_tokens = input_metadata.num_valid_tokens
        if num_valid_tokens > 0:
            # The stride is 3 because the key and value are sliced from qkv.
            cache_ops.reshape_and_cache(
                key[:num_valid_tokens],
                value[:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata.slot_mapping,
            )

        if input_metadata.num_generation_tokens > 0:
            # Compute the attention op for generation tokens.
            self.single_query_cached_kv_attention(
                output[num_prompt_tokens:num_valid_tokens],
                query[num_prompt_tokens:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata)

        # Reshape the output tensor.
        # NOTE(woosuk): The output tensor may include paddings.
        return output.view(-1, num_heads * head_size)


class GPTNeoXCacheFlowAttention(GPTCacheFlowAttention):
    """Attention with GPT-NeoX style rotary embedding."""

    def __init__(
        self,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
    ) -> None:
        super().__init__(scale)

        # Create the cos and sin cache.
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j -> ij', t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        # FIXME(woosuk): This assumes that we configure the default dtype when
        # initializing the model. Make it more robust.
        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        # Embedding size: [max_position, rotary_dim]
        self.register_buffer('cos_sin_cache', cache, persistent=False)

    def forward(
        self,
        positions: torch.LongTensor,            # [num_tokens]
        query: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key: torch.Tensor,                      # [num_tokens, num_heads * head_size]
        value: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key_cache: torch.Tensor,                # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: torch.Tensor,              # [num_blocks, num_heads, head_size, block_size]
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:                          # [num_tokens, num_heads * head_size]
        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        head_size = value_cache.shape[2]
        pos_encoding_ops.rotary_embedding_neox(
            positions,
            query,
            key,
            head_size,
            self.cos_sin_cache,
        )
        return super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )
