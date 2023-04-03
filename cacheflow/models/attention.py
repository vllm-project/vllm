from typing import Optional

from flash_attn.flash_attn_interface import _flash_attn_forward
import torch
import torch.nn as nn

from cacheflow import attention_ops
from cacheflow import cache_ops
from cacheflow import pos_encoding_ops
from cacheflow.models import InputMetadata


class GPTCacheFlowAttention(nn.Module):

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,                   # [num_prompt_tokens, num_heads, head_size]
        query: torch.Tensor,                    # [num_prompt_tokens, num_heads, head_size]
        key: torch.Tensor,                      # [num_prompt_tokens, num_heads, head_size]
        value: torch.Tensor,                    # [num_prompt_tokens, num_heads, head_size]
        cumulative_prompt_lens: torch.Tensor,   # [num_prompts + 1]
        max_prompt_len: int,
    ) -> None:
        if query.dtype == torch.float:
            raise ValueError('The float data type is not supported by '
                             'FlashAttention. Use the half data type instead.')
        head_size = query.shape[-1]
        if head_size > 128:
            raise ValueError('FlashAttention does not support head_size > 128.')

        # Directly call FlashAttention's internal function to avoid allocating
        # a new tensor for the output.
        _flash_attn_forward(
            query,
            key,
            value,
            output,
            cumulative_prompt_lens,
            cumulative_prompt_lens,
            max_prompt_len,
            max_prompt_len,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            return_softmax=False,
        )

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
                input_metadata.cumulative_prompt_lens,
                input_metadata.max_prompt_len,
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


class OPTCacheFlowAttention(GPTCacheFlowAttention):
    """OPT uses the same attention mechanism as GPT."""

    def __init__(self, scale: float) -> None:
        super().__init__(scale)


class LlamaCacheFlowAttention(GPTCacheFlowAttention):
    """Llama uses GPT-NeoX style rotary embedding."""

    def __init__(
        self,
        scale: float,
        head_size: int,
        max_position: int = 8192,
        base: int = 10000,
    ) -> None:
        super().__init__(scale)

        # Create the cos and sin cache.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2) / head_size))
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j -> ij', t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        # FIXME(woosuk): This assumes that we configure the default dtype when
        # initializing the model. Make it more robust.
        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        # Embedding size: [max_position, head_size]
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
        pos_encoding_ops.rotary_embedding_neox(
            positions,
            query,
            key,
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
