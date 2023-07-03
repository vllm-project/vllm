"""Multi-head attention."""
from typing import List, Optional

import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm import attention_ops
from vllm import cache_ops
from vllm import pos_encoding_ops
from vllm.model_executor.input_metadata import InputMetadata

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128]


class PagedAttention(nn.Module):
    # pylint: disable=line-too-long
    """GPT-style multi-head PagedAttention.

    This class takes flattened 1D query, key, and value tensors as input. The
    input 1D tensors can be split into three parts: the prompt tokens, the
    generation tokens, and the paddings.

    |<------------------------------------- num_valid_tokens ------------------------------------->|
    |<--------------- num_prompt_tokens -------------->|<------- num_generation_tokens (M) ------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|<--generation_0-->|...|<--generation_M-1-->|<--padding-->|

    The prompts might have different lengths, while the generation tokens always
    have length 1. The paddings are appended to make the input length a multiple
    of 8, which is desirable for Tensor Cores.

    The class does the following:
    1. Perform multi_query_kv_attention for the prompts. This operation does
        not use the KV cache.
    2. Wait for the cache operations (e.g., swap, copy) to finish. The cache
        operations are issued by the cache engine before executing the forward
        pass of the model, and they are executed asynchronously.
    3. Reshape and store the input key and value tensors in the KV cache.
    4. Perform single_query_cached_kv_attention for the generation tokens.
        This operation reads the previous key and value tensors from the KV
        cache.
    5. Output a flattened 1D tensor.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.attn_op = xops.fmha.cutlass.FwOp()

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def set_attn_bias(self, input_metadata: InputMetadata) -> None:
        if input_metadata.attn_bias:
            # Already set by a previous layer.
            return
        prompt_lens = input_metadata.prompt_lens
        attn_bias = BlockDiagonalCausalMask.from_seqlens(prompt_lens)
        input_metadata.attn_bias.append(attn_bias)

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Normal attention for the prompt tokens.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_heads, head_size]
            value: shape = [num_prompt_tokens, num_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        # TODO(woosuk): The unsqueeze op may incur some CPU overhead. Optimize.
        out = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=input_metadata.attn_bias[0],
            p=0.0,
            scale=self.scale,
            op=self.attn_op,
        )
        # TODO(woosuk): Unnecessary copy. Optimize.
        output.copy_(out.squeeze(0))
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        """PagedAttention for the generation tokens.

        Args:
            output: shape = [num_generation_tokens, num_heads, head_size]
            query: shape = [num_generation_tokens, num_heads, head_size]
            key_cache: shape = [num_blocks, num_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_heads, head_size, block_size]
            input_metadata: metadata for paged attention.
        """
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
            None,  # alibi_slopes
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        NOTE: The query, key, and value tensors must be sliced from a qkv
        tensor of shape [num_tokens, 3 * num_heads * head_size].

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_heads * head_size]
            value: shape = [num_tokens, num_heads * head_size]
            key_cache: shape = [num_blocks, num_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_heads, head_size, block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_heads, self.head_size)
        value = value.view(-1, self.num_heads, self.head_size)

        # Pre-allocate the output tensor.
        output = torch.empty_like(query)

        # Compute the attention op for prompts.
        num_prompt_tokens = input_metadata.num_prompt_tokens
        if num_prompt_tokens > 0:
            self.set_attn_bias(input_metadata)
            self.multi_query_kv_attention(
                output[:num_prompt_tokens],
                query[:num_prompt_tokens],
                key[:num_prompt_tokens],
                value[:num_prompt_tokens],
                input_metadata,
            )

        # Wait until the cache op is done.
        if cache_event is not None:
            cache_event.wait()

        # Reshape the keys and values and store them in the cache.
        # When key_cache and value_cache are not provided, the new key
        # and value vectors will not be cached.
        num_valid_tokens = input_metadata.num_valid_tokens
        if (num_valid_tokens > 0 and key_cache is not None
                and value_cache is not None):
            # The stride is 3 because the key and value are sliced from qkv.
            cache_ops.reshape_and_cache(
                key[:num_valid_tokens],
                value[:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata.slot_mapping,
            )

        if input_metadata.num_generation_tokens > 0:
            assert key_cache is not None and value_cache is not None, (
                "key_cache and value_cache must be provided when "
                "generating tokens.")
            # Compute the attention op for generation tokens.
            self.single_query_cached_kv_attention(
                output[num_prompt_tokens:num_valid_tokens],
                query[num_prompt_tokens:num_valid_tokens], key_cache,
                value_cache, input_metadata)

        # Reshape the output tensor.
        # NOTE(woosuk): The output tensor may include paddings.
        return output.view(-1, self.num_heads * self.head_size)


class PagedAttentionWithRoPE(PagedAttention):
    """PagedAttention with GPT-NeoX style rotary embedding."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
    ) -> None:
        super().__init__(num_heads, head_size, scale)

        # Create the cos and sin cache.
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum("i,j -> ij", t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        # FIXME(woosuk): This assumes that we configure the default dtype when
        # initializing the model.
        # TODO(woosuk): Make it more robust.
        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        # Embedding size: [max_position, rotary_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """ PagedAttention forward pass with rotary embedding.

        Args:
            positions: shape = [num_tokens]
                        query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_heads * head_size]
            value: shape = [num_tokens, num_heads * head_size]
            key_cache: shape = [num_blocks, num_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_heads, head_size, block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        pos_encoding_ops.rotary_embedding_neox(
            positions,
            query,
            key,
            self.head_size,
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


class PagedAttentionWithALiBi(PagedAttention):
    """PagedAttention with ALiBi attention bias."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        slopes: List[float],
    ) -> None:
        super().__init__(num_heads, head_size, scale)
        assert len(slopes) == num_heads

        slopes = torch.tensor(slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def set_attn_bias(self, input_metadata: InputMetadata) -> None:
        if input_metadata.attn_bias:
            # Already set by a previous layer.
            return
        # Generates ALiBi mask for each prompt.
        for prompt_len in input_metadata.prompt_lens:
            bias = torch.arange(prompt_len)
            bias = bias[None, :] - bias[:, None]
            bias = bias.to(self.alibi_slopes.device)

            # When using custom attention bias, xformers requires the bias to
            # be sliced from a tensor whose length is a multiple of 8.
            padded_len = (prompt_len + 7) // 8 * 8
            bias = torch.empty(
                self.num_heads,
                padded_len,
                padded_len,
                device=self.alibi_slopes.device,
            )[:, :prompt_len, :prompt_len].copy_(bias)
            bias.mul_(self.alibi_slopes[:, None, None])
            attn_bias = LowerTriangularMaskWithTensorBias(bias)
            input_metadata.attn_bias.append(attn_bias)

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Attention with ALiBi bias for the prompt tokens.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_heads, head_size]
            value: shape = [num_prompt_tokens, num_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        start = 0
        for i, prompt_len in enumerate(input_metadata.prompt_lens):
            end = start + prompt_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=input_metadata.attn_bias[i],
                p=0.0,
                scale=self.scale,
                op=self.attn_op,
            )
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.squeeze(0))
            start += prompt_len
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        """PagedAttention with ALiBi bias for the generation tokens.

        Args:
            output: shape = [num_generation_tokens, num_heads, head_size]
            query: shape = [num_generation_tokens, num_heads, head_size]
            key_cache: shape = [num_blocks, num_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_heads, head_size, block_size]
            input_metadata: metadata for paged attention.
        """
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
            self.alibi_slopes,
        )
