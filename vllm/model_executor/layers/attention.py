"""Multi-head attention."""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm import attention_ops
from vllm import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.rotary_embedding import get_rope

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttention(nn.Module):
    # pylint: disable=line-too-long
    """GPT-style multi-head PagedAttention.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens, in addition to
    paddings.

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
    5. Return the output tensor.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 num_kv_heads: Optional[int] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_mapping = torch.repeat_interleave(
            torch.arange(self.num_kv_heads, dtype=torch.int32, device="cuda"),
            self.num_queries_per_kv)

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def set_attn_bias(
        self,
        input_metadata: InputMetadata,
        dtype: torch.dtype,
    ) -> None:
        del dtype  # Unused.
        if input_metadata.attn_bias is not None:
            # Already set by a previous layer.
            return
        prompt_lens = [input_metadata.max_prompt_len
                       ] * input_metadata.num_prompts
        attn_bias = BlockDiagonalCausalMask.from_seqlens(prompt_lens)
        if self.sliding_window is not None:
            attn_bias = attn_bias.make_local_attention(self.sliding_window)
        input_metadata.attn_bias = attn_bias

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
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        if self.num_kv_heads != self.num_heads:
            # Project the key and value tensors to the desired number of heads.
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=1)

        # TODO(woosuk): The unsqueeze op may incur some CPU overhead. Optimize.
        out = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=input_metadata.attn_bias,
            p=0.0,
            scale=self.scale,
        )
        # TODO(woosuk): Unnecessary copy. Optimize.
        output.copy_(out.squeeze(0))
        return output

    def get_alibi_slopes(self) -> Optional[torch.Tensor]:
        """Returns the slopes for the alibi attention bias.

        Returns:
            slopes: shape = [num_heads]
        """
        return None

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        alibi_slopes: Optional[torch.Tensor],
    ) -> None:
        """PagedAttention for the generation tokens.

        Args:
            output: shape = [num_generation_tokens, num_heads, head_size]
            query: shape = [num_generation_tokens, num_heads, head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            alibi_slopes: shape = [num_heads]
        """
        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = (
            (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
            _PARTITION_SIZE)
        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        use_v1 = input_metadata.max_context_len <= 8192 and (
            max_num_partitions == 1 or num_seqs * num_heads > 512)
        if use_v1:
            # Run PagedAttention V1.
            attention_ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                self.head_mapping,
                self.scale,
                input_metadata.block_tables,
                input_metadata.context_lens,
                block_size,
                input_metadata.max_context_len,
                alibi_slopes,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            attention_ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                self.head_mapping,
                self.scale,
                input_metadata.block_tables,
                input_metadata.context_lens,
                block_size,
                input_metadata.max_context_len,
                alibi_slopes,
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
        tensor of shape [batch_size, seq_len, 3 * num_heads * head_size].

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, _ = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Pre-allocate the output tensor.
        output = torch.empty_like(query)

        # Compute the attention op for prompts.
        num_prompt_tokens = input_metadata.num_prompt_tokens
        if num_prompt_tokens > 0:
            # Prompt run.
            assert input_metadata.num_generation_tokens == 0
            self.set_attn_bias(input_metadata, dtype=query.dtype)
            self.multi_query_kv_attention(
                output,
                query,
                key,
                value,
                input_metadata,
            )

        # Wait until the cache op is done.
        if cache_event is not None:
            cache_event.wait()

        # Reshape the keys and values and store them in the cache.
        # When key_cache and value_cache are not provided, the new key
        # and value vectors will not be cached.
        if key_cache is not None and value_cache is not None:
            key_to_cache = key
            value_to_cache = value
            slot_mapping = input_metadata.slot_mapping.view(-1)
            if input_metadata.to_cache is not None:
                key_to_cache = key_to_cache[input_metadata.to_cache]
                value_to_cache = value_to_cache[input_metadata.to_cache]
                slot_mapping = slot_mapping[input_metadata.to_cache]

            cache_ops.reshape_and_cache(
                key_to_cache,
                value_to_cache,
                key_cache,
                value_cache,
                slot_mapping,
            )

        if input_metadata.num_generation_tokens > 0:
            # Decoding run.
            assert input_metadata.num_prompt_tokens == 0
            assert key_cache is not None and value_cache is not None, (
                "key_cache and value_cache must be provided when "
                "generating tokens.")
            # Compute the attention op for generation tokens.
            self.single_query_cached_kv_attention(output, query, key_cache,
                                                  value_cache, input_metadata,
                                                  self.get_alibi_slopes())

        # Reshape the output tensor.
        # NOTE(woosuk): The output tensor may include paddings.
        return output.view(batch_size, seq_len,
                           self.num_heads * self.head_size)


class PagedAttentionWithRoPE(PagedAttention):
    """PagedAttention with rotary positional embedding."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
        num_kv_heads: Optional[int] = None,
        is_neox_style: bool = True,
        rope_scaling: Optional[Dict[str, Any]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__(num_heads,
                         head_size,
                         scale,
                         num_kv_heads,
                         sliding_window=sliding_window)
        self.rotary_emb = get_rope(head_size, rotary_dim, max_position, base,
                                   is_neox_style, rope_scaling)

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
            positions: shape = [batch_size, seq_len]
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """

        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        query, key = self.rotary_emb(positions, query, key)
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

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 slopes: List[float],
                 num_kv_heads: Optional[int] = None) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads)
        assert len(slopes) == num_heads

        slopes = torch.tensor(slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def set_attn_bias(self, input_metadata: InputMetadata,
                      dtype: torch.dtype) -> None:
        if input_metadata.attn_bias is not None:
            # Already set by a previous layer.
            return
        # Generates ALiBi mask based on the max prompt length.
        max_prompt_len = input_metadata.max_prompt_len
        bias = torch.arange(max_prompt_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(prompt_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        bias = bias[None, :] - bias[:, None]
        bias = bias.to(self.alibi_slopes.device)

        # When using custom attention bias, xformers requires the bias to
        # be sliced from a tensor whose length is a multiple of 8.
        padded_len = (max_prompt_len + 7) // 8 * 8
        bias = torch.empty(
            input_metadata.num_prompts,
            self.num_heads,
            max_prompt_len,
            padded_len,
            device=self.alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :max_prompt_len].copy_(bias)
        bias.mul_(self.alibi_slopes[:, None, None])
        attn_bias = LowerTriangularMaskWithTensorBias(bias)
        input_metadata.attn_bias = attn_bias

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
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        if self.num_kv_heads != self.num_heads:
            # Project the key and value tensors to the desired number of heads.
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=1)
        batch_size = input_metadata.num_prompts
        seq_len = input_metadata.max_prompt_len

        out = xops.memory_efficient_attention_forward(
            query.view(batch_size, seq_len, self.num_heads, self.head_size),
            key.view(batch_size, seq_len, self.num_heads, self.head_size),
            value.view(batch_size, seq_len, self.num_heads, self.head_size),
            attn_bias=input_metadata.attn_bias,
            p=0.0,
            scale=self.scale,
        )
        # TODO(woosuk): Unnecessary copy. Optimize.
        output.copy_(out.view(-1, self.num_heads, self.head_size))
        return output

    def get_alibi_slopes(self) -> Optional[torch.Tensor]:
        return self.alibi_slopes
