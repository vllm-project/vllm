"""Attention layer with torch scaled_dot_product_attention and PagedAttention."""
from typing import List, Optional

import torch

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention.ops.paged_attn import (
    PagedAttentionImpl)

class TorchSDPABackend:

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
        self.need_mask = (alibi_slopes is not None) or (sliding_window is not None)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        suppored_head_sizes = PagedAttentionImpl.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Forward pass with torch scaled_dot_product_attention and PagedAttention.
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

        if key_cache is not None and value_cache is not None:
            PagedAttentionImpl.reshape_and_cache(key, value, key_cache,
                                                 value_cache, input_metadata)

        if input_metadata.is_prompt:
            if (key_cache is None or value_cache is None
                    or input_metadata.block_tables.numel() == 0):
                if self.num_kv_heads != self.num_heads:
                    query = query.view(query.shape[0], self.num_kv_heads,
                                       self.num_queries_per_kv,
                                       query.shape[-1])
                    key = key[:, :,
                              None, :].expand(key.shape[0], self.num_kv_heads,
                                              self.num_queries_per_kv,
                                              key.shape[-1])
                    value = value[:, :,
                                  None, :].expand(value.shape[0],
                                                  self.num_kv_heads,
                                                  self.num_queries_per_kv,
                                                  value.shape[-1])

                if self.need_mask and input_metadata.attn_bias is None:
                    if self.alibi_slopes is not None:
                        att_bias = _make_alibi_bias(self.alibi_slopes, self.num_kv_heads, batch_size, seq_len, query.dtype)
                    elif self.sliding_window is not None:
                        att_bias = _make_sliding_window_bias(seq_len, self.sliding_window, query.dtype)
                    input_metadata.attn_bias = att_bias

                query = query.unflatten(0, (batch_size, seq_len)) 
                key = key.unflatten(0, (batch_size, seq_len))
                value = value.unflatten(0, (batch_size, seq_len))

                query = query.movedim(1, query.dim() - 2)
                key = key.movedim(1, key.dim() - 2)
                value = value.movedim(1, value.dim() - 2)
                out = torch.nn.functional.scaled_dot_product_attention(
                    query, 
                    key, 
                    value, 
                    input_metadata.attn_bias,
                    0.0, 
                    is_causal=not self.need_mask).movedim(query.dim() - 2, 1).contiguous()
                # output = out.view_as(query)
                # FIXME: half input will generate float output, next ipex release will fix this.
                output = out.view_as(query).to(query.dtype)
                
            else:
                # prefix-enabled attention
                raise RuntimeError("SDPA backend doesn't support prefix decoding.") 

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
        return output.view(batch_size, seq_len, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    bias = torch.arange(seq_len, dtype=dtype)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))

    return bias

def _make_sliding_window_bias(
    seq_len: int,
    window_size: int,
    dtype: torch.dtype, 
) -> torch.Tensor:
    tensor = torch.full(
        (1, seq_len, seq_len),
        dtype=dtype,
        fill_value=1,
    )
    shift = 0
    mask = torch.tril(tensor, diagonal=shift).to(dtype)  # type: ignore
    mask = torch.triu(mask, diagonal=shift - window_size + 1)
    mask = torch.log(mask)
    return mask.to(dtype)