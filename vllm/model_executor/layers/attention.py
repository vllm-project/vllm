"""Multi-head attention."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch._custom_ops as torch_custom_ops
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


@torch_custom_ops.custom_op("vllm::cache_kv")
def cache_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError()


@torch_custom_ops.impl("vllm::cache_kv", device_types="cuda")
def cache_kv_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # NOTE: query is actually not used in the custom op. We pass it to create
    # a fake dependency so that the compiler cannot skip the custom op.
    # FIXME: The custom op should not be in-place.
    cache_ops.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    )
    return query, key


@torch_custom_ops.impl_abstract("vllm::cache_kv")
def cache_kv_abstract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(query), torch.empty_like(key)


@torch_custom_ops.custom_op("vllm::paged_attn")
def paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    head_mapping: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor],
) -> torch.Tensor:
    raise NotImplementedError()


@torch_custom_ops.impl("vllm::paged_attn", device_types="cuda")
def paged_attn_impl(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    head_mapping: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor],
) -> torch.Tensor:
    output = torch.empty_like(query)
    block_size = value_cache.shape[3]
    ops.paged_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        8192,  # FIXME
        alibi_slopes,
    )
    return output


@torch_custom_ops.impl_abstract("vllm::paged_attn")
def paged_attn_abstract(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    head_mapping: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.empty_like(query)


class PagedAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_mapping = torch.repeat_interleave(
            torch.arange(self.num_kv_heads, dtype=torch.int32, device="cuda"),
            self.num_queries_per_kv)

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        slot_mapping = input_metadata.slot_mapping.flatten()

        # Reshape the keys and values and store them in the cache.
        if input_metadata.is_prompt:
            # If key_cache and value_cache are not provided, the new key
            # and value vectors will not be cached. This happens during
            # the initial profile run.
            if key_cache is not None and value_cache is not None:
                cache_ops.reshape_and_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slot_mapping,
                )
        else:
            query, key = torch.ops.vllm.cache_kv(
                query,
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
            )

        if input_metadata.is_prompt:
            # Prompt run.
            if self.num_kv_heads != self.num_heads:
                # For MQA/GQA, project the key and value tensors to the desired
                # number of heads.
                query = query.view(query.shape[0], self.num_kv_heads,
                                   self.num_queries_per_kv, query.shape[-1])
                key = key[:, :,
                          None, :].expand(key.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          key.shape[-1])
                value = value[:, :, None, :].expand(value.shape[0],
                                                    self.num_kv_heads,
                                                    self.num_queries_per_kv,
                                                    value.shape[-1])
            # Set attention bias.
            # FIXME: This is a hack.
            if not hasattr(input_metadata, "attn_bias"):
                prompt_lens = [seq_len] * batch_size
                attn_bias = BlockDiagonalCausalMask.from_seqlens(prompt_lens)
                # FIXME: Sliding window is not properly applied.
                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                input_metadata.attn_bias = attn_bias

            # TODO(woosuk): The unsqueeze op may incur some CPU overhead. Optimize.
            out = xops.memory_efficient_attention_forward(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                attn_bias=input_metadata.attn_bias,
                p=0.0,
                scale=self.scale,
            )
            output = out.view_as(query)
        else:
            # Decoding run.
            output = torch.ops.vllm.paged_attn(
                query,
                key_cache,
                value_cache,
                self.head_mapping,
                self.scale,
                input_metadata.block_tables,
                input_metadata.context_lens,
                self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)


# FIXME: Temporary hack to avoid import errors.
PagedAttentionWithRoPE = PagedAttention
PagedAttentionWithALiBi = PagedAttention
