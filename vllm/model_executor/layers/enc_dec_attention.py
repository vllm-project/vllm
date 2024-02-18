"""Multi-head attention for encoder-decoder models."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalMask,
    LowerTriangularMaskWithTensorBias,
)
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.utils import is_hip
from vllm.model_executor.layers.attention import paged_attention

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class EncDecAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(
                f"head_size ({self.head_size}) is not supported. "
                f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}."
            )


class EncoderAttention(EncDecAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
    ) -> None:
        super().__init__(num_heads, head_size, scale)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Encoder attention forward pass.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            custom_bias: Custom bias tensor.

        Returns:
            Output tensor.
        """
        # query: [batch_size, seq_len, num_heads * head_size]
        # key: [batch_size, seq_len, num_heads * head_size]
        # value: [batch_size, seq_len, num_heads * head_size]
        # custom_bias: [batch_size, seq_len, seq_len]
        # output: [batch_size, seq_len, num_heads * head_size]

        assert input_metadata.is_prompt
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_size)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_size)
        # print("query shape: ", query.shape)
        if input_metadata.attn_bias is None:
            input_metadata.attn_bias = BlockDiagonalCausalMask.from_seqlens(
                [seq_len] * batch_size
            )
        # When using custom attention bias, xformers requires the bias to
        # be sliced from a tensor whose length is a multiple of 8.
        # padded_len = (seq_len + 7) // 8 * 8
        # pad_len = padded_len - seq_len
        # input_metadata.attn_bias = F.pad(input_metadata.attn_bias, (0, pad_len))
        # print("attention bias padded shape: ", input_metadata.attn_bias.shape)

        input_metadata.attn_bias = input_metadata.attn_bias[:, :, :, :seq_len]

        # print("attention bias shape: ", input_metadata.attn_bias.shape)
        # Normal attention
        out = xops.memory_efficient_attention_forward(
            query,
            key,
            value,
            attn_bias=input_metadata.attn_bias,
            p=0.0,
            scale=self.scale,
            op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0]
            if (is_hip())
            else None,
        )
        output = out.view(batch_size, seq_len, hidden_size)
        return output


class DecoderAttention(EncDecAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
    ) -> None:
        super().__init__(num_heads, head_size, scale)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ):
        """Decoder attention forward pass.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            key_cache: Key cache tensor.
            value_cache: Value cache tensor.
            custom_bias: Custom bias tensor.

        Returns:
            Output tensor.
        """

        # print("key shape pre view: ", key.shape)
        # print("value shape pre view: ", value.shape)

        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_heads, self.head_size)
        value = value.view(-1, self.num_heads, self.head_size)
        # print("key shape: ", key.shape)
        # print("key: ", key)
        # print("value shape: ", value.shape)
        # print("value: ", value)
        # print("slot mapping: ", input_metadata.slot_mapping[:, -1].flatten())
        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            # print("key_cache before: ", key_cache)
            # print("value_cache before: ", value_cache)

            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                input_metadata.slot_mapping[:, -1].flatten().contiguous()
            )

            # print("key_cache after: ", key_cache)
            # print("value_cache after: ", value_cache)

        max_prompt_len = input_metadata.prompt_lens.max().item()
        block_size = value_cache.shape[3]
        prompt_table_len = (max_prompt_len + block_size - 1) // block_size
        block_tables = input_metadata.block_tables[:, prompt_table_len:].contiguous()
        # print("decoder self attention block_tables", block_tables)
        output = paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=input_metadata.context_lens,
            max_context_len=input_metadata.max_context_len,
            num_kv_heads=self.num_heads,
            scale=self.scale,
            alibi_slopes=None,
            custom_bias=input_metadata.attn_bias.to(torch.float32),
        )
        return output.view(batch_size, seq_len, hidden_size)


class CrossAttention(EncDecAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
    ) -> None:
        super().__init__(num_heads, head_size, scale)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ):
        """Cross attention forward pass.
        Args:
            query: Query tensor.
            key_cache: Key cache tensor.
            value_cache: Value cache tensor.
            input_metadata: Input metadata.
            key: Key tensor. Only needed in the first pass.
            value: Value tensor. Only needed in the first pass.
            custom_bias: Custom bias tensor.
        Returns:
            Output tensor.
        """
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            # print("key shape pre view: ", key.shape)
            key = key.view(-1, self.num_heads, self.head_size)
            # print("key_shape: ", key.shape)
            # print("key sum", key.sum((1, 2)))
        if value is not None:
            # print("value shape pre view: ", value.shape)
            value = value.view(-1, self.num_heads, self.head_size)
            # print("value_shape: ", value.shape)
            # print("value sum", value.sum((1, 2)))

        # print("slot mapping: ", input_metadata.slot_mapping[:, :-1].flatten().shape)
        # print("slot mapping: ", input_metadata.slot_mapping[:, :-1].flatten())
        # Reshape the keys and values and store them in the cache.
        # It only happens during the first pass.
        if (
            input_metadata.is_prompt
            and key_cache is not None
            and value_cache is not None
        ):
            assert key is not None and value is not None
            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                input_metadata.slot_mapping[:, :-1].flatten().contiguous(),
            )
        
        # for slot in input_metadata.slot_mapping[:, :-1].flatten():
        #     if slot != -1:
        #         block_number = slot//16;
        #         block_offset = slot%16;
        #         print(f"key_cache sum at {slot}: ", key_cache[block_number, :, :, block_offset, :].sum())
        #         print(f"value_cache sum at {slot}: ", value_cache[block_number, :, :, block_offset].sum())
        max_prompt_len = input_metadata.prompt_lens.int().max().item()
        # print("max_prompt_len: ", max_prompt_len)
        block_size = value_cache.shape[3]
        prompt_table_len = (max_prompt_len + block_size - 1) // block_size
        block_tables = input_metadata.block_tables[:, :prompt_table_len].contiguous()

        output = paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=input_metadata.prompt_lens.int(),
            max_context_len=max_prompt_len,
            num_kv_heads=self.num_heads,
            scale=self.scale,
            alibi_slopes=None,
            custom_bias=None,
        )

        return output.view(batch_size, seq_len, hidden_size)
