###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import os
import torch
import math
from vllm.hpu import cache_ops, xops
from vllm.hpu.utils import Matmul, Softmax, VLLMKVCache
from vllm.hpu.attn_bias import (AttentionBias,
                                LowerTriangularMaskWithTensorBias)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.ops.habana_paged_attn import (HabanaPagedAttention,
                                                  HabanaPagedAttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


class HabanaAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["HabanaAttentionImpl"]:
        return HabanaAttentionImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "HabanaAttentionMetadata":
        return HabanaAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return HabanaPagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                       num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        HabanaPagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        HabanaPagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass(frozen=True)
class HabanaAttentionMetadata(HabanaPagedAttentionMetadata, AttentionMetadataPerStage):
    """Metadata for HabanaAttentionbackend."""
    attn_bias: Optional[torch.Tensor]
    seq_lens_tensor: Optional[torch.Tensor]


class HabanaAttentionImpl(AttentionImpl, torch.nn.Module):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

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
        super(AttentionImpl, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.qk_matmul = Matmul()
        self.softmax = Softmax()
        self.kv_matmul = Matmul()
        self.key_cache = VLLMKVCache()
        self.value_cache = VLLMKVCache()
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = HabanaPagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata[HabanaAttentionMetadata],
        kv_scale: float,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape
        _, seq_len_kv, _ = key.shape

        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if prefill_meta := attn_metadata.prefill_metadata:
            block_indices = prefill_meta.block_indices
            block_offsets = prefill_meta.block_offsets
        if decode_meta := attn_metadata.decode_metadata:
            block_indices = decode_meta.block_indices
            block_offsets = decode_meta.block_offsets
        if kv_cache is not None:
            key_cache, value_cache = HabanaPagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            key_cache = self.key_cache(key, key_cache, block_indices, block_offsets)
            value_cache = self.value_cache(value, value_cache, block_indices, block_offsets)

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            assert prefill_meta.attn_bias is not None, 'attn_bias must be set before calling model.forward!'
            query_shape = (batch_size, seq_len, self.num_heads, self.head_size)
            kv_shape = (batch_size, seq_len_kv, self.num_kv_heads, self.head_size)
            out = xops.prompt_attention(
                query.view(query_shape),
                key.view(kv_shape),
                value.view(kv_shape),
                attn_bias=prefill_meta.attn_bias,
                p=0.0,
                scale=self.scale,
                qk_matmul_op=self.qk_matmul,
                softmax_op=self.softmax,
                kv_matmul_op=self.kv_matmul,
            )
            output = out.reshape(batch_size, seq_len, hidden_size)
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output = HabanaPagedAttention.forward_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_list=decode_meta.block_list,
                block_mapping=decode_meta.block_mapping,
                block_bias=decode_meta.attn_bias,
                scale=self.scale,
                qk_matmul_op=self.qk_matmul,
                kv_matmul_op=self.kv_matmul,
                keys_fetch_func=self.key_cache.fetch_from_cache,
                values_fetch_func=self.value_cache.fetch_from_cache)

        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> LowerTriangularMaskWithTensorBias:
    attn_biases = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (seq_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            seq_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :seq_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        if num_heads != num_kv_heads:
            bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases
