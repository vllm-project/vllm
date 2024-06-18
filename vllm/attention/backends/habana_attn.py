###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import os
import torch
import math
from vllm.hpu import cache_ops
from vllm.hpu.utils import VLLMKVCache
from vllm.hpu.attn_bias import (AttentionBias,
                                LowerTriangularMaskWithTensorBias)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.ops.habana_paged_attn import (HabanaPagedAttention,
                                                  HabanaPagedAttentionMetadata)
from vllm.logger import init_logger
from vllm.utils import Matmul, Softmax

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


@dataclass
class HabanaAttentionMetadata(AttentionMetadataPerStage, HabanaPagedAttentionMetadata):
    """Metadata for HabanaAttentionbackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch.
    max_query_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None


import habana_frameworks.torch.utils.experimental as htexp
PA_SPLIT_VALUE_DEFAULT = '0' if (htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi3) else '1'
PA_SPLIT_VALUE = (os.environ.get('PA_SPLIT_VALUE', PA_SPLIT_VALUE_DEFAULT) == '1')

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

    def prompt_attention(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_bias: Optional[torch.Tensor] = None,
            p: float = 0.0,
            scale: Optional[float] = None,
    ) -> torch.Tensor:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query_heads = query.size(1)
        kv_heads = key.size(1)
        if query_heads != kv_heads:
            query = query.unflatten(1, (kv_heads, -1))
            key = key.unflatten(1, (kv_heads, 1))
            value = value.unflatten(1, (kv_heads, 1))
            attn_bias = attn_bias.unsqueeze(2)
        attn_weights = self.qk_matmul(query * scale, key.transpose(-1, -2))
        if attn_bias is not None:
            attn_weights.add_(attn_bias)
        attn_weights = self.softmax(attn_weights, dim=-1)
        attn_weights = self.kv_matmul(attn_weights, value)
        if query_heads != kv_heads:
            attn_weights = attn_weights.flatten(1, 2)
        attn_weights = attn_weights.transpose(1, 2)
        return attn_weights

    def paged_attention_v1(self, query, key_cache, value_cache, head_mapping, scale, block_tables, context_lens, block_size, alibi_slopes, kv_cache_dtype=None)  -> None:
        seq_len = block_tables.size(1)
        batch_size, query_heads, _ = query.shape
        _, _, kv_heads, _ = key_cache.shape
        min_inf = torch.finfo(query.dtype).min
        mask = (torch.arange(0, seq_len * block_size, dtype=torch.int32, device=key_cache.device)
                .view(1, -1)
                .expand(batch_size, -1)
                .ge(context_lens.view(-1, 1))
                .view(batch_size, 1, 1, -1))
        query.mul_(scale)
        query = query.unsqueeze(-2)
        keys = self.key_cache.fetch_from_cache(key_cache, block_tables, (0, 2, 3, 1))
        if query_heads != kv_heads:
            query = query.unflatten(1, (kv_heads, -1))
            keys = [k.unflatten(1, (kv_heads, 1)) for k in keys]
            mask = mask.unsqueeze(2)

        attn_weights = [self.qk_matmul(query, k) for k in keys]
        attn_weights = self.softmax(torch.cat(attn_weights, dim=-1).masked_fill(mask, min_inf),
                                    dim=-1)

        values = self.value_cache.fetch_from_cache(value_cache, block_tables, (0, 2, 1, 3))
        if PA_SPLIT_VALUE:
            attn_weights = attn_weights.split(block_size, dim=-1)
        else:
            values = [torch.cat(values, dim=-2)]
            attn_weights = [attn_weights]
        if query_heads != kv_heads:
            values = [v.unflatten(1, (kv_heads, 1)) for v in values]
        attn_weights = [self.kv_matmul(a, v) for a, v in zip(attn_weights, values)]
        if query_heads != kv_heads:
            attn_weights = [a.flatten(1, 2) for a in attn_weights]
        attn_weights = sum(attn_weights)
        return attn_weights.squeeze(-2)

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
        if kv_cache is not None:
            key_cache, value_cache = HabanaPagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            block_indices, block_offset = cache_ops.prepare_to_cache(key_cache, attn_metadata.slot_mapping)
            key_cache = self.key_cache(key, key_cache, block_indices, block_offset)
            value_cache = self.value_cache(value, value_cache, block_indices, block_offset)

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # TODO: move this outside of model
                assert prefill_meta.attn_bias is not None, 'attn_bias must be set before calling model.forward!'
                query_shape = (batch_size, seq_len, self.num_heads, self.head_size)
                kv_shape = (batch_size, seq_len_kv, self.num_kv_heads, self.head_size)
                out = self.prompt_attention(
                    query.view(query_shape),
                    key.view(kv_shape),
                    value.view(kv_shape),
                    attn_bias=prefill_meta.attn_bias,
                    p=0.0,
                    scale=self.scale,
                )
                output = out.reshape(batch_size, seq_len, hidden_size)
            else:
                # prefix-enabled attention
                output = HabanaPagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.subquery_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.context_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                )
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            block_size = value_cache.shape[1]
            output = self.paged_attention_v1(
                query,
                key_cache,
                value_cache,
                self.num_kv_heads,
                self.scale,
                decode_meta.block_tables,
                decode_meta.seq_lens_tensor,
                block_size,
                self.alibi_slopes,
                attn_metadata.kv_cache_dtype,
            )

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
