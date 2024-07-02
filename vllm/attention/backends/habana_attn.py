###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import vllm.hpu.ops as ops

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
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


@dataclass
class HabanaAttentionMetadata(AttentionMetadata, HabanaPagedAttentionMetadata):
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
        self.attn_bias: Optional[List[torch.Tensor]] = None


class HabanaAttentionImpl(AttentionImpl):
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
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        max_seq_len : Optional[int] = 4096,
    ) -> None:
        self.kv_cache_dtype = kv_cache_dtype
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        self.position_bias = None
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.bfloat16)
            self.position_bias = _make_alibi_bias(alibi_slopes,
                                                  num_kv_heads,
                                                  alibi_slopes.dtype,
                                                  max_seq_len)
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
        attn_metadata: HabanaAttentionMetadata,
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
            HabanaPagedAttention.write_to_paged_cache(key, value, key_cache,
                                                      value_cache,
                                                      attn_metadata.slot_mapping,
                                                      self.kv_cache_dtype,
                                                      attn_metadata.is_prompt)

        if attn_metadata.is_prompt:
            # Prompt run.
            if kv_cache is None or attn_metadata.block_tables.numel() == 0:
                # TODO: move this outside of model
                assert attn_metadata.attn_bias is not None, 'attn_bias must be set before calling model.forward!'
                attn_bias = attn_metadata.attn_bias
                if self.alibi_slopes is not None:
                    attn_bias.add_(self.position_bias[:, :, -attn_bias.size(2):, -attn_bias.size(3):])

                query_shape = (batch_size, seq_len, self.num_heads, self.head_size)
                kv_shape = (batch_size, seq_len_kv, self.num_kv_heads, self.head_size)
                out = ops.prompt_attention(
                    query.view(query_shape),
                    key.view(kv_shape),
                    value.view(kv_shape),
                    attn_bias=attn_bias,
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
                    attn_metadata.block_tables,
                    attn_metadata.subquery_start_loc,
                    attn_metadata.seq_lens_tensor,
                    attn_metadata.context_lens_tensor,
                    attn_metadata.max_query_len,
                    self.alibi_slopes,
                )
        else:
            # Decoding run.
            output = HabanaPagedAttention.forward_decode(
                query,
                key_cache,
                value_cache,
                attn_metadata.block_tables,
                attn_metadata.seq_lens_tensor,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.position_bias,
                kv_scale
            )
        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_len: int,
) -> torch.Tensor:
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
    return bias
