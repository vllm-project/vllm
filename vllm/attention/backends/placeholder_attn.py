from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)

# Placeholder attention backend for models like Mamba that don't have attention.
# Mainly exists to sidestep get_attn_backend.
# The attention metadata is still needed for Mamba.


class PlaceholderAttentionBackend(AttentionBackend):
    """Placeholder backend for when no attention is needed."""

    @staticmethod
    def get_name() -> str:
        return "No attention"

    @staticmethod
    def get_impl_cls() -> Type["PlaceholderAttentionImpl"]:
        return PlaceholderAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["PlaceholderAttentionMetadata"]:
        return PlaceholderAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (1, 1, 1, 1, 1)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        return

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        return


@dataclass
class PlaceholderAttentionMetadata(AttentionMetadata):
    """Attention metadata for prefill and decode batched together."""
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int]
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    _cached_prefill_metadata: Optional["PlaceholderAttentionMetadata"] = None
    _cached_decode_metadata: Optional["PlaceholderAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["PlaceholderAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens_tensor is not None
        assert self.block_tables is not None
        assert self.seq_start_loc is not None

        self._cached_prefill_metadata = PlaceholderAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1],
            seq_start_loc=self.seq_start_loc[:self.num_prefills + 1],
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills],
            block_tables=self.block_tables[:self.num_prefills],
            use_cuda_graph=False,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["PlaceholderAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = PlaceholderAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            max_query_len=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills:],
            use_cuda_graph=self.use_cuda_graph,
        )
        return self._cached_decode_metadata


class PlaceholderAttentionImpl(AttentionImpl):

    def __init__(self, *args, **kwargs) -> None:
        return

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
