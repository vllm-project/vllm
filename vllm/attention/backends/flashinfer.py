from typing import Type, Tuple, List, Dict, Optional
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)

import torch
import flashinfer
from vllm._C import cache_ops
from dataclasses import dataclass


class FlashInferBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def make_metadata(**kwargs) -> "FlashInferMetadata":
        return FlashInferMetadata.new(**kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        raise NotImplementedError


@dataclass
class FlashInferMetadata(AttentionMetadataPerStage):
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool

    # The indptr of the paged kv-cache, shape: [batch_size + 1].
    # Please follow the definition in the FlashInfer documentation: https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout.
    paged_kv_indptr: List[int]

    # The indices of the paged kv-cache of all sequences.
    paged_kv_indices: List[int]

    # The last page length of the paged kv-cache of all sequences, shape: [batch_size].
    paged_kv_last_page_len: List[int]

    # The number of query/output heads.
    num_qo_heads: int

    # The number of key/value heads.
    num_kv_heads: int

    # The dimension of the heads
    head_dim: int

    # The wrapper for the prefill or decode operation.
    wrapper = None

    # The indptr of the query/output sequence, shape: [batch_size + 1].
    # This is only used for the prefill operation.
    subquery_start_loc: Optional[torch.Tensor] = None

    # The block size for the decode operation.
    block_size: Optional[int] = None

    use_cuda_graph: bool = False

    def __post_init__(self):
        assert not self.use_cuda_graph, "CUDA graph is not supported yet."
        # Allocate 16MB workspace buffer
        # Follow the example: https://docs.flashinfer.ai/api/python/prefill.html#batch-prefill-append-attention
        workspace_buffer = torch.empty(16 * 1024 * 1024,
                                       dtype=torch.uint8,
                                       device="cuda:0")
        if self.is_prompt:
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD")
            self.wrapper.begin_forward(self.subquery_start_loc,
                                       self.paged_kv_indptr,
                                       self.paged_kv_indices,
                                       self.paged_kv_last_page_len,
                                       self.num_qo_heads, self.num_kv_heads,
                                       self.head_dim)
        else:
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, "NHD")
            self.wrapper.begin_forward(
                self.paged_kv_indptr,
                self.paged_kv_indices,
                self.paged_kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                pos_encoding_mode=
                "NONE",  # FIXME: Add support for pos_encoding_mode
                data_type=torch.float16  # FIXME: Add support for data_type
            )


class FlashInferImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        pass

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, kv_cache: Optional[torch.Tensor],
                attn_metadata: AttentionMetadata[FlashInferMetadata]):
        if kv_cache is not None:
            # Use the same reshape and cache kernel as flash attention.
            cache_ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping.flatten(),
                attn_metadata.kv_cache_dtype,
            )

        if attn_metadata.is_prompt:
            assert kv_cache is None, "Does not support prefix caching yet."
            attn_metadata.prefill_metadata.wrapper.forward(query,
                                                           kv_cache,
                                                           causal=True)

        else:
            attn_metadata.decode_metadata.wrapper.forward(query, kv_cache)
