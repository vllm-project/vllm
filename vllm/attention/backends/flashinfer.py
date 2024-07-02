from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np

try:
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    from vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
    BatchDecodeWithPagedKVCacheWrapper = None
    BatchPrefillWithPagedKVCacheWrapper = None

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           is_block_tables_empty)
from vllm.sequence import SequenceGroupMetadata
from vllm.utils import get_kv_cache_torch_dtype, make_tensor_with_pad


class FlashInferBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "flashinfer"

    @staticmethod
    def get_impl_cls() -> Type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashInferMetadata

    @staticmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        return FlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 128, 256]


@dataclass
class FlashInferMetadata(AttentionMetadata):
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int

    use_cuda_graph: bool = True

    prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None

    # Metadata for the prefill stage
    seq_start_loc: Optional[torch.Tensor] = None
    query_start_loc: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None

    # An example for paged_kv_indices, paged_kv_indptr:
    # request 1, page indices [0, 5, 8]
    # request 2, page indices [1, 6, 7]
    # request 3, page indices [3, 4]
    # paged_kv_indices is a concatenation of page indices of all requests:
    # [0, 5, 8, 1, 6, 7, 3, 4]
    # paged_kv_indptr is used to index into paged_kv_indices:
    # [0, 3, 6, 8]
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: Optional[torch.Tensor] = None
    # The page indices of the paged kv cache
    paged_kv_indices: Optional[torch.Tensor] = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: Optional[torch.Tensor] = None
    # The number of query/output heads
    num_qo_heads: Optional[int] = None
    # The number of key/value heads
    num_kv_heads: Optional[int] = None
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    # Block size of vllm
    page_size: Optional[int] = None
    # The data type of the paged kv cache
    data_type: torch.dtype = None
    device: torch.device = torch.device("cuda")

    def __post_init__(self):
        # Refer to
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        supported_head_sizes = FlashInferBackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim \
                not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f"received {self.head_dim}.")

    def begin_forward(self):
        if self.num_prefill_tokens > 0:
            if self.paged_kv_indices is None:
                return

            assert self.prefill_wrapper is not None
            assert self.paged_kv_indices is not None
            assert self.paged_kv_indptr is not None
            assert self.paged_kv_last_page_len is not None
            self.paged_kv_indices = self.paged_kv_indices.to(self.device)
            self.paged_kv_indptr = self.paged_kv_indptr.to(self.device)
            self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(
                self.device)
            self.prefill_wrapper.end_forward()
            self.prefill_wrapper.begin_forward(
                self.query_start_loc, self.paged_kv_indptr,
                self.paged_kv_indices, self.paged_kv_last_page_len,
                self.num_qo_heads, self.num_kv_heads, self.head_dim,
                self.page_size)
        else:
            if not self.use_cuda_graph:
                assert self.paged_kv_indices is not None
                assert self.paged_kv_indptr is not None
                assert self.paged_kv_last_page_len is not None
                self.paged_kv_indices = self.paged_kv_indices.to(self.device)
                self.paged_kv_indptr = self.paged_kv_indptr.to(self.device)
                self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(
                    self.device)

            assert self.decode_wrapper is not None
            self.decode_wrapper.end_forward()
            self.decode_wrapper.begin_forward(
                self.paged_kv_indptr,
                self.paged_kv_indices,
                self.paged_kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                # Disable flashinfer's pos encoding and use vllm's rope.
                pos_encoding_mode="NONE",
                data_type=self.data_type)

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        if skip_fields is None:
            skip_fields = set()
        # We need to skip the prefill/decode_wrapper field since it cannot be
        # broadcasted with nccl when TP is enabled.
        skip_fields.add('prefill_wrapper')
        skip_fields.add('decode_wrapper')
        return super().asdict_zerocopy(skip_fields)

    @property
    def prefill_metadata(self) -> Optional["FlashInferMetadata"]:
        # Currently chunked prefill is not supported
        if self.num_decode_tokens == 0:
            assert self.num_prefills > 0
            return self

        return None

    @property
    def decode_metadata(self) -> Optional["FlashInferMetadata"]:
        # Currently chunked prefill is not supported
        if self.num_prefills > 0:
            assert self.num_decode_tokens == 0
            return None

        return self


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):

    def __init__(self, block_size, sliding_window, use_v2_block_manager):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        self.sliding_window = sliding_window
        self.block_size = block_size
        self.use_v2_block_manager = use_v2_block_manager

        # Please follow https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
        # for the precise definition of the following fields.
        # An example:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]
        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        # paged_kv_indptr is used to index into paged_kv_indices:
        # [0, 3, 6, 8]
        self.paged_kv_indices: List[int] = []
        # 0 at the beginning of paged_kv_indptr indicates the start of the
        # first request’s page indices in the paged_kv_indices list.
        self.paged_kv_indptr: List[int] = [0]
        # paged_kv_last_page_len is the length of the last page of each request
        self.paged_kv_last_page_len: List[int] = []

    def _update_paged_kv_metadata(self, seq_data, block_table):
        if block_table is None:
            return

        seq_len = seq_data.get_len()
        # Get the number of valid blocks based on sequence length.
        # If seq_len = 16, block_size = 16,
        # block_table_bound is 1 with 1 valid block.
        # If seq_len = 15, block_size = 16,
        # block_table_bound is 0 + 1 with 1 valid block.
        block_table_bound = seq_len // self.block_size + 1 \
                            if seq_len % self.block_size != 0 \
                            else seq_len // self.block_size

        self.paged_kv_indices.extend(block_table[:block_table_bound])
        self.paged_kv_indptr.append(self.paged_kv_indptr[-1] +
                                    block_table_bound)

        last_page_len = seq_len % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.paged_kv_last_page_len.append(last_page_len)

    def add_prefill_seq_group(self, seq_group_metadata: SequenceGroupMetadata,
                              tokens: List[int], seq_id: int, seq_len: int,
                              query_len: int, context_len: int,
                              prefix_cache_hit, chunked_prefill_enabled,
                              computed_block_nums,
                              curr_sliding_window_blocks) -> None:
        # Compute block table.
        # TODO(sang): Combine chunked prefill and prefix caching by
        # only allowing multiple of block_size chunk size.
        # NOTE: This only works for oooooooxxx style attention.
        if prefix_cache_hit:
            assert computed_block_nums is not None
            assert self.sliding_window is None
            block_table = computed_block_nums
        elif (chunked_prefill_enabled
              and seq_group_metadata.block_tables is not None):
            block_table = seq_group_metadata.block_tables[seq_id]
            if curr_sliding_window_blocks is not None:
                block_table = block_table[-curr_sliding_window_blocks:]
        else:
            # Prefill without chunked prefill or memory profiling.
            block_table = []

        self.block_tables.append(block_table)

        self.num_prefills += 1
        self.num_prefill_tokens += len(tokens)
        self.prefill_seq_lens.append(seq_len)

        # Compute slot mapping.
        block_table = None
        is_profile_run = is_block_tables_empty(seq_group_metadata.block_tables)
        if not is_profile_run:
            block_table = seq_group_metadata.block_tables[seq_id]

        start_idx = 0
        if self.sliding_window is not None:
            assert self.use_v2_block_manager \
                or context_len == 0, (
                "Prefix caching is currently not supported with "
                "sliding window attention in V1 block manager")
            # When prefill, we use it to not write slots to kv cache
            # to save memory.
            start_idx = max(0, query_len - self.sliding_window)

        compute_slot_mapping(self.slot_mapping, seq_len, context_len,
                             start_idx, self.block_size, block_table)

        # FlashInfer specific
        self._update_paged_kv_metadata(seq_group_metadata.seq_data[seq_id],
                                       block_table)

    def add_decode_seq_group(self, seq_group_metadata: SequenceGroupMetadata,
                             seq_id, seq_len, query_len, context_len,
                             curr_sliding_window_blocks, sliding_seq_len,
                             sliding_context_len):
        seq_data = seq_group_metadata.seq_data[seq_id]

        if seq_group_metadata.block_tables is not None:
            # chunked prefill or decode
            block_table = seq_group_metadata.block_tables[seq_id]
            if curr_sliding_window_blocks is not None:
                block_table = block_table[-curr_sliding_window_blocks:]
        else:
            # Only happens when memory profiling runs.
            block_table = []

        self.block_tables.append(block_table)

        assert query_len == 1, (
            "seq_len: {}, context_len: {}, query_len: {}".format(
                seq_len, context_len, query_len))
        self.num_decode_tokens += query_len

        # Compute the slot mapping.
        block_table = None
        is_profile_run = is_block_tables_empty(seq_group_metadata.block_tables)
        if not is_profile_run:
            block_table = seq_group_metadata.block_tables[seq_id]

        compute_slot_mapping(self.slot_mapping, seq_len, context_len, 0,
                             self.block_size, block_table)

        self._update_paged_kv_metadata(seq_data, block_table)

    def build(self, model_config, parallel_config, kv_cache_dtype, seq_lens,
              query_lens, decode_seq_lens, use_captured_graph: bool,
              cuda_graph_pad_size: int, graph_block_tables: np.ndarray,
              batch_size: int, device):
        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size + cuda_graph_pad_size

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = graph_block_tables[:batch_size]
            for i, block_table in enumerate(self.block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device=device)

            last_paged_kv_indptr = self.paged_kv_indptr[-1]
            self.paged_kv_indptr.extend([last_paged_kv_indptr] *
                                        cuda_graph_pad_size)
            self.paged_kv_last_page_len.extend([0] * cuda_graph_pad_size)
        else:
            max_block_table_len = max(
                len(block_table) for block_table in self.block_tables)
            block_tables = make_tensor_with_pad(
                self.block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=device)
        query_lens_tensor = torch.tensor(query_lens,
                                         dtype=torch.long,
                                         device=device)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        slot_mapping_tensor = torch.tensor(self.slot_mapping,
                                           dtype=torch.long,
                                           device=device)

        if len(self.paged_kv_indptr) > 0:
            paged_kv_indices_tensor = torch.tensor(self.paged_kv_indices,
                                                   device="cpu",
                                                   dtype=torch.int)
            paged_kv_indptr_tensor = torch.tensor(self.paged_kv_indptr,
                                                  device="cpu",
                                                  dtype=torch.int)
            paged_kv_last_page_len_tensor = torch.tensor(
                self.paged_kv_last_page_len, device="cpu", dtype=torch.int)
        else:
            paged_kv_indices_tensor = None
            paged_kv_indptr_tensor = None
            paged_kv_last_page_len_tensor = None

        kv_cache_dtype = get_kv_cache_torch_dtype(kv_cache_dtype,
                                                  model_config.dtype)

        return FlashInferMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            max_prefill_seq_len=max_prefill_seq_len,
            block_tables=block_tables,
            paged_kv_indptr=paged_kv_indptr_tensor,
            paged_kv_indices=paged_kv_indices_tensor,
            paged_kv_last_page_len=paged_kv_last_page_len_tensor,
            num_qo_heads=model_config.get_num_attention_heads(parallel_config),
            num_kv_heads=model_config.get_num_kv_heads(parallel_config),
            head_dim=model_config.get_head_size(),
            page_size=self.block_size,
            seq_start_loc=seq_start_loc,
            query_start_loc=query_start_loc,
            device=device,
            data_type=kv_cache_dtype,
            use_cuda_graph=use_captured_graph)


class FlashInferImpl(AttentionImpl):

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
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is not None:
            raise ValueError("Sliding window is not supported in FlashInfer.")
        self.sliding_window = (-1, -1)
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: FlashInferMetadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        assert kv_scale == 1.0
        num_tokens, hidden_size = query.shape
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if attn_metadata.num_prefill_tokens > 0:
            assert attn_metadata.num_decode_tokens == 0, (
                "Chunked prefill is not supported with flashinfer yet.")
        if attn_metadata.num_decode_tokens > 0:
            assert attn_metadata.num_prefill_tokens == 0, (
                "Chunked prefill is not supported with flashinfer yet.")

        if kv_cache is not None:
            # Use the same reshape and cache kernel as flash attention.
            ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
            )

        query = query.contiguous(
        )  # Flashinfer requires query to be contiguous
        if prefill_meta := attn_metadata.prefill_metadata:
            # We will use flash attention for prefill
            # when kv_cache is not provided.
            # This happens when vllm runs the profiling to
            # determine the number of blocks.
            if kv_cache is None:
                output = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
            else:
                assert prefill_meta is not None
                assert prefill_meta.prefill_wrapper is not None
                output = prefill_meta.prefill_wrapper.forward(query,
                                                              kv_cache,
                                                              causal=True)
        else:
            assert attn_metadata.decode_metadata is not None
            assert attn_metadata.decode_metadata.decode_wrapper is not None
            output = attn_metadata.decode_metadata.decode_wrapper.forward(
                query,
                kv_cache,
                sm_scale=self.scale,
            )
        return output.view(num_tokens, hidden_size)
