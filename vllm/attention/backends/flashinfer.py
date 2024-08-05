from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type

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
                                              AttentionMetadataBuilder,
                                              AttentionType)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.utils import (async_tensor_h2d, get_kv_cache_torch_dtype,
                        make_tensor_with_pad)

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUBuilder


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
    def get_builder_cls() -> Type["FlashInferMetadataBuilder"]:
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
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)

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
            assert self.query_start_loc is not None
            assert self.paged_kv_indices is not None
            assert self.paged_kv_indptr is not None
            assert self.paged_kv_last_page_len is not None
            batch_size = self.query_start_loc.shape[0] - 1
            assert batch_size >= 0
            # The prefill stage does not read kv cache.
            # Both paged_kv_indices and paged_kv_last_page_len are empty.
            # paged_kv_indptr is a zero tensor with size batch_size + 1.
            self.paged_kv_indptr = torch.zeros(batch_size + 1,
                                               device=self.device)
            self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(
                self.device)
            self.paged_kv_indices = self.paged_kv_indices.to(self.device)
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

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        self.input_builder = input_builder
        self.runner = input_builder.runner

        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.use_v2_block_manager = (
            input_builder.scheduler_config.use_v2_block_manager)

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

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables
        computed_block_nums = inter_data.computed_block_nums

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if inter_data.prefix_cache_hit:
                block_table = computed_block_nums
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            is_profile_run = is_block_tables_empty(block_tables)

            # Compute slot mapping.
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window,
                self.use_v2_block_manager)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

            # It is not necessary to add paged_kv_indices, paged_kv_indptr,
            # and paged_kv_last_page_len for profile run because we will
            # create dummy inputs.
            if is_profile_run:
                return

            block_table = block_tables[seq_id]
            self._update_paged_kv_tensors(block_table, seq_len)

    def _update_paged_kv_tensors(self, block_table: List[int], seq_len: int):
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

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.runner.graph_block_tables[:batch_size]
            for i, block_table in enumerate(self.block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.from_numpy(input_block_tables).to(
                device, non_blocking=True)

            last_paged_kv_indptr = self.paged_kv_indptr[-1]
            self.paged_kv_indptr.extend([last_paged_kv_indptr] *
                                        cuda_graph_pad_size)
            self.paged_kv_last_page_len.extend([0] * cuda_graph_pad_size)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        query_lens_tensor = async_tensor_h2d(query_lens, torch.long, device,
                                             self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
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

        kv_cache_dtype = get_kv_cache_torch_dtype(
            self.runner.kv_cache_dtype, self.runner.model_config.dtype)
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
            num_qo_heads=self.runner.model_config.get_num_attention_heads(
                self.runner.parallel_config),
            num_kv_heads=self.runner.model_config.get_num_kv_heads(
                self.runner.parallel_config),
            head_dim=self.runner.model_config.get_head_size(),
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
        logits_soft_cap: Optional[float] = None,
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
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: FlashInferMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in FlashInfer.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferImpl")
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
                k_scale,
                v_scale,
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
                output = prefill_meta.prefill_wrapper.forward(
                    query,
                    kv_cache,
                    logits_soft_cap=self.logits_soft_cap,
                    causal=True)
        else:
            assert attn_metadata.decode_metadata is not None
            assert attn_metadata.decode_metadata.decode_wrapper is not None
            output = attn_metadata.decode_metadata.decode_wrapper.forward(
                query,
                kv_cache,
                sm_scale=self.scale,
                logits_soft_cap=self.logits_soft_cap)
        return output.view(num_tokens, hidden_size)
