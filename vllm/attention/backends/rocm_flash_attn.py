"""Attention layer ROCm GPUs."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           is_block_tables_empty)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger
from vllm.sequence import SequenceGroupMetadata
from vllm.utils import make_tensor_with_pad

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUBuilder


class ROCmFlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "rocm-flash-attn"

    @staticmethod
    def get_impl_cls() -> Type["ROCmFlashAttentionImpl"]:
        return ROCmFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return ROCmFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["ROCmFlashAttentionMetadataBuilder"]:
        return ROCmFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

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


@dataclass
class ROCmFlashAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

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

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]
    _cached_prefill_metadata: Optional["ROCmFlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["ROCmFlashAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["ROCmFlashAttentionMetadata"]:
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

        self._cached_prefill_metadata = ROCmFlashAttentionMetadata(
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
    def decode_metadata(self) -> Optional["ROCmFlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = ROCmFlashAttentionMetadata(
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


class ROCmFlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[ROCmFlashAttentionMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.use_v2_block_manager = (
            input_builder.scheduler_config.use_v2_block_manager)

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
        self.context_lens.append(context_len)

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

    def add_decode_seq_group(self, seq_group_metadata: SequenceGroupMetadata,
                             seq_id, seq_len, query_len, context_len,
                             curr_sliding_window_blocks, sliding_seq_len,
                             sliding_context_len):

        # Compute block table.
        if seq_group_metadata.block_tables is not None:
            block_table = seq_group_metadata.block_tables[seq_id]
            if curr_sliding_window_blocks is not None:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_table[-curr_sliding_window_blocks:]
        else:
            # Only happens when memory profiling runs.
            block_table = []

        self.block_tables.append(block_table)
        self.context_lens.append(sliding_context_len)

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

    def build(self, model_config, parallel_config, kv_cache_dtype, seq_lens,
              query_lens, decode_seq_lens, use_captured_graph: bool,
              cuda_graph_pad_size: int, graph_block_tables: np.ndarray,
              batch_size: int, device):
        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(decode_seq_lens, default=0)
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

        context_lens_tensor = torch.tensor(self.context_lens,
                                           dtype=torch.int,
                                           device=device)
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

        return ROCmFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )


class ROCmFlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prompt_tokens -------------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|

    Otherwise, the layout is as follows:
    |<------------------ num_generation_tokens (M) ----------------->|
    |<--generation_0-->|..........|<--generation_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens ----------->|	
    |<-prompt_0->|...|<-prompt_N-1->|<-generation_0->|...|<-generation_M-1->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
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
    ) -> None:
        assert blocksparse_params is None, ValueError(
            "ROCFlashAttention does not support blocksparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")

        self.use_naive_attn = False
        # NOTE: Allow for switching between Triton and CK. Defaulting to triton.
        self.use_triton_flash_attn = envs.VLLM_USE_TRITON_FLASH_ATTN
        if self.use_triton_flash_attn:
            from vllm.attention.ops.triton_flash_attention import (  # noqa: F401
                triton_attention)
            self.attn_func = triton_attention
            logger.debug("Using Triton FA in ROCmBackend")
        else:
            # if not using triton, navi3x/navi21/navi10 do not use flash-attn
            # either
            if torch.cuda.get_device_capability()[0] != 9:
                self.use_naive_attn = True
            else:
                try:
                    from flash_attn import flash_attn_varlen_func  # noqa: F401
                    self.attn_func = flash_attn_varlen_func
                    logger.debug("Using CK FA in ROCmBackend")
                except ModuleNotFoundError:
                    self.use_naive_attn = True

            if self.use_naive_attn:
                self.attn_func = _sdpa_attention
                logger.debug("Using naive attention in ROCmBackend")

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
        tokens, n_kv_heads, head_dim = x.shape
        return (x[:, :,
                  None, :].expand(tokens, n_kv_heads, n_rep,
                                  head_dim).reshape(tokens, n_kv_heads * n_rep,
                                                    head_dim))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ROCmFlashAttentionMetadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                kv_scale,
            )

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            assert prefill_meta.seq_lens is not None
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # triton attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                if self.use_triton_flash_attn:
                    out, _ = self.attn_func(
                        query,
                        key,
                        value,
                        None,
                        prefill_meta.seq_start_loc,
                        prefill_meta.seq_start_loc,
                        prefill_meta.max_prefill_seq_len,
                        prefill_meta.max_prefill_seq_len,
                        True,
                        self.scale,
                    )
                elif self.use_naive_attn:
                    if self.num_kv_heads != self.num_heads:
                        # Interleave for MQA workaround.
                        key = self.repeat_kv(key, self.num_queries_per_kv)
                        value = self.repeat_kv(value, self.num_queries_per_kv)
                    query = query.movedim(0, query.dim() - 2)
                    key = key.movedim(0, key.dim() - 2)
                    value = value.movedim(0, value.dim() - 2)
                    # sdpa math backend attention
                    out = self.attn_func(
                        query,
                        key,
                        value,
                        prefill_meta.seq_lens,
                        num_tokens,
                        self.num_heads,
                        self.head_size,
                        self.scale,
                    )
                else:
                    out = self.attn_func(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=prefill_meta.seq_start_loc,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_q=prefill_meta.max_prefill_seq_len,
                        max_seqlen_k=prefill_meta.max_prefill_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                    )

                # common code for prefill
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                output[:num_prefill_tokens] = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.context_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window[0],
                )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                decode_meta.block_tables,
                decode_meta.seq_lens_tensor,
                decode_meta.max_decode_seq_len,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
            )

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)


def _sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: List[int],
    num_tokens: int,
    num_heads: int,
    head_size: int,
    scale: float,
) -> torch.Tensor:
    start = 0
    output = torch.empty((num_tokens, num_heads, head_size),
                         dtype=query.dtype,
                         device=query.device)

    for seq_len in seq_lens:
        end = start + seq_len
        with torch.backends.cuda.sdp_kernel(enable_math=True,
                                            enable_flash=False,
                                            enable_mem_efficient=False):
            sub_out = torch.nn.functional.scaled_dot_product_attention(
                query[:, start:end, :],
                key[:, start:end, :],
                value[:, start:end, :],
                dropout_p=0.0,
                is_causal=True,
                scale=scale).movedim(query.dim() - 2, 0)
            output[start:end, :, :] = sub_out
            start = end

    return output
