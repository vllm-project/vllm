# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.ops.blocksparse_attention.interface import (
    LocalStridedBlockSparseAttn, get_head_sliding_step)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)


@dataclass
class BlocksparseParams:
    max_seqlen: int

    # Num q heads per tensor-parallel rank/partition
    num_heads: int  # per TP partition
    # Num kv heads per tensor-parallel rank/partition
    num_kv_heads: int

    # block size used for blocksparse attention.
    # This is the block_size used in `local_blocks`, `vert_stride`.
    block_size: int

    # Number of blocks for local attention, i.e., number of
    # local attended tokens / `sparse_block_size`
    local_blocks: int

    # Attend to one block per every `vert_stride` blocks.
    # Controlling the sparsity
    vert_stride: int
    """
    If to use the same vertical stride offset for all heads, 
    i.e., attend to the same block of tokens on all heads.
    By default, it is False, i.e., attention on the non-local 
    blocks depends on the `head_idx`, that is on
    blocks satisfying 
    `(block_idx + head_idx * head_sliding_step + 1) % vert_stride == 0`
    where `head_sliding_step=max(1, int(vert_stride / num_total_heads))`,
            `block_idx = position_id // sparse_block_size`.
    See `..ops.blocksparse_attention.utils:get_sparse_attn_mask`
    for more detail.
    """
    homo_head: bool = False

    # If within a group, the kv offsets that each q attends is the same or no.
    homo_head_group: bool = False

    # Decided by homo_head and homo_head group
    head_sliding_step: int = field(init=False)

    # range of q heads to for a TP rank
    active_head_range: Tuple = field(init=False)

    def __post_init__(self):
        assert self.block_size > 0
        assert self.local_blocks >= 0
        assert self.vert_stride >= 1
        assert self.num_heads % self.num_kv_heads == 0

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        total_heads = tp_size * self.num_heads
        total_kv_heads = tp_size * self.num_kv_heads

        if self.homo_head:
            self.head_sliding_step = 0
        elif self.homo_head_group:
            head_sliding_step = get_head_sliding_step(total_kv_heads,
                                                      self.vert_stride)
            # negative indicates sliding along kv heads, i.e., homo q group
            self.head_sliding_step = -head_sliding_step
        else:
            self.head_sliding_step = get_head_sliding_step(
                total_heads, self.vert_stride)

        self.active_head_range = (
            tp_rank * self.num_heads,
            (tp_rank + 1) * self.num_heads,
        )


class BlocksparseFlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "BLOCK_SPARSE_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["BlocksparseFlashAttentionImpl"]:
        return BlocksparseFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return BlocksparseFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["BlocksparseFlashAttentionMetadataBuilder"]:
        return BlocksparseFlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

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
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class BlocksparseFlashAttentionMetadata(AttentionMetadata):
    """A copy of Metadata for FlashAttentionBackend,
    to avoid having to install flash_attn.

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

    # Max number of query tokens for among request in the batch.
    max_decode_query_len: Optional[int] = None

    _cached_prefill_metadata: Optional[
        "BlocksparseFlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional[
        "BlocksparseFlashAttentionMetadata"] = None

    @property
    def prefill_metadata(
            self) -> Optional["BlocksparseFlashAttentionMetadata"]:
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

        self._cached_prefill_metadata = BlocksparseFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
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
    def decode_metadata(self) -> Optional["BlocksparseFlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = BlocksparseFlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
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


class BlocksparseFlashAttentionMetadataBuilder(
        CommonMetadataBuilder[BlocksparseFlashAttentionMetadata]):

    _metadata_cls = BlocksparseFlashAttentionMetadata


class BlocksparseFlashAttentionImpl(AttentionImpl):
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
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        assert blocksparse_params is not None
        assert alibi_slopes is None, ValueError(
            "Alibi not support for blocksparse flash attention.")
        assert sliding_window is None, ValueError(
            "sliding_window is invalid for blocksparse attention.")
        assert logits_soft_cap is None, ValueError(
            "logits_soft_cap is invalid for blocksparse attention.")

        if "num_heads" not in blocksparse_params:
            blocksparse_params["num_heads"] = num_heads
        if "num_kv_heads" not in blocksparse_params:
            blocksparse_params["num_kv_heads"] = num_kv_heads or num_heads
        self.blocksparse_params = BlocksparseParams(**blocksparse_params)
        self.kv_cache_dtype = kv_cache_dtype

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.alibi_slopes = alibi_slopes
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.local_blocks = self.blocksparse_params.local_blocks
        self.vert_stride = self.blocksparse_params.vert_stride
        self.sparse_block_size = self.blocksparse_params.block_size
        self.head_sliding_step = self.blocksparse_params.head_sliding_step

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        total_num_heads = num_heads * self.tp_size
        self.bs_attn = LocalStridedBlockSparseAttn(
            total_num_heads,
            self.blocksparse_params.max_seqlen,
            self.blocksparse_params.local_blocks,
            self.blocksparse_params.vert_stride,
            self.blocksparse_params.block_size,
            homo_head=self.blocksparse_params.homo_head,
            active_head_range=self.blocksparse_params.active_head_range,
        )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "BlocksparseFlashAttentionImpl")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BlocksparseFlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache.numel() > 0:
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
                layer._k_scale,
                layer._v_scale,
            )

        if prefill_meta := attn_metadata.prefill_metadata:

            # Prompt run.
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.

            assert kv_cache.numel() == 0 \
                    or prefill_meta.block_tables is None \
                    or prefill_meta.block_tables.numel() == 0, \
                "Does not support prefix-enabled attention."

            output = self.bs_attn(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=prefill_meta.seq_start_loc,
                cu_seqlens_k=prefill_meta.seq_start_loc,
                sm_scale=self.scale,
            )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output = PagedAttention.forward_decode(
                query,
                key_cache,
                value_cache,
                decode_meta.block_tables,
                decode_meta.seq_lens_tensor,
                self.blocksparse_params.max_seqlen,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                layer._k_scale,
                layer._v_scale,
                tp_rank=self.tp_rank,
                blocksparse_local_blocks=self.local_blocks,
                blocksparse_vert_stride=self.vert_stride,
                blocksparse_block_size=self.sparse_block_size,
                blocksparse_head_sliding_step=self.head_sliding_step,
            )

        assert output is not None
        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)
