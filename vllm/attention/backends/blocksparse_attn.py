from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
from vllm.attention.ops.blocksparse_attention.interface import (
    LocalStridedBlockSparseAttn, get_head_sliding_step)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
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
    def get_impl_cls() -> Type["BlocksparseFlashAttentionImpl"]:
        return BlocksparseFlashAttentionImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "BlocksparseFlashAttentionMetadata":
        return BlocksparseFlashAttentionMetadata(*args, **kwargs)

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
class BlocksparseFlashAttentionMetadata(AttentionMetadata,
                                        PagedAttentionMetadata):
    """A copy of Metadata for FlashAttentionBackend,
    to avoid having to install flash_attn.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The prompt length per sequence. None if it is a decoding.
    prompt_lens: Optional[List[int]]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]
    # The number of prompt tokens. Doesn't include padding.
    num_prompt_tokens: int
    # The number of generation tokens. Doesn't include padding.
    num_generation_tokens: int

    # NOTE(sang): Definition of seq_len, subquery_len, and seqlen.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- seq_len ----------|
    # |-------------------- seqlen ----------------------|
    #                                   |- subquery_len -|

    # WARNING(sang): seq_len has different definition depending on if it is
    # prefill vs decoding. When it is prefill, it doesn't include new tokens.
    # When it is for decoding, it includes a new token.

    # Maximum subquery length in the batch.
    max_subquery_len: Optional[int]
    # Maximum prompt length in the batch.
    max_prompt_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool


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
    ) -> None:
        assert blocksparse_params is not None
        assert alibi_slopes is None, ValueError(
            "Alibi not support for blocksparse flash attention.")
        assert sliding_window is None, ValueError(
            "sliding_window is invalid for blocksparse attention.")

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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata[BlocksparseFlashAttentionMetadata],
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

        if prefill_meta := attn_metadata.prefill_metadata:

            # Prompt run.
            # normal attention
            # When block_tables are not filled, it means q and k are the
            # prompt, and they have the same length.

            assert kv_cache is None or prefill_meta.block_tables.numel() == 0,\
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
                decode_meta.max_seq_len,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
                tp_rank=self.tp_rank,
                blocksparse_local_blocks=self.local_blocks,
                blocksparse_vert_stride=self.vert_stride,
                blocksparse_block_size=self.sparse_block_size,
                blocksparse_head_sliding_step=self.head_sliding_step,
            )

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)
