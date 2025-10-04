# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with XFormersAttention."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (get_context_parallel_rank,
                                             get_context_parallel_world_size,
                                             get_cp_group)
from vllm.logger import init_logger
from vllm.v1.attention.backends.cp_utils import (cp_get_neighbor_ranks,
                                                 cp_pass_around)
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder, CommonAttentionMetadata,
    reorder_batch_to_split_decodes_and_prefills, split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec

try:
    from xformers import ops as xops
    from xformers.ops.fmha.attn_bias import (
        AttentionBias, BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalGappyKeysMask,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask)

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm import _custom_ops as ops

logger = init_logger(__name__)


class XFormersAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
            136,
            144,
            152,
            160,
            168,
            176,
            184,
            192,
            200,
            208,
            216,
            224,
            232,
            240,
            248,
            256,
        ]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes.")

    @staticmethod
    def get_name() -> str:
        return "XFORMERS"

    @staticmethod
    def get_impl_cls() -> type["XFormersAttentionImpl"]:
        return XFormersAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return XFormersAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["XFormersAttentionMetadataBuilder"]:
        return XFormersAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class XFormersAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # Biases for different attention types.
    attn_bias: Optional["AttentionBias"] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["XFormersAttentionMetadata"] = None
    _cached_decode_metadata: Optional["XFormersAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["XFormersAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        q_start_loc = self.query_start_loc[self.num_decodes:]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[self.num_decodes:]
        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = XFormersAttentionMetadata(
            num_actual_tokens=self.num_prefill_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc - q_start_loc[0],
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_table=self.block_table[self.num_decodes:],
            slot_mapping=self.slot_mapping[self.num_decode_tokens:],
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["XFormersAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata

        q_start_loc = self.query_start_loc
        q_seqlens = torch.diff(q_start_loc)
        decode_kv_seqlens = self.seq_lens[:self.num_decodes]
        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = XFormersAttentionMetadata(
            num_actual_tokens=self.num_decode_tokens,
            max_query_len=int(q_seqlens[:self.num_decodes].max().item()),
            query_start_loc=q_start_loc[:self.num_decodes + 1],
            max_seq_len=int(decode_kv_seqlens.max().item()),
            seq_lens=decode_kv_seqlens,
            block_table=self.block_table[:self.num_decodes],
            slot_mapping=self.slot_mapping[:self.num_decode_tokens],
            attn_bias=self.attn_bias,
        )
        return self._cached_decode_metadata


class XFormersAttentionMetadataBuilder(
        AttentionMetadataBuilder[XFormersAttentionMetadata]):

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        assert XFORMERS_AVAILABLE
        self.block_size = kv_cache_spec.block_size
        self._num_decodes = 0
        self._num_decode_tokens = 0

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return reorder_batch_to_split_decodes_and_prefills(
            input_batch,
            scheduler_output,
            decode_threshold=self.reorder_batch_threshold)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XFormersAttentionMetadata:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold))

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        q_seqlens = torch.diff(q_start_loc)
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = common_attn_metadata.max_seq_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        bias = None
        if num_decodes > 0:
            # Construct the decoder bias.
            decode_q_seqlens = q_seqlens[:num_decodes]
            decode_kv_seqlens = kv_seqlens[:num_decodes]
            bias = (
                PagedBlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                    q_seqlen=decode_q_seqlens.tolist(),
                    kv_seqlen=decode_kv_seqlens.tolist(),
                    page_size=self.block_size,
                    block_tables=block_table[:num_decodes],
                    device=block_table.device,
                ))

        return XFormersAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            max_query_len=max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=kv_seqlens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            attn_bias=bias,
        )


def _get_decode_attn_bias(
    q_seqlens: list[int],
    kv_seqlens: list[int],
    current_kv_rank: torch.Tensor,
) -> list[BlockDiagonalGappyKeysMask]:
    """
    Generate attention bias masks for decode phase in context parallel.

    This function creates attention masks that allow queries to attend to
    KV cache distributed across different CP ranks. Each sequence's KV cache
    is owned by a specific rank, and we need to create appropriate masks for
    cross-rank attention.

    Example:
        If we have 2 CP ranks and 3 sequences:
        - q_seqlens = [1, 1, 1]  # Each decode query has length 1
        - kv_seqlens = [10, 15, 8]  # KV cache lengths for each sequence
        - current_kv_rank = [0, 1, 0]  # Which rank owns the CURRENT token's
          KV pair

        For src_rank=0: mask=[0, -1, 0] -> adjusted_kv_seqlens=[10, 14, 8]
        For src_rank=1: mask=[-1, 0, -1] -> adjusted_kv_seqlens=[9, 15, 7]

        This creates masks where sequences not owned by src_rank have reduced
        length, effectively masking out the last token position.

    Args:
        q_seqlens: Query sequence lengths for each sequence in the batch
        kv_seqlens: Key-value cache lengths for each sequence
        current_kv_rank: Tensor indicating which CP rank owns each sequence's
            KV cache

    Returns:
        List of BlockDiagonalGappyKeysMask objects, one for each source rank
    """
    assert current_kv_rank.shape[0] == len(kv_seqlens)
    cp_size = get_context_parallel_world_size()

    cp_pass_q_attn_bias = []
    for src_rank in range(cp_size):
        # Create mask: 0 for sequences owned by src_rank, -1 for others
        # This effectively reduces KV length by 1 for non-owned sequences
        mask = torch.full((len(kv_seqlens), ), -1, dtype=torch.int32)
        mask[current_kv_rank == src_rank] = 0

        # Adjust KV sequence lengths based on ownership
        # Sequences not owned by src_rank get length reduced by 1
        adjusted_kv_seqlens = [
            kv_len + mask_val
            for kv_len, mask_val in zip(kv_seqlens, mask.tolist())
        ]

        # Calculate cumulative start positions for each sequence
        kv_seqstarts = [0]
        for kv_len in adjusted_kv_seqlens:
            kv_seqstarts.append(kv_seqstarts[-1] + kv_len)

        # Create block diagonal mask with gaps for this source rank
        cp_pass_q_attn_bias.append(
            BlockDiagonalGappyKeysMask.from_seqlens(
                q_seqlen=q_seqlens,
                kv_seqlen=adjusted_kv_seqlens,
                kv_seqstarts=kv_seqstarts,
            ))
    return cp_pass_q_attn_bias


def _get_prefill_attn_bias(
    cp_sharded_q_seqlen: list[list[int]],
    cp_sharded_pass_x_kvlens_per_rank: list[list[list[int]]],
) -> list[BlockDiagonalGappyKeysMask]:
    """
    Generate attention bias masks for prefill phase in context parallel.

    This function creates attention masks for distributed prefill computation
    where queries and KV cache are sharded across multiple CP ranks. It handles
    both causal masking (for local rank) and block diagonal masking (for remote
    ranks).

    Example with 2 CP ranks and 2 requests:
        cp_sharded_q_seqlen = [[4, 6], [3, 5]]  # Sharded query lengths per req
        cp_sharded_pass_x_kvlens_per_rank = [
            [[8, 12], [6, 10]],  # KV lengths for rank 0
            [[7, 11], [5, 9]]    # KV lengths for rank 1
        ]

        For rank 0 (cp_rank=0):
        - Uses BlockDiagonalCausalWithOffsetGappyKeysMask for local data
        - Uses BlockDiagonalGappyKeysMask for remote data

        For rank 1 (cp_rank=1):
        - Uses BlockDiagonalGappyKeysMask for remote data
        - Uses BlockDiagonalCausalWithOffsetGappyKeysMask for local data

    Args:
        cp_sharded_q_seqlen: Query sequence lengths [request][cp_shard]
        cp_sharded_pass_x_kvlens_per_rank: KV lengths
            [src_rank][request][cp_shard]

    Returns:
        List of attention bias masks, one for each source rank
    """
    cp_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank()

    def flatten(kv_seqlens: list[list[int]]) -> list[int]:
        """Flatten nested list structure into single list."""
        return [item for sublist in kv_seqlens for item in sublist]

    # Flatten query sequence lengths across all ranks and sequences
    cp_sharded_q_seqlen_flatten = flatten(cp_sharded_q_seqlen)

    # Determine bias type for each source rank:
    # - Causal mask for local rank (allows attending to past and current
    #   tokens)
    # - Block diagonal mask for remote ranks (allows attending to all tokens
    #   in block)
    # TODO: use PagedBlockDiagonalCausalWithOffsetGappyKeysMask for local
    #   attention
    bias_type = [(BlockDiagonalCausalWithOffsetGappyKeysMask
                  if cp_rank == i else BlockDiagonalGappyKeysMask)
                 for i in range(cp_size)]

    def get_kv_seqstarts(kv_seqlen: list[int]) -> list[int]:
        """
        Calculate starting positions for KV sequences in attention
        computation.

        Processes pairs of KV lengths to determine where each sequence
        block starts.
        Example: kv_seqlen=[8, 12, 6, 10] ->
        kv_seqstarts=[0, 0, 12, 12, 22]
        """
        kv_seqstarts = [0]
        for i in range(0, len(kv_seqlen), 2):
            second = kv_seqlen[i + 1] if i + 1 < len(kv_seqlen) else 0
            kv_seqstarts.append(kv_seqstarts[-1])
            kv_seqstarts.append(kv_seqstarts[-1] + second)
        return kv_seqstarts

    # Create attention bias mask for each source rank
    cp_pass_kv_attn_bias = [
        bias_type[cp_src_rank].from_seqlens(
            q_seqlen=cp_sharded_q_seqlen_flatten,
            kv_seqlen=flatten(cp_sharded_pass_x_kvlens_per_rank[cp_src_rank]),
            kv_seqstarts=get_kv_seqstarts(
                flatten(cp_sharded_pass_x_kvlens_per_rank[cp_src_rank])),
        ) for cp_src_rank in range(cp_size)
    ]

    return cp_pass_kv_attn_bias


def _cp_partial_prefill_get_kv_seqlens(
    cp_shard_sizes_all: list[list[int]],
    num_computed_tokens: int,
) -> list[list[int]]:
    # For prefill by passing KV among CP group, get
    # the KV seqlens (part of the attention bias) for computing partial attn
    # on KV received from each CP rank.
    cp_world_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank()

    cp_sharded_pass_kv_kvlens = []
    for src_rank in range(cp_world_size):
        # src_rank is the rank that pass kv or q from
        cp_shard_sizes = cp_shard_sizes_all[src_rank]
        assert len(cp_shard_sizes) == 2
        if src_rank == cp_rank:
            """
            When local q is to attent with local kv cache,
            it's always causual mask and the shape is always like this:
            |__\
            |__|__\
            """
            cp_sharded_pass_kv_kvlens.append([
                num_computed_tokens + cp_shard_sizes[0],
                num_computed_tokens + sum(cp_shard_sizes)
            ])
        elif src_rank > cp_rank:
            """
            when src_rank > cp_rank, it's always block diagonal mask.

            When we pass kv, the shape is always horizontal:
            |__|__|
            """
            cp_sharded_pass_kv_kvlens.append([
                num_computed_tokens, num_computed_tokens + sum(cp_shard_sizes)
            ])
        else:
            """
            when src_rank < cp_rank, it's always block diagonal mask.

            When we pass kv, the shape is always vertical:
            |__|
            |__|
            """
            cp_sharded_pass_kv_kvlens.append([
                num_computed_tokens + cp_shard_sizes[0],
                num_computed_tokens + cp_shard_sizes[0]
            ])

    return cp_sharded_pass_kv_kvlens


def _merge_attn_flash_partial(
    attn_out: list[torch.Tensor],
    attn_lse: list[torch.Tensor],
) -> torch.Tensor:
    # merges partial attention outputs from flash varseq fwd to final output
    assert len(attn_out) == len(attn_lse)
    assert len(attn_out) >= 1

    if len(attn_out) == 1:
        return attn_out[0]

    M, H, Kq = attn_out[0].shape
    attn_out_t = [x.view(1, M, 1, H, Kq) for x in attn_out]
    lse_out_t = [x.view(1, 1, H, M) for x in attn_lse]
    return xops.fmha.merge_attentions(
        attn_out_t,
        lse_out_t,
        write_lse=False,
    )[0]


def _prefill_pass_kv_attention(
        cp_world_size: int,
        cp_rank: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        xq_out: torch.Tensor,
        slot_mapping: torch.Tensor,
        B_T: int,
        N_H_L: int,
        D_H: int,
        attn_bias: list[BlockDiagonalGappyKeysMask],  # type: ignore
) -> torch.Tensor:
    """
    Computes attention for fused varseq prompt by passing KV among CP group for
    best overlap between CP comms and attention compute. KV from different
    prefill batches are padded to the maximum seqlen in the fused prefill.

    Args:
        max_global_kvlen: maximum seqlen in current batch, used for pass_kv
            only
        prefetched_lengths: indicates the starting position of cache, used for
            duplicate_kv with persistent cache enabled
        varseq_batch_dedup: batch indices of the current batch.
        varseq_seqlen: padded seqlen after cp sharding
    """

    assert XFORMERS_AVAILABLE
    # TODO: extract KV pieces after local attention
    cache_k_ = torch.index_select(cache_k, 1, slot_mapping)
    cache_v_ = torch.index_select(cache_v, 1, slot_mapping)
    cache_k_self, cache_v_self = (t.view(1, -1, N_H_L, D_H)
                                  for t in (cache_k_, cache_v_))

    src_rank = torch.tensor(
        cp_rank,
        device=torch.cuda.current_device(),
        dtype=torch.int32,
    )
    to_rank, from_rank = cp_get_neighbor_ranks()

    next_tensors, reqs = cp_pass_around([cache_k_, cache_v_, src_rank],
                                        to_rank, from_rank)
    # local partial attn
    attn_out_self, lse_out_self = xops.fmha.memory_efficient_attention_partial(  # type: ignore
        xq_out,
        cache_k_self,
        cache_v_self,
        attn_bias[cp_rank],
    )
    attn_out_self = attn_out_self.squeeze(0)

    attn_out, lse_out = [attn_out_self], [lse_out_self]

    for i in range(1, cp_world_size):
        cache_k_i, cache_v_i, src_rank_i_t = next_tensors
        for req in reqs:
            req.wait()

        src_rank_i = int(src_rank_i_t.item())

        if i < cp_world_size - 1:
            next_tensors, reqs = cp_pass_around(
                [cache_k_i, cache_v_i, src_rank_i_t], to_rank, from_rank)

        cache_k_i_, cache_v_i_ = (t.view(1, -1, N_H_L, D_H)
                                  for t in (cache_k_i, cache_v_i))

        attn_out_i, lse_out_i = xops.fmha.memory_efficient_attention_partial(  # type: ignore
            xq_out,
            cache_k_i_,
            cache_v_i_,
            attn_bias[src_rank_i],
        )
        attn_out_i = attn_out_i.squeeze(0)
        attn_out.append(attn_out_i)
        lse_out.append(lse_out_i)

    merged_out = _merge_attn_flash_partial(attn_out, lse_out)

    return merged_out.view(1, B_T, N_H_L * D_H)


def _decode_allgather_attention(
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        xq_out: torch.Tensor,
        slot_mapping: torch.Tensor,
        B_T: int,
        N_H_L: int,
        D_H: int,
        attn_bias: list[BlockDiagonalGappyKeysMask],  # type: ignore
) -> torch.Tensor:
    """
    Supports CP decode by allgather partial attention among CP ranks.
    This function distributes attention computation across multiple CP ranks by:
    1. Each CP rank computes partial attention: Attn(local_Q, local_KV)
    2. All ranks gather partial attention outputs and log-sum-exp values via
       allgather
    3. Merges all partial attention results to produce final attention output

    Returns:
        Merged attention output tensor [1, B_T, N_H_L * D_H]
    """

    assert XFORMERS_AVAILABLE
    cp_rank = get_context_parallel_rank()

    cache_k_ = torch.index_select(cache_k, 1,
                                  slot_mapping).view(1, -1, N_H_L, D_H)
    cache_v_ = torch.index_select(cache_v, 1,
                                  slot_mapping).view(1, -1, N_H_L, D_H)

    xq_out = xq_out.view(1, B_T, N_H_L, D_H)

    attn_out_ = xops.fmha.memory_efficient_attention_partial(  # type: ignore
        xq_out,
        cache_k_,
        cache_v_,
        attn_bias[cp_rank],
    )

    attn_out = get_cp_group().all_gather(attn_out_[0], dim=0)
    lse_out = get_cp_group().all_gather(attn_out_[1], dim=0)
    merged_out = _merge_attn_flash_partial(list(attn_out.unbind()),
                                           list(lse_out.unbind()))

    return merged_out.view(1, B_T, N_H_L * D_H)


class XFormersAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in V0.")
        if alibi_slopes is not None:
            raise NotImplementedError(
                "XFormers does not support alibi slopes yet.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        if logits_soft_cap is None:
            # Setting logits_soft_cap to 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        XFormersAttentionBackend.validate_head_size(head_size)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "XFormersAttentionImpl.")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XFormersAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with XFormers.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for XFormersAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

        # Cache the input KVs.
        key_cache, value_cache = kv_cache.unbind(0)
        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        if prefill_meta := attn_metadata.prefill_metadata:
            descale_shape = (prefill_meta.query_start_loc.shape[0] - 1,
                             key.shape[1])
            unified_attention(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                seqused_k=prefill_meta.seq_lens,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=prefill_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )

        if decode_meta := attn_metadata.decode_metadata:
            # Query for decode. KV is not needed because it is already cached.
            decode_query = query[:num_decode_tokens]
            # Reshape query to [1, B_T, G, H, D].
            q = decode_query.view(1, -1, self.num_kv_heads,
                                  self.num_queries_per_kv, self.head_size)
            # Reshape the k and v caches to [1, Bkv_T, G, H, D]
            cache_k = key_cache.view(1, -1, self.num_kv_heads, 1,
                                     self.head_size).expand(
                                         1,
                                         -1,
                                         self.num_kv_heads,
                                         self.num_queries_per_kv,
                                         self.head_size,
                                     )
            cache_v = value_cache.view(1, -1, self.num_kv_heads, 1,
                                       self.head_size).expand(
                                           1,
                                           -1,
                                           self.num_kv_heads,
                                           self.num_queries_per_kv,
                                           self.head_size,
                                       )

            attn_bias = decode_meta.attn_bias
            output[:
                   num_decode_tokens] = xops.memory_efficient_attention_forward(
                       q,
                       cache_k,
                       cache_v,
                       attn_bias=attn_bias,
                       p=0.0,
                       scale=self.scale,
                   ).view(decode_query.shape)

        # Reshape the output tensor.
        return output
