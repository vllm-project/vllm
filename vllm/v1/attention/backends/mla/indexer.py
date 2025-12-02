# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    MultipleOf,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata, is_deep_gemm_supported
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)

logger = init_logger(__name__)


class DeepseekV32IndexerBackend(AttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1 if current_platform.is_rocm() else 64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor


@dataclass
class DeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    # The dimension of the attention heads
    head_dim: int

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


# TODO (zyongye) optimize this, this is now vibe coded
def kv_spans_from_batches(
    start_seq_loc: torch.Tensor, seq_len_per_batch: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      start_seq_loc: 1D long tensor [B+1], cumulative counts of
                     selected tokens per batch.
            Example: [0, 2, 4, 7] ->
                     batch sizes (selected) [2, 2, 3], N=7 tokens total.
      seq_len_per_batch: 1D long tensor [B],
                         full sequence length (KV length) of each batch.
                         Example: [5, 9, 4].

    Returns:
      start_tensor: 1D long tensor [N], start offset in the
                    concatenated KV cache for each token's batch.
      end_location: 1D long tensor [N],
                    **exclusive** end = start + token's local position.
                    (So the attended KV slice is kv[start:end].)

    Assumes each batch contributes its full `seq_len_per_batch[i]`
    keys to the KV cache, andthe selected tokens within a batch
    are the **last** `counts[i]` positions of that sequence.
    """
    q = start_seq_loc.to(dtype=torch.long)
    L = seq_len_per_batch.to(dtype=torch.long)
    assert q.dim() == 1 and L.dim() == 1
    assert q.numel() == L.numel() + 1, "start_seq_loc must have length B+1"

    # Selected tokens per batch and totals
    counts = q[1:] - q[:-1]  # [B]
    N = int(q[-1].item())  # total selected tokens
    B = L.numel()

    if N == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    # KV start offsets per batch in the concatenated KV cache
    kv_starts_per_batch = torch.cumsum(L, dim=0) - L  # [B]

    # For each selected token, which batch does it belong to?
    batch_id = torch.repeat_interleave(torch.arange(B), counts)  # [N]

    # Map batch KV start to each token
    start_tensor = kv_starts_per_batch[batch_id]  # [N]

    # End-align local positions inside each batch:
    # local_pos = L[b] - counts[b] + (1..counts[b])  for each batch b
    L_expand = torch.repeat_interleave(L, counts)  # [N]
    m_expand = torch.repeat_interleave(counts, counts)  # [N]
    # position within the selected block: 1..counts[b]
    pos_within = (
        torch.arange(N, dtype=torch.long) - torch.repeat_interleave(q[:-1], counts) + 1
    )

    local_pos = L_expand - m_expand + pos_within  # [N], 1-based
    end_location = start_tensor + local_pos  # exclusive end

    return start_tensor.int().to(device), end_location.int().to(device)


def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 2 is a magic number for controlling the prefill buffer size.
    # May be tuned later.
    return max_model_len * 2


def split_prefill_chunks(
    seq_lens_cpu: torch.Tensor, max_prefill_buffer_size: int, reqs_start: int
) -> list[tuple[int, int]]:
    """
    Split the prefill chunks into a list of tuples of (reqs_start, reqs_end)
    such that the total sequence length of each chunk is less than the
    maximum prefill buffer size.

    Args:
        seq_lens_cpu: The sequence lengths of the prefill requests.
        max_prefill_buffer_size: The maximum prefill buffer size.
        reqs_start: The start index of the prefill requests.

    Returns:
        A list of tuples of (reqs_start, reqs_end).
    """
    chunk_seq_ids = []
    total_seq_lens = 0
    for i in range(reqs_start, len(seq_lens_cpu)):
        cur_seq_len = seq_lens_cpu[i].item()
        assert cur_seq_len <= max_prefill_buffer_size
        total_seq_lens += cur_seq_len
        if total_seq_lens > max_prefill_buffer_size:
            chunk_seq_ids.append((reqs_start, i))
            reqs_start = i
            total_seq_lens = cur_seq_len
    if total_seq_lens > 0:
        chunk_seq_ids.append((reqs_start, len(seq_lens_cpu)))
    return chunk_seq_ids


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    reorder_batch_threshold: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        # Now deepgemm fp8_paged_mqa_logits does not support next_n > 2
        self.reorder_batch_threshold += min(self.num_speculative_tokens, 1)

        props = torch.cuda.get_device_properties(self.device)
        sm_count = props.multi_processor_count
        self.num_sms = sm_count

        self.decode_lens_buffer = torch.empty(
            (scheduler_config.max_num_seqs,), dtype=torch.int32, device=self.device
        )

        # See: DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )

    def build_one_prefill_chunk(
        self, reqs_start, reqs_end, query_start_loc_cpu, seq_lens_cpu, block_table
    ):
        prefill_query_start_loc = (
            query_start_loc_cpu[reqs_start : reqs_end + 1]
            - query_start_loc_cpu[reqs_start]
        )
        cu_seqlen_ks, cu_seqlen_ke = kv_spans_from_batches(
            prefill_query_start_loc, seq_lens_cpu[reqs_start:reqs_end], self.device
        )
        token_start = query_start_loc_cpu[reqs_start].item()
        token_end = query_start_loc_cpu[reqs_end].item()
        total_seq_lens = seq_lens_cpu[reqs_start:reqs_end].sum()
        assert total_seq_lens <= self.max_prefill_buffer_size
        cu_seq_lens = (
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32),
                    seq_lens_cpu[reqs_start:reqs_end].cumsum(dim=0),
                ]
            )
            .to(torch.int32)
            .to(self.device)
        )
        return DeepseekV32IndexerPrefillChunkMetadata(
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            cu_seq_lens=cu_seq_lens,
            total_seq_lens=total_seq_lens,
            block_table=block_table[reqs_start:reqs_end],
            token_start=token_start,
            token_end=token_end,
            num_reqs=reqs_end - reqs_start,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            chunk_seq_ids = split_prefill_chunks(
                common_attn_metadata.seq_lens_cpu,
                self.max_prefill_buffer_size,
                num_decodes,
            )
            chunks = [
                self.build_one_prefill_chunk(
                    reqs_start,
                    reqs_end,
                    query_start_loc_cpu,
                    common_attn_metadata.seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                )
                for reqs_start, reqs_end in chunk_seq_ids
            ]
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(
                chunks=chunks,
            )

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )

            # Use CPU to avoid GPU sync; breaking async scheduling
            requires_padding = (decode_lens_cpu.max() > decode_lens_cpu.min()).item()

            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            if is_deep_gemm_supported():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens, self.kv_cache_spec.block_size, self.num_sms
                )
            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=common_attn_metadata.block_table_tensor[:num_decodes, ...],
                seq_lens=common_attn_metadata.seq_lens[:num_decodes],
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=self.scheduler_metadata_buffer,
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            head_dim=128,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        # if get_tensor_model_parallel_rank() == 0:
        #     logger.info(f"attn_metadata: {attn_metadata}")
        return attn_metadata
