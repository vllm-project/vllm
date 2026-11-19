# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 indexer metadata on SM80.

Each step, the indexer metadata builder tells the indexer which tokens are
decodes and which are prefills, which compressed keys each one may score,
and where the key cache pages of each request are. The stock builder also
plans DeepGEMM's paged-logits schedule for every decode step, and DeepGEMM
has no SM80 kernels (a vLLM install bundles it, so it is present but fails).
This builder keeps the stock one's state and its prefill chunk planning, and
builds the decode part the SM80 logits kernel reads:

* one row per decode token, holding that token's causal compressed length
  and a copy of its request's block table (the stock flattening kernel);
* no DeepGEMM schedule.

Neither decode nor prefill context parallelism is supported, and the index
K cache is always MXFP4 (`indexer_uses_fp4`).
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV41IndexerBackend,
    build_prefill_chunk_metadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.metadata import _indexer_decode_metadata_kernel

from . import indexer_uses_fp4
from .row_shard import (
    RowShardedPrefillMetadata,
    balanced_prefill_row_shard,
    prefill_row_sharding_supported,
)

logger = init_logger(__name__)

# A step's prefill rows split over the TP ranks (``row_shard.py``) once their
# scoring work (rows times compressed keys, an upper bound) reaches this. SM80
# scores on BF16 tensor cores, several times slower per key than SM90's FP8
# kernels, so it splits from a quarter of SM90's threshold; from there on,
# exchanging the int32 top-k rows costs far less than the scoring it saves.
_ROW_SHARD_MIN_WORK = 1 << 22


class AmpereIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """The stock indexer metadata on the MXFP4 cache, with one decode row per
    token and no DeepGEMM schedule."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.dcp_world_size > 1 or self.use_pcp:
            raise NotImplementedError(
                "The DeepSeek-V4.1 SM80 indexer supports no context parallelism."
            )
        self.indexer_uses_fp4 = indexer_uses_fp4(self.vllm_config)
        # The fused MQA+top-k kernel takes no MXFP4 cache.
        self.fused_mqa_topk_eligible = False
        logger.info_once("DSA indexer on SM80: MXFP4 cache, one decode row per token")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        """Decode rows per token, stock prefill chunks."""
        m = common_attn_metadata
        num_tokens = m.num_actual_tokens
        # As in the stock builder, an index cache without compression (a layer
        # of compress ratio 1) keeps the token slots.
        slot_mapping = m.slot_mapping
        if self.compress_ratio > 1:
            slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                m.slot_mapping,
                m.query_start_loc,
                m.seq_lens,
                m.block_table_tensor,
                self.kv_cache_spec.num_states,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
        metadata = DeepseekV32IndexerMetadata(
            seq_lens=m.seq_lens,
            max_seq_len=m.max_seq_len,
            slot_mapping=slot_mapping,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=0,
            num_prefill_tokens=0,
            block_table=m.block_table_tensor,
        )
        if num_tokens == 0:
            return metadata
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                m, decode_threshold=self.decode_threshold, require_uniform=False
            )
        )
        metadata.num_decodes = num_decodes
        metadata.num_decode_tokens = num_decode_tokens
        metadata.num_prefills = num_prefills
        metadata.num_prefill_tokens = num_prefill_tokens
        if num_prefills > 0:
            metadata.prefill = self._build_prefill(m, num_decodes, num_prefills)
        if num_decodes > 0:
            metadata.decode = self._build_decode(m, num_decodes, num_decode_tokens)
        return metadata

    def _build_prefill(
        self, m: CommonAttentionMetadata, num_decodes: int, num_prefills: int
    ) -> RowShardedPrefillMetadata:
        """The stock prefill chunks under the logits budget, and the TP row
        shares once the step's scoring work reaches `_ROW_SHARD_MIN_WORK`."""
        compressed_seq_lens = m.seq_lens // self.compress_ratio
        # An upper bound for async-spec extend rows, exact for prefill rows.
        assert m.seq_lens_cpu_upper_bound is not None
        seq_lens_cpu = m.seq_lens_cpu_upper_bound
        compressed_seq_lens_cpu = seq_lens_cpu // self.compress_ratio
        query_lens_cpu = torch.diff(
            m.query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
        )
        chunk_specs = self._split_indexer_prefill_chunks(
            self._prefill_split_seq_lens(compressed_seq_lens_cpu[num_decodes:]),
            query_lens_cpu,
            self.max_prefill_buffer_size,
            envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024,
            request_offset=num_decodes,
        )
        chunks = []
        for req_slice, query_slice in chunk_specs:
            chunk = build_prefill_chunk_metadata(
                req_slice.start,
                req_slice.stop,
                m.query_start_loc,
                m.query_start_loc_cpu,
                m.seq_lens,
                compressed_seq_lens,
                compressed_seq_lens_cpu,
                m.block_table_tensor,
                self.compress_ratio,
                query_slice=query_slice,
                skip_kv_gather=query_slice.start > 0,
            )
            # None when the chunk has no compressed key yet.
            if chunk is not None:
                chunks.append(chunk)
        row_shard_sizes = None
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if prefill_row_sharding_supported(self.vllm_config, 1, False, tp_size):
            # In int64: a long chunk over a long context passes 2^31.
            work = int(
                (query_lens_cpu.long() * compressed_seq_lens_cpu[num_decodes:])
                .sum()
                .item()
            )
            if work >= _ROW_SHARD_MIN_WORK:
                row_shard_sizes = balanced_prefill_row_shard(
                    seq_lens_cpu[num_decodes:],
                    query_lens_cpu,
                    self.compress_ratio,
                    tp_size,
                )
        return RowShardedPrefillMetadata(
            chunks,
            max_prefill_seq_len=int(seq_lens_cpu[num_decodes:].max().item()),
            row_shard_sizes=row_shard_sizes,
        )

    def _build_decode(
        self, m: CommonAttentionMetadata, num_decodes: int, num_decode_tokens: int
    ) -> DeepSeekV32IndexerDecodeMetadata:
        """One row per decode token: its causal compressed length (zero for a
        CUDA-graph padding row) and its request's block table."""
        block_table = m.block_table_tensor
        capacity = self.decode_seq_lens_buffer.numel()
        # Token programs fill the rows; the others zero the lengths past them.
        grid = max(
            num_decodes,
            num_decode_tokens + triton.cdiv(capacity - num_decode_tokens, 256),
        )
        _indexer_decode_metadata_kernel[(grid,)](
            m.query_start_loc,
            m.seq_lens,
            block_table,
            self.decode_seq_lens_buffer,
            self.expanded_block_table_buffer,
            self.decode_lens_buffer,
            self.decode_indices_buffer,
            self.per_req_decode_lens_buffer,
            num_decodes,
            num_decode_tokens,
            capacity,
            block_table.stride(0),
            self.expanded_block_table_buffer.stride(0),
            BLOCK_COLS=block_table.shape[1],
            num_warps=4,
        )
        seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
        seq_lens //= self.compress_ratio
        decode_lens_cpu = torch.diff(m.query_start_loc_cpu[: num_decodes + 1])
        max_decode_len = int(decode_lens_cpu.max().item())
        uniform = max_decode_len == int(decode_lens_cpu.min().item())
        if self.use_adaptive_verification:
            # Graphs are captured for up to next_n tokens per request.
            max_decode_len, uniform = 1 + self.num_speculative_tokens, False
        return DeepSeekV32IndexerDecodeMetadata(
            block_table=self.expanded_block_table_buffer[:num_decode_tokens],
            seq_lens=seq_lens.unsqueeze(-1),
            decode_lens=self.decode_lens_buffer[:num_decode_tokens],
            requires_padding=False,
            # Unread: the SM80 kernel takes no DeepGEMM schedule.
            schedule_metadata=self.scheduler_metadata_buffer,
            indices=self.decode_indices_buffer[:num_decode_tokens],
            per_req_decode_lens=self.per_req_decode_lens_buffer[:num_decodes],
            decode_is_uniform=uniform,
            write_max_decode_len=max_decode_len,
        )

    def update_draft_decode_metadata(
        self, metadata: DeepseekV32IndexerMetadata
    ) -> None:
        """The stock draft-step update (one token per request), without the
        DeepGEMM schedule."""
        decode = metadata.decode
        if decode is None or metadata.num_decode_tokens == 0:
            return
        num_tokens = metadata.num_decode_tokens
        assert metadata.num_prefills == 0 and metadata.num_decodes == num_tokens
        assert metadata.block_table is not None
        decode.block_table.copy_(metadata.block_table[:num_tokens])
        if decode.indices is not None:
            decode.indices.copy_(self.arange_buffer[:num_tokens])
        if decode.per_req_decode_lens is not None:
            decode.per_req_decode_lens.fill_(1)
        get_compressed_slot_mapping(
            num_tokens,
            self.arange_buffer[:num_tokens],
            self.arange_buffer[: num_tokens + 1],
            metadata.seq_lens,
            decode.block_table,
            self.kv_cache_spec.num_states,
            self.compress_ratio,
            out=metadata.slot_mapping,
        )
        torch.div(
            metadata.seq_lens,
            self.compress_ratio,
            rounding_mode="floor",
            out=decode.seq_lens.view(-1),
        )
        decode.decode_lens.fill_(1)


class AmpereIndexerBackend(DeepseekV41IndexerBackend):
    """The index K cache backend of the SM80 layer."""

    @staticmethod
    def get_builder_cls() -> type[AmpereIndexerMetadataBuilder]:
        return AmpereIndexerMetadataBuilder
