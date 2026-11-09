# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 indexer metadata on SM90: the stock builder, with a
step's prefill rows split over the TP ranks from less work.

Under tensor parallelism every rank holds the whole indexer: it scores each
prefill row of a step against the row's compressed keys and keeps the row's
top-k. The ranks can split the rows instead (`../ampere/row_shard.py`:
each rank scores a contiguous share, balanced by keys, and the ranks
exchange the int32 top-k rows).

`HopperIndexerMetadataBuilder` builds the stock metadata, then splits a
step's prefill rows once their scoring work (rows times compressed keys, an
upper bound) reaches `ROW_SHARD_MIN_WORK`: from there on, one exchange per
indexer layer costs less than the scoring it saves. On H100 at TP8 this
takes 2.3% off the time to first token of a 96K-token prompt. The SM90
indexer op (`indexer.py`) runs the stock op on its rank's rows and
exchanges them; it keeps the candidate rows of the two-level selection
replicated, and splitting those as well gained another 0.3% there.
"""

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV41IndexerBackend,
)

from ..ampere.row_shard import (
    RowShardedPrefillMetadata,
    balanced_prefill_row_shard,
    prefill_row_sharding_supported,
)

# Rows times compressed keys from which a step's prefill rows split: a
# 4096-row chunk over 4096 compressed keys.
ROW_SHARD_MIN_WORK = 1 << 24


def prefill_row_shard(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    compress_ratio: int,
    tp_size: int,
) -> list[int] | None:
    """The `tp_size` ranks' shares of a step's prefill rows (balanced by
    keys), from the prefill requests' sequence lengths, their new tokens and
    the index cache's `compress_ratio`; None below `ROW_SHARD_MIN_WORK`."""
    # In int64: a long chunk over a long context passes 2^31.
    work = int((query_lens_cpu.long() * (seq_lens_cpu // compress_ratio)).sum())
    if work < ROW_SHARD_MIN_WORK:
        return None
    return balanced_prefill_row_shard(
        seq_lens_cpu, query_lens_cpu, compress_ratio, tp_size
    )


class HopperIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """The stock indexer metadata, with prefill rows split over the TP ranks
    from `ROW_SHARD_MIN_WORK`."""

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        """The stock metadata, whose prefill rows split when their scoring
        work is large enough."""
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        prefill = metadata.prefill
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if prefill is None or not prefill_row_sharding_supported(
            self.vllm_config, self.dcp_world_size, self.use_pcp, tp_size
        ):
            return metadata
        m = common_attn_metadata
        num_decodes = metadata.num_decodes
        # An upper bound for async-spec extend rows, exact for prefill rows.
        assert m.seq_lens_cpu_upper_bound is not None
        seq_lens_cpu = m.seq_lens_cpu_upper_bound[num_decodes:]
        query_lens_cpu = torch.diff(
            m.query_start_loc_cpu[num_decodes : num_decodes + metadata.num_prefills + 1]
        )
        metadata.prefill = RowShardedPrefillMetadata.of(
            prefill,
            prefill_row_shard(
                seq_lens_cpu, query_lens_cpu, self.compress_ratio, tp_size
            ),
        )
        return metadata


class HopperIndexerBackend(DeepseekV41IndexerBackend):
    """The index K cache backend of the SM90 layer."""

    @staticmethod
    def get_builder_cls() -> type[HopperIndexerMetadataBuilder]:
        return HopperIndexerMetadataBuilder
