# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen4Exp PLE helpers."""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)


@dataclass(frozen=True)
class PLEShardOverlap:
    """Source and destination slices for one checkpoint embedding shard."""

    source_start: int
    destination_start: int
    row_count: int


class PLEVocabParallelEmbedding(VocabParallelEmbedding):
    """Vocab-parallel embedding that accepts checkpoint row shards."""

    def _compute_shard_overlap(
        self,
        *,
        checkpoint_start: int,
        checkpoint_rows: int,
    ) -> PLEShardOverlap | None:
        """Compute the overlap of a checkpoint shard and one TP vocabulary range."""
        tp_start = self.shard_indices.org_vocab_start_index
        tp_end = self.shard_indices.org_vocab_end_index
        if checkpoint_start < 0 or checkpoint_rows < 0:
            raise ValueError("checkpoint shard bounds must be non-negative")
        if tp_start < 0 or tp_end < tp_start:
            raise ValueError("invalid TP vocabulary range")
        checkpoint_end = checkpoint_start + checkpoint_rows
        overlap_start = max(checkpoint_start, tp_start)
        overlap_end = min(checkpoint_end, tp_end)
        if overlap_start >= overlap_end:
            return None
        return PLEShardOverlap(
            source_start=overlap_start - checkpoint_start,
            destination_start=overlap_start - tp_start,
            row_count=overlap_end - overlap_start,
        )

    def _copy_embedding_shard_(
        self,
        destination: torch.Tensor,
        loaded_weight: torch.Tensor,
        *,
        checkpoint_start: int,
    ) -> int:
        """Copy the overlapping rows of a PLE checkpoint shard into a TP table."""
        tp_start = self.shard_indices.org_vocab_start_index
        tp_end = self.shard_indices.org_vocab_end_index
        if destination.ndim == 0 or loaded_weight.ndim != destination.ndim:
            raise ValueError("destination and loaded weight must have matching ranks")
        if destination.shape[1:] != loaded_weight.shape[1:]:
            raise ValueError(
                "embedding shard dimensions do not match: "
                f"{tuple(destination.shape[1:])} != {tuple(loaded_weight.shape[1:])}"
            )
        if destination.shape[0] < tp_end - tp_start:
            raise ValueError("destination does not cover the requested TP range")
        overlap = self._compute_shard_overlap(
            checkpoint_start=checkpoint_start,
            checkpoint_rows=loaded_weight.shape[0],
        )
        if overlap is None:
            return 0
        source = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
        target = destination.narrow(0, overlap.destination_start, overlap.row_count)
        with torch.no_grad():
            target.copy_(source.to(device=target.device, dtype=target.dtype))
        return overlap.row_count

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        checkpoint_start: int | None = None,
    ) -> None:
        if checkpoint_start is None:
            super().weight_loader(param, loaded_weight)
            return
        self._copy_embedding_shard_(
            param,
            loaded_weight,
            checkpoint_start=checkpoint_start,
        )
