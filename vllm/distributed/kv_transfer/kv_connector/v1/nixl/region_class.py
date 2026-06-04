# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-region transfer classes for the NIXL connector.

Each region in a KV-cache group is tagged with one ``RegionTransferClass``,
which owns the descriptor rules for that region kind:

- ``SPLIT`` — full-attention (GQA), head-sharded across TP. Remote block holds
  ``tp_ratio`` x local heads; a rank reads its head slice at a per-rank offset,
  and under the FlashInfer "virtually split" layout K/V are two streams.
- ``REPLICATE`` — MLA (``num_kv_heads==1``), identical on every rank: whole
  block read from one rank at offset 0, key-only (no V stream).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionTransferClass:
    """How one KV region is split (or not) when transferred over NIXL."""

    is_replicate: bool

    def num_streams(self, virtually_split: bool) -> int:
        """Descriptor streams emitted: 2 (K, V) for a virtually-split SPLIT
        region, else 1 (REPLICATE is key-only)."""
        if self.is_replicate or not virtually_split:
            return 1
        return 2

    def remote_num_reads(self, split_reads: int) -> int:
        """Remote ranks read from: 1 for REPLICATE; ``split_reads`` for SPLIT
        (gathered head slice when P_TP > D_TP)."""
        return 1 if self.is_replicate else split_reads

    def remote_rank_offset(self, offset_factor: int, remote_kv_block_len: int) -> int:
        """Byte offset into the remote block: 0 for REPLICATE (whole block),
        else this rank's head slice."""
        return 0 if self.is_replicate else offset_factor * remote_kv_block_len

    def local_split_desc(
        self,
        addr: int,
        local_len: int,
        device: int,
        head_slot: int,
        num_splits: int,
    ) -> tuple[int, int, int]:
        """Local dest descriptor for one source-rank read when gathering from
        multiple remote ranks (P_TP > D_TP): REPLICATE writes the whole block,
        SPLIT writes only this rank's head slice."""
        if self.is_replicate:
            return (addr, local_len, device)
        chunk = local_len // num_splits
        return (addr + head_slot * chunk, chunk, device)

    def validate_block_len(
        self,
        region_idx: int,
        local_len: int,
        remote_len: int,
        tp_ratio: int,
        block_size_ratio: int,
    ) -> None:
        """Assert the P/D block-length invariant for a region of this class."""
        if self.is_replicate:
            # Whole block copied; only the number of blocks may differ.
            assert local_len // block_size_ratio == remote_len, (
                "KV cache sizes must match between P and D when replicated "
                f"(region {region_idx}: local={local_len}, remote={remote_len}, "
                f"bsr={block_size_ratio})."
            )
        elif tp_ratio > 0:
            # D_TP >= P_TP: remote holds tp_ratio x local heads.
            assert remote_len == (local_len * tp_ratio) // block_size_ratio, (
                f"SPLIT region {region_idx}: remote P KV block_len {remote_len} "
                f"must equal local {local_len} * tp_ratio {tp_ratio} "
                f"// block_size_ratio {block_size_ratio}."
            )
        else:
            # P_TP > D_TP: local holds |tp_ratio| x remote heads.
            assert block_size_ratio == 1, (
                "Different local/remote block sizes are not supported when P TP > D TP."
            )
            assert remote_len == local_len // (-tp_ratio), (
                f"SPLIT region {region_idx}: remote P KV block_len {remote_len} "
                f"must equal local {local_len} // |tp_ratio| {-tp_ratio}."
            )


SPLIT_CLASS = RegionTransferClass(is_replicate=False)
REPLICATE_CLASS = RegionTransferClass(is_replicate=True)
