# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-region transfer classes for the NIXL connector.

Within a single KV-cache group the connector may transfer regions of two
different kinds (e.g. a full-attention (GQA) main model paired with an MLA
draft such as Eagle-3). Rather than scatter ``if is_replicate`` branches across
the gate and the descriptor builders, each region is tagged with one
``RegionTransferClass`` that owns the rules for that kind of region:

- ``SPLIT`` — full-attention (GQA), KV is head-sharded across TP. The remote
  block holds ``tp_ratio`` times the local heads; a decode rank reads its head
  slice at a per-rank offset, and (under the FlashInfer "virtually split" block
  layout) K and V are two separate descriptor streams.
- ``REPLICATE`` — MLA, ``num_kv_heads==1``. The cache is identical on every
  rank, so the whole block is read from a single remote rank at offset 0; it is
  key-only, so it has no V stream.

This is purely a connector-internal descriptor concern; it does not change the
scheduler's KV-cache grouping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionTransferClass:
    """How one KV region is split (or not) when transferred over NIXL."""

    name: str
    is_replicate: bool

    def num_streams(self, virtually_split: bool) -> int:
        """Number of descriptor streams a region of this class emits.

        Under the blocks-first ("virtually split") layout a SPLIT region is
        indexed as two streams (K then V); a REPLICATE region is key-only, so
        it is always a single stream. Without virtual splitting every region is
        a single registered stream.
        """
        if self.is_replicate or not virtually_split:
            return 1
        return 2

    def remote_num_reads(self, split_reads: int) -> int:
        """How many remote ranks a region of this class is read from.

        REPLICATE reads the whole block from one rank; SPLIT reads its head
        slice, which may be gathered from ``split_reads`` remote ranks when
        P_TP > D_TP.
        """
        return 1 if self.is_replicate else split_reads

    def remote_rank_offset(self, offset_factor: int, remote_kv_block_len: int) -> int:
        """Byte offset into the remote block for this rank's head slice.

        REPLICATE copies the whole block (offset 0); SPLIT hops to its head
        slice.
        """
        return 0 if self.is_replicate else offset_factor * remote_kv_block_len

    def local_split_desc(
        self,
        addr: int,
        local_len: int,
        device: int,
        head_slot: int,
        num_splits: int,
    ) -> tuple[int, int, int]:
        """Local destination descriptor for one source-rank read when gathering
        from multiple remote ranks (P_TP > D_TP).

        REPLICATE writes the whole block (every remote rank holds the same
        data). SPLIT writes only this rank's head slice into its slot.
        """
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
                "Different local/remote block sizes are not supported"
                " when P TP > D TP."
            )
            assert remote_len == local_len // (-tp_ratio), (
                f"SPLIT region {region_idx}: remote P KV block_len {remote_len} "
                f"must equal local {local_len} // |tp_ratio| {-tp_ratio}."
            )


SPLIT_CLASS = RegionTransferClass(name="split", is_replicate=False)
REPLICATE_CLASS = RegionTransferClass(name="replicate", is_replicate=True)
