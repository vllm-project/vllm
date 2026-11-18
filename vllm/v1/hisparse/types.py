# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

# Sealed resident blocks this many positions behind the block-table tail stay
# pinned so a block written by an in-flight step is never handed out under it.
ACTIVE_TAIL_BLOCKS = 2


@dataclass(frozen=True, slots=True)
class SparseKVPageTransfer:
    """Copy one host page between cache-manager and worker-owned tiers.

    The page is the `resident_page`-th of the resident blocks it maps to."""

    transfer_id: int
    host_block_id: int
    resident_block_ids: tuple[int, ...]
    restore: bool = False
    resident_page: int = 0


@dataclass(frozen=True, slots=True)
class SparseKVRowMirror:
    """Mirror one contiguous resident-row span into the host cache."""

    source_starts: tuple[int, ...]
    destination_start: int
    num_rows: int


@dataclass(frozen=True, slots=True)
class SparseKVResidencyUpdate:
    """GPU block ids of some of a request's resident blocks, per resident
    group. A null block id means the block is read from the host."""

    block_indices: list[int]
    block_ids: tuple[list[int], ...]


@dataclass(slots=True)
class SparseKVOffloadCommand:
    """Opaque scheduler-to-worker command for one model step."""

    page_transfers: list[SparseKVPageTransfer]
