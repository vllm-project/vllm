# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

# Sealed pages this many positions behind the block-table tail stay pinned so
# a page written by an in-flight step is never handed out under it.
ACTIVE_TAIL_PAGES = 2


@dataclass(frozen=True)
class SparseKVPageTransfer:
    """Copy one logical KV page between cache-manager and worker-owned tiers."""

    transfer_id: int
    host_block_id: int
    resident_block_ids: tuple[int, ...]
    after_forward: bool
    restore: bool = False


@dataclass(frozen=True)
class SparseKVRowMirror:
    """Mirror one contiguous resident-row span into the host cache."""

    source_starts: tuple[int, ...]
    destination_start: int
    num_rows: int


@dataclass(frozen=True)
class SparseKVResidencyUpdate:
    """GPU block ids of some of a request's resident pages, per resident group.
    A null block id means the page is read from the host."""

    pages: list[int]
    block_ids: tuple[list[int], ...]


@dataclass
class SparseKVOffloadCommand:
    """Opaque scheduler-to-worker command for one model step."""

    page_transfers: list[SparseKVPageTransfer]
