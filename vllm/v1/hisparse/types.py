# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

# Sealed pages this many positions behind the block-table tail stay pinned so
# a page written by an in-flight step is never handed out under it.
ACTIVE_TAIL_PAGES = 2


@dataclass(frozen=True)
class HiSparsePageTransfer:
    """Copy one logical KV page between cache-manager and worker-owned tiers."""

    transfer_id: int
    host_block_id: int
    device_block_ids: tuple[int, ...]
    runs_after_forward: bool
    is_restore: bool = False


@dataclass(frozen=True)
class HiSparseRowMirror:
    """Mirror one contiguous device-row span into the host cache."""

    source_starts: tuple[int, ...]
    destination_start: int
    num_rows: int


@dataclass(frozen=True)
class HiSparseResidencyUpdate:
    """GPU block ids of some of a request's device pages, per device group.
    A null block id means the page is read from the host."""

    pages: list[int]
    block_ids: tuple[list[int], ...]


@dataclass
class HiSparseTransferCommand:
    """Opaque scheduler-to-worker command for one model step."""

    page_transfers: list[HiSparsePageTransfer]
