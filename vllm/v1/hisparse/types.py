# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class HiSparsePageTransfer:
    """Copy one logical KV page between cache-manager and worker-owned tiers."""

    transfer_id: int
    host_block_id: int
    resident_block_ids: tuple[int, ...]
    runs_after_forward: bool
    is_restore: bool = False


@dataclass(frozen=True)
class HiSparseRowMirror:
    """Mirror one contiguous resident-row span into the host cache."""

    source_starts: tuple[int, ...]
    destination_start: int
    num_rows: int


@dataclass
class HiSparseTransferCommand:
    """Opaque scheduler-to-worker command for one model step."""

    page_transfers: list[HiSparsePageTransfer]
