# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class SparseKVPageTransfer:
    """Copy one logical KV page between cache-manager and worker-owned tiers."""

    transfer_id: int
    destination_block_id: int
    source_block_ids: tuple[int, ...]
    after_forward: bool


@dataclass(frozen=True)
class SparseKVRowMirror:
    """Mirror one contiguous resident-row span into the host cache."""

    source_starts: tuple[int, ...]
    destination_start: int
    num_rows: int


@dataclass
class SparseKVOffloadCommand:
    """Opaque scheduler-to-worker command for one model step."""

    page_transfers: list[SparseKVPageTransfer]
