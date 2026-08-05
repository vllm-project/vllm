# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class SparseKVPageTransfer:
    """Copy one logical KV page between cache-manager and worker-owned tiers."""

    transfer_id: int
    destination_block_id: int
    destination_page_offset: int
    source_block_ids: tuple[int, ...]
    after_forward: bool


@dataclass
class SparseKVOffloadCommand:
    """Opaque scheduler-to-worker command for one model step."""

    block_table_updates: dict[str, tuple[list[int], ...]]
    page_transfers: list[SparseKVPageTransfer]
    fully_resident: bool
