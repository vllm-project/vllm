# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class ShmWriteRequest:
    """A single item described by a client for a write operation."""

    uuid: str
    size: int
    use_cache: bool


@dataclass
class ShmAllocation(ShmWriteRequest):
    """Result of a successful allocation, including the assigned physical blocks."""

    blocks: list[int]


@dataclass
class ShmSlot(ShmAllocation):
    """
    Internal representation used by the manager to track an allocated slot.
    Includes reference count and block count helper.
    """

    ref_count: int = 0

    def n_block(self) -> int:
        """Number of physical blocks occupied by this slot."""
        return len(self.blocks)


@dataclass
class PagedShmTensor:
    uuid: str
    size: int
    blocks: list[int]
    dtype: str
    shape: tuple[int, ...]
