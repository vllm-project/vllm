# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class ShmWriteRequest:
    """Request to allocate blocks for a write operation."""
    uuid: str
    size: int
    use_cache: bool = True
    generate_read_token: bool = False


@dataclass
class ShmAllocation:
    """Response from server after allocation, or from open_read."""
    uuid: str
    size: int
    blocks: list[int]
    use_cache: bool = True
    read_token: str | None = None


@dataclass
class ShmSlot:
    """Internal representation of an allocated item in the manager."""
    uuid: str
    size: int
    use_cache: bool
    blocks: list[int]
    ref_count: int  # -1: writing, 0: idle, >0: reading

    def n_block(self) -> int:
        return len(self.blocks)


@dataclass
class PagedShmTensor:
    uuid: str
    size: int
    blocks: list[int]
    dtype: str
    shape: tuple[int, ...]
