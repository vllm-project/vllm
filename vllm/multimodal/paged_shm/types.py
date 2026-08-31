# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.processing.processor import ResolvedPromptUpdate


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
    is_new: bool = False


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


class PagedShmTensorShmSlot(TypedDict):
    token: str
    meta_size: int
    meta_block: int
    data_size: int
    data_blocks: list[int]


@dataclass
class ShmItem:
    kwargs_item: MultiModalKwargsItem
    prompt_updates: Sequence["ResolvedPromptUpdate"]
