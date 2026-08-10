# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class ShmItem:
    uuid: str
    size: int
    use_cache: bool


@dataclass
class AllocatedShmItem(ShmItem):
    blocks: list[int]


@dataclass
class ShmTensor(ShmItem):
    blocks: list[int]
    dtype: str
    shape: tuple[int, ...]


@dataclass
class AllocatedShmItemInternal(AllocatedShmItem):
    ref_count: int = 0

    def n_block(self):
        return len(self.blocks)
