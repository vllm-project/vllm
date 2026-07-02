# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Literal

from vllm.renderers.paged_shm.client import PagedSHMClient

SHM_OBJECT_MODE = Literal["read", "write"]


@dataclass
class SHMObject:
    name: str
    size: int
    block_size: int
    blocks: list[int]
    mode: SHM_OBJECT_MODE
    client: PagedSHMClient

    def save(self):
        pass

    def numpy(self):
        assert self.mode == "read"

    def tensor(self, device: str):
        assert self.mode == "read"
