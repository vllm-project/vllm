# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .manager import PagedSHMManager
from .storage import PagedSHMStorage


class PagedSHMServer:
    def __init__(self, size: int, block_size: int):
        self.storage = PagedSHMStorage(size, block_size, pin=False)
        self.manager = PagedSHMManager(size, block_size)


class PagedSHMServerProcess:
    pass
