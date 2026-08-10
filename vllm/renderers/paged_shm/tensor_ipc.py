# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.config import ModelConfig
from vllm.inputs import MultiModalInput
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
)

from .client import PagedShmClient
from .client_async import AsyncPagedShmClient


def format_size(
    num_bytes: int,
    decimal_places: int = 4,
    use_binary: bool = True,
    target_unit: str = None,
) -> str:
    """Format a byte count as a human-readable string."""
    if num_bytes == 0:
        return f"0 {target_unit or 'B'}"
    units = ["B", "KiB", "MiB", "GiB"] if use_binary else ["B", "KB", "MB", "GB"]
    base = 1024 if use_binary else 1000
    if target_unit is not None:
        target_exp = units.index(target_unit)
        size = num_bytes / (base**target_exp)
        return f"{size:.{decimal_places}f} {target_unit}"
    exponent = 0
    size = num_bytes
    while size >= base and exponent < len(units) - 1:
        size /= base
        exponent += 1
    return f"{size:.{decimal_places}f} {units[exponent]}"


class PagedShmTensorIPC:
    def __init__(self, model_config: ModelConfig, pin: bool = False):
        self.is_paged_shm_enabled = False
        self.multimodal_config = model_config.multimodal_config
        if self.multimodal_config is None:
            return

        self.is_paged_shm_enabled = self.multimodal_config.is_paged_shm_enabled()
        if not self.is_paged_shm_enabled:
            return

        self.pin = pin
        self.block_size = self.multimodal_config.paged_shm_block_size
        self.client_async: AsyncPagedShmClient | None = None
        self.client_sync: PagedShmClient | None = None

    def connect(self):
        if not self.is_paged_shm_enabled:
            return

        self.client_async = AsyncPagedShmClient(
            address=self.multimodal_config.paged_shm_server_address, pin=self.pin
        )
        self.client_sync = self.client_async.sync_client

    def write(self, mm_inputs: MultiModalInput):
        if not self.is_paged_shm_enabled:
            return None

        print()
        print("PagedShmMMInputs write")
        print("-" * 80)
        self._traversal(mm_inputs)
        print("-" * 80)
        return None

    def _write(self, elem: MultiModalFieldElem):
        data = elem.data
        if isinstance(data, torch.Tensor) and data.nbytes > self.block_size:
            print(format_size(data.nbytes))

    def _traversal(self, obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._traversal(v)
            return None
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self._traversal(v)
            return None
        elif isinstance(obj, MultiModalKwargsItem):
            for k, v in obj.items():
                self._traversal(v)
            return None
        elif isinstance(obj, MultiModalFieldElem):
            self._write(obj)
            return None
        elif isinstance(obj, MultiModalKwargsItems):
            for modality, itemlist in obj.items():
                for item in itemlist:
                    self._traversal(item)
            return None
        return None
