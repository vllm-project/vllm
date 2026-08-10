# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import torch

from vllm.config import ModelConfig
from vllm.inputs import MultiModalInput
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
)
from vllm.utils import random_uuid

from .client import PagedShmClient
from .client_async import AsyncPagedShmClient
from .types import ShmItem, ShmTensor


def format_size(
    num_bytes: int,
    decimal_places: int = 4,
    use_binary: bool = True,
    target_unit: str | None = None,
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
        self.model_config = model_config
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

        self.client_async = AsyncPagedShmClient.from_model_config(
            self.model_config, pin=self.pin
        )
        assert self.client_async is not None
        self.client_sync = self.client_async.sync_client

    def write(self, mm_inputs: MultiModalInput) -> None:
        if not self.is_paged_shm_enabled:
            return None

        assert self.client_sync is not None

        elements: list[MultiModalFieldElem] = []

        def _func(elem: MultiModalFieldElem):
            if not isinstance(elem.data, torch.Tensor):
                return
            if elem.data.nbytes < self.block_size:
                return

            elements.append(elem)

        self._traversal(mm_inputs, _func)

        items: list[ShmItem] = []
        for elem in elements:
            assert isinstance(elem.data, torch.Tensor)
            item = ShmItem(uuid=random_uuid(), size=elem.data.nbytes, use_cache=False)
            items.append(item)

        try:
            alloc = self.client_sync.open_write(items)
        except MemoryError:
            return None

        for elem, item in zip(elements, alloc):
            assert isinstance(elem.data, torch.Tensor)

            elem.shm_object = ShmTensor(
                dtype=str(elem.data.dtype).removeprefix("torch."),
                shape=tuple(elem.data.shape),
                **asdict(item),
            )
            self.client_sync.write(
                uuid=item.uuid,
                data=elem.data,
                use_cache=item.use_cache,
                blocks=item.blocks,
            )
            elem.data = None
            self.client_sync.open_read(item.uuid)
        return None

    def _traversal(self, obj: Any, func: Callable[[MultiModalFieldElem], None]):
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._traversal(v, func)
            return None
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self._traversal(v, func)
            return None
        elif isinstance(obj, MultiModalKwargsItem):
            for k, v in obj.items():
                self._traversal(v, func)
            return None
        elif isinstance(obj, MultiModalFieldElem):
            func(obj)
            return None
        elif isinstance(obj, MultiModalKwargsItems):
            for modality, itemlist in obj.items():
                for item in itemlist:
                    self._traversal(item, func)
            return None
        return None
