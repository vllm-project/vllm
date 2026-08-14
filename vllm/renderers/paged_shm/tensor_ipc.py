# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
import weakref
from contextlib import ExitStack
from dataclasses import asdict

import torch

from vllm.config import ModelConfig
from vllm.inputs import MultiModalInput
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
)
from vllm.utils import random_uuid

from .client import PagedShmClient
from .client_async import AsyncPagedShmClient
from .types import PagedShmTensor, ShmItem


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
        self._resources = ExitStack()
        self._finalizer = weakref.finalize(self, self._resources.close)

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
        if self.client_async is not None:
            return

        self.client_async = AsyncPagedShmClient.from_model_config(
            self.model_config, pin=self.pin
        )

        if self.client_async is not None:
            self.client_sync = self.client_async.sync_client
            self._resources.callback(self.client_async.close)

    def write(self, mm_inputs: MultiModalInput) -> None:
        if not self.is_paged_shm_enabled:
            return None
        if self.client_sync is None:
            return None

        elements: list[MultiModalFieldElem] = []
        mm_kwargs = mm_inputs["mm_kwargs"]
        for modality, mm_items in mm_kwargs.items():
            for mm_item in mm_items:
                if mm_item is None:
                    continue
                if "pixel_values" in mm_item:
                    elem: MultiModalFieldElem = mm_item["pixel_values"]
                    if not isinstance(elem.data, torch.Tensor):
                        continue
                    if elem.data.nbytes < self.block_size:
                        continue
                    elements.append(elem)

        items: list[ShmItem] = []
        for elem in elements:
            assert isinstance(elem.data, torch.Tensor)
            item = ShmItem(uuid=random_uuid(), size=elem.data.nbytes, use_cache=True)
            items.append(item)

        start = time.perf_counter()
        try:
            alloc = self.client_sync.open_write(items)
        except MemoryError:
            return None

        for elem, item in zip(elements, alloc):
            assert isinstance(elem.data, torch.Tensor)

            elem.pshm_tensor = PagedShmTensor(
                dtype=str(elem.data.dtype).removeprefix("torch."),
                shape=tuple(elem.data.shape),
                **asdict(item),
            )
            self.client_sync.write(
                uuid=item.uuid,
                data=elem.data,
                use_cache=item.use_cache,
                blocks=item.blocks,
                open_read=True,
                async_write=True,
            )
            elem.data = None
        end = time.perf_counter()
        elapsed_time = end - start
        print(
            f"PagedShmTensorIPC.write {elapsed_time * 1000} ms",
        )
        return None

    def read(
        self,
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
        device: torch.types.Device,
    ):
        if not self.is_paged_shm_enabled:
            return None
        if self.client_sync is None:
            return None

        for modality, items in mm_kwargs:
            if "pixel_values" not in items:
                continue

            pixel_values = items["pixel_values"]
            pshm_tensor: PagedShmTensor | None = pixel_values.pshm_tensor

            if pshm_tensor is not None:
                torch_dtype = getattr(torch, pshm_tensor.dtype)
                tensor_gpu = self.client_sync.read(
                    pshm_tensor.uuid, pshm_tensor.size, pshm_tensor.blocks, device
                )
                tensor_gpu = tensor_gpu.view(torch_dtype).view(pshm_tensor.shape)
                pixel_values.data = tensor_gpu

    def shutdown(self):
        if not self.is_paged_shm_enabled:
            return None
        if self.client_async is None:
            return None
        self._resources.close()
