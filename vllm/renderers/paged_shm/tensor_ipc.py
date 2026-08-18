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
from .types import PagedShmTensor, ShmItem


class PagedShmTensorIPC:
    def __init__(
        self, model_config: ModelConfig, pin: bool = False, connect: bool = True
    ):
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
        self.client: PagedShmClient | None = None

        if connect:
            self.connect()

    def connect(self):
        if not self.is_paged_shm_enabled:
            return

        self.client = PagedShmClient.from_model_config(self.model_config, pin=self.pin)

        if self.client is not None:
            self._resources.callback(self.client.close)

    def write(self, mm_inputs: MultiModalInput) -> None:
        if not self.is_paged_shm_enabled:
            return None
        if self.client is None:
            return None

        # 1. Get all mm tensors that need to be ipc
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

        # 2. Allocate shm for all these mm tensors at once
        # Refer to the wiki:Dining philosophers problem.
        start = time.perf_counter()
        try:
            alloc = self.client.open_write(items, timeout=5.0)
        except RuntimeError:
            return None

        # 3. Write all mm tensors to shm async, and notify other clients
        # to read the data when the write operation is complete.
        for elem, item in zip(elements, alloc):
            assert isinstance(elem.data, torch.Tensor)

            elem.pshm_tensor = PagedShmTensor(
                dtype=str(elem.data.dtype).removeprefix("torch."),
                shape=tuple(elem.data.shape),
                **asdict(item),
            )
            self.client.write(
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
        if self.client is None:
            return None

        # 1. wait for the write operation to complete.
        # 2. reads the data from the shared memory.
        # 3. release the shared memory.
        for modality, items in mm_kwargs:
            if "pixel_values" not in items:
                continue

            pixel_values = items["pixel_values"]
            pshm_tensor: PagedShmTensor | None = pixel_values.pshm_tensor

            if pshm_tensor is not None:
                torch_dtype = getattr(torch, pshm_tensor.dtype)
                print("pshm_tensor.blocks", pshm_tensor.blocks)
                print("device:", device)
                tensor_gpu = self.client.read(
                    pshm_tensor.uuid,
                    # pshm_tensor.size,
                    # pshm_tensor.blocks,
                    device=device,
                    timeout=-1,
                )
                tensor_gpu = tensor_gpu.view(torch_dtype).view(pshm_tensor.shape)
                pixel_values.data = tensor_gpu

    def shutdown(self):
        if not self.is_paged_shm_enabled:
            return None
        self._resources.close()
