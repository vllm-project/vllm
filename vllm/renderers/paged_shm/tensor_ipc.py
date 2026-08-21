# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PagedShmTensorIPC

Details of `PagedShmTensorIPC.write` called in the API Server (Renderer):

1. Get all (large) multimodal tensors that need to be transmitted via shm.
2. Allocate SHM for all these multimodal tensors at once. Refer to the wiki:
    Dining philosophers problem.
3. `client.open_write` implements a timeout to wait until the SHM has enough
    space to allocate all these multimodal tensors at once. (Think about how
    the shm server implements this — things get interesting.)
4. `client.open_write` generates a `read_token` (with `generate_read_token=True`),
    which serves as a one-time-use read token to avoid erroneous duplicate
    releases of the same read reference.
5. Write the multimodal tensors asynchronously, and close the write notification
    so that reads can proceed.
6. Replace the actual tensors with SHM metadata (`PagedShmTensor`). ZMQ IPC only
    transmits the small SHM metadata; the actual tensor data is transmitted via SHM.

Details of `PagedShmTensorIPC.read` called in the GPU Worker (EncoderRunner):

1. Since the write is asynchronous, wait for the write to complete here
    (`client.wait_write`) using the `read_token`.
2. Use the `read_token` to read the tensors. This employs H2D batched transfer
    to read data directly from SHM into GPU tensors.
3. Place the GPU tensors back to their original positions.
4. Release the `read_token`.

### Performance improvement
The performance improvement mainly comes from replacing ZMQ IPC for multimodal
    tensor transmission with SHM-based asynchronous writing.
- For a typical 10 MiB data transfer at a typical speed of 10 GiB/s, ZMQ IPC would
    be sped up by approximately 10 ms.
- Although this is very small compared to the ~100 ms preprocessing time or the 1s+
    TTFT, it significantly accelerates ZMQ IPC (which typically takes ~1 ms when not
    transmitting large multimodal tensors).
- Therefore, the improvement is still quite noticeable.
"""

import time
import weakref
from contextlib import ExitStack

import torch

from vllm.config import ModelConfig
from vllm.inputs import MultiModalInput
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
)
from vllm.utils import random_uuid

from .client import PagedShmClient
from .types import PagedShmTensor, ShmWriteRequest


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

        # 1. Get all (large) mm tensors that need to be ipc
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

        items: list[ShmWriteRequest] = []
        for elem in elements:
            assert isinstance(elem.data, torch.Tensor)
            item = ShmWriteRequest(
                uuid=random_uuid(),
                size=elem.data.nbytes,
                use_cache=True,
                generate_read_token=True,
            )
            items.append(item)

        # 2. Allocate shm for all these mm tensors at once
        # Refer to the wiki:Dining philosophers problem.
        start = time.perf_counter()
        try:
            alloc = self.client.open_write(
                items,
                timeout=5.0,
            )
        except RuntimeError:
            return None

        # 3. Write all mm tensors to shm async, and notify other clients
        # to read the data when the async write operation is complete.
        for elem, a in zip(elements, alloc):
            assert isinstance(elem.data, torch.Tensor)
            assert a.read_token is not None

            # "activate_tokens" means to open all read tokens
            self.client.write(
                uuid=a.uuid,
                data=elem.data,
                use_cache=a.use_cache,
                blocks=a.blocks,
                activate_tokens=True,
                async_write=True,
            )

            elem.pshm_tensor = PagedShmTensor(
                uuid=a.read_token,
                size=a.size,
                blocks=a.blocks,
                dtype=str(elem.data.dtype).removeprefix("torch."),
                shape=tuple(elem.data.shape),
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

        for modality, items in mm_kwargs:
            if "pixel_values" not in items:
                continue

            pixel_values = items["pixel_values"]
            pshm_tensor: PagedShmTensor | None = pixel_values.pshm_tensor

            if pshm_tensor is not None:
                torch_dtype = getattr(torch, pshm_tensor.dtype)
                # 1. wait for the write operation to complete.
                # 2. reads the data from the shared memory.
                # 3. release the shared memory.
                # todo: support tp:
                #  wait & reads
                #  But 'release' shouldn't be here.
                tensor_gpu = self.client.read(
                    pshm_tensor.uuid, device=device, timeout=-1
                )
                tensor_gpu = tensor_gpu.view(torch_dtype).view(pshm_tensor.shape)
                pixel_values.data = tensor_gpu

                self.client.close_read(pshm_tensor.uuid)

    def shutdown(self):
        if not self.is_paged_shm_enabled:
            return None
        self._resources.close()
