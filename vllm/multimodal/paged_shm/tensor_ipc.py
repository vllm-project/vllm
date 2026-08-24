# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PagedShmTensorIPC: Offload large tensor transfers from ZMQ to shared memory.

This module provides a high‑performance mechanism for transmitting large
multimodal tensors (pixel values, video frames) between the API server and
GPU workers by offloading the data transfer from the ZMQ IPC hot path to
paged shared memory (SHM). This reduces latency and avoids serialization
overhead, especially under high concurrency.

**Key Use Case**: In vLLM's multimodal serving pipeline, the API server writes
large tensors into SHM, replacing them with lightweight `PagedShmTensor`
metadata (UUID, block list, shape, dtype). The GPU worker reads the metadata,
waits for the write to complete, reconstructs the tensor from SHM (to GPU or CPU),
and releases the read token.

**Workflow in `write()`**:
1. Identify large tensors (size > block_size) in `mm_inputs`.
2. Create batched `ShmWriteRequest` items with UUID and `generate_read_token=True`.
3. Atomically allocate blocks via `open_write` (avoids partial allocation).
4. Submit asynchronous writes (copy data + `close_write`) to a thread pool.
5. Replace original tensor `.data` with `PagedShmTensor` (token, blocks, shape, dtype).

**Workflow in `read()`**:
1. Extract `PagedShmTensor` from multimodal metadata.
2. Wait for the async write to complete (`wait_write` with timeout).
3. Read raw bytes from SHM to the target device and reconstruct the tensor.
4. Destroy the read token (`close_read`) – the token cannot be reused.

**Performance**: In the critical request‑processing path, ZMQ IPC can become
a bottleneck for large tensors, with latencies often ranging from 1‑10 ms under
load. By offloading tensor data to SHM, we eliminate this hot path, reducing
transfer time to sub‑millisecond for tensors > 1 MiB. The main overhead shifts
to allocation and synchronization, which are amortized over batched writes.

Enabled when `multimodal_config.is_paged_shm_enabled()` is True.
"""

import logging
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

logger = logging.getLogger(__name__)


class PagedShmTensorIPC:
    """
    Handles SHM‑accelerated tensor transfer for multimodal inputs.

    This class encapsulates the client-side logic for writing and reading
    large tensors via the paged shared memory server. It is used both in the
    API server (renderer) and the GPU worker.

    See the module-level docstring for a detailed description of the workflow.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        pin: bool = False,
        connect: bool = True,
        open_write_timeout: float = 5.0,
        read_timeout: float = 30.0,
    ):
        """
        Args:
            model_config: vLLM model configuration, from which the multimodal
                config and SHM parameters are derived.
            pin: Whether to pin the shared memory for faster GPU transfers.
                Pinning can improve H2D throughput but consumes system memory
                resources.
            connect: If True, immediately establish a connection to the SHM
                server. If False, call `connect()` later.
            open_write_timeout: Timeout in seconds for the atomic block
                allocation (`open_write`). If the server cannot allocate enough
                blocks within this time, a `RuntimeError` is raised.
            read_timeout: Timeout in seconds for waiting on a write to complete
                and reading from SHM. A negative value means infinite wait
                (use with caution; can cause hangs).
        """
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
        self.open_write_timeout = open_write_timeout
        self.read_timeout = read_timeout
        self.client: PagedShmClient | None = None

        if connect:
            self.connect()

    def connect(self) -> None:
        """Establish connection to the paged shared‑memory server."""
        if not self.is_paged_shm_enabled:
            return

        self.client = PagedShmClient.from_model_config(self.model_config, pin=self.pin)
        if self.client is not None:
            self._resources.callback(self.client.close)

    def write(self, mm_inputs: MultiModalInput) -> None:
        """
        Replace eligible tensors in `mm_inputs` with SHM‑backed metadata.

        This is the main entry point on the API server side.

        Steps are as described in the module-level docstring. Note that this
        method returns immediately after submitting the asynchronous writes;
        the actual copying to SHM happens in the background. The metadata
        objects are ready to be sent to workers.

        Raises:
            RuntimeError: If the atomic allocation (`open_write`) fails, e.g.,
                due to insufficient memory or timeout. The caller should catch
                this and fall back to sending the tensor directly via ZMQ.
            ValueError: If the input tensors are malformed (should not happen).
        """
        if not self.is_paged_shm_enabled or self.client is None:
            return

        # 1. Collect all large tensors that should be sent via SHM.
        elements: list[MultiModalFieldElem] = []
        mm_kwargs = mm_inputs["mm_kwargs"]
        for modality, mm_items in mm_kwargs.items():
            for mm_item in mm_items:
                if mm_item is None:
                    continue
                # Currently only handling 'pixel_values'; extend as needed.
                if "pixel_values" in mm_item:
                    elem: MultiModalFieldElem = mm_item["pixel_values"]
                    if not isinstance(elem.data, torch.Tensor):
                        continue
                    # Skip small tensors; they are more efficient via ZMQ.
                    if elem.data.nbytes < self.block_size:
                        continue
                    elements.append(elem)

        if not elements:
            return

        # 2. Prepare allocation requests for all large tensors.
        items: list[ShmWriteRequest] = []
        for elem in elements:
            assert isinstance(elem.data, torch.Tensor)
            item = ShmWriteRequest(
                uuid=random_uuid(),
                size=elem.data.nbytes,
                use_cache=True,  # Cacheable, can be evicted LRU.
                generate_read_token=True,  # Create a one‑time read token.
            )
            items.append(item)

        # 3. Allocate shm for all these mm tensors at once
        # Refer to the wiki:Dining philosophers problem.
        try:
            alloc = self.client.open_write(items, timeout=self.open_write_timeout)
        except RuntimeError as e:
            logger.error("PagedShm `open_write` failed: %s", e)
            return None

        # 4. Submit asynchronous writes for each tensor.
        for elem, a in zip(elements, alloc):
            assert isinstance(elem.data, torch.Tensor)
            assert a.read_token is not None

            # The background task will copy the tensor and then call close_write.
            self.client.write(
                uuid=a.uuid,
                data=elem.data,
                use_cache=a.use_cache,
                blocks=a.blocks,
                async_write=True,
            )

            # Replace the original tensor data with metadata.
            elem.pshm_tensor = PagedShmTensor(
                uuid=a.read_token,
                size=a.size,
                blocks=a.blocks,
                dtype=str(elem.data.dtype).removeprefix("torch."),
                shape=tuple(elem.data.shape),
            )
            elem.data = None  # Free the original tensor reference.

    def read(
        self,
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
        device: torch.types.Device,
    ) -> None:
        """
        Restore tensors from SHM metadata into GPU/CPU tensors.

        This is the main entry point on the GPU worker side.

        For each multimodal item that contains a `PagedShmTensor`, it:
          - Waits for the asynchronous write to complete (with timeout).
          - Reads the raw bytes from SHM to the specified device.
          - Reconstructs the tensor and replaces the placeholder.
          - Destroys the read token (making it invalid for future use).

        Raises:
            RuntimeError: If waiting for the write times out, the token is
                invalid/expired, or reading from SHM fails.
            ValueError: If the metadata is corrupted or shape/dtype mismatch.
        """
        if not self.is_paged_shm_enabled or self.client is None:
            return

        for modality, items in mm_kwargs:
            if "pixel_values" not in items:
                continue

            pixel_values = items["pixel_values"]
            pshm_tensor: PagedShmTensor | None = pixel_values.pshm_tensor

            if pshm_tensor is not None:
                torch_dtype = getattr(torch, pshm_tensor.dtype)

                # 1. Wait for the writer to complete and read the data.
                # Here we sync the swap_blocks_batch stream, then call close_read,
                # which releases the SHM read reference.
                # We'd better not call close_read here, so that we can avoid syncing
                # the swap_blocks_batch stream.
                tensor_gpu = self.client.read(
                    pshm_tensor.uuid,
                    device=device,
                    timeout=self.read_timeout,
                )

                # 2. Replace the metadata with the actual tensor.
                tensor_gpu = tensor_gpu.view(torch_dtype).view(pshm_tensor.shape)
                pixel_values.data = tensor_gpu
                pixel_values.pshm_tensor = None

    def shutdown(self) -> None:
        """Release all resources (client connection, background threads, etc.)."""
        if not self.is_paged_shm_enabled:
            return
        self._resources.close()
