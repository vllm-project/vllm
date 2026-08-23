# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PagedShmTensorIPC: Accelerate multimodal tensor transfer via shared memory.

This module provides a high-performance mechanism for transmitting large
multimodal tensors (e.g., pixel values, video frames) between the API server
and GPU workers using paged shared memory (SHM), significantly reducing the
overhead of ZMQ IPC.

The primary use case is within vLLM's multimodal serving pipeline:

- **API Server (Renderer)**: When processing a request with large multimodal
  inputs, the renderer calls `write()` to replace these tensors with lightweight
  metadata (`PagedShmTensor`). The actual tensor data is asynchronously copied
  into a paged SHM pool managed by a separate server process. Only the small
  metadata (UUID, block list, shape, dtype) is sent over ZMQ to the GPU worker.

- **GPU Worker (EncoderRunner)**: The worker receives the metadata and calls
  `read()` to wait for the asynchronous write to complete, then reads the
  tensor data directly from SHM into GPU memory (or CPU), and finally releases
  the read token.

**Detailed workflow in `write()`:**

1. **Identify large tensors**: Scan the `mm_inputs` for tensors whose size
   exceeds the configured `block_size`. These are candidates for SHM transfer.
   (Small tensors are left untouched and sent via ZMQ as usual.)

2. **Batch allocation request**: For each candidate, create a `ShmWriteRequest`
   with a random UUID, the tensor size, `use_cache=True`, and
   `generate_read_token=True`. All requests are collected into a batch.

3. **Atomic allocation (`open_write`)**: The batch is sent to the SHM server
   which attempts to allocate the required number of blocks atomically. This
   avoids the "dining philosophers" problem by preventing partial allocation.
   If memory is insufficient or the timeout expires, a `RuntimeError` is raised.

4. **Asynchronous write submission**: For each allocated block, an asynchronous
   write task is submitted to the client's thread pool. This task copies the
   tensor data into the SHM blocks and then calls `close_write` to make the
   item readable. The main thread does not block on these writes.

5. **Metadata replacement**: The original tensor `.data` is set to `None` and
   replaced by a `PagedShmTensor` object containing the read token (which acts
   as a secure, one‑time‑use reference), block list, size, dtype, and shape.
   This metadata is then transmitted via ZMQ to the worker.

**Detailed workflow in `read()`:**

1. **Iterate over metadata**: For each multimodal item that has a
   `PagedShmTensor` attached, the worker extracts the read token and other info.

2. **Wait for completion (`wait_write`)**: Using the read token, the worker
   waits for the asynchronous write to finish. A configurable timeout prevents
   indefinite blocking.

3. **Read from SHM**: The tensor data is read directly from the SHM blocks into
   the specified device (GPU or CPU) using batched H2D copies for efficiency.

4. **Tensor reconstruction**: The raw bytes are reshaped and cast to the
   original dtype and shape, restoring the full tensor.

5. **Token destruction (`close_read`)**: The read token is consumed/destroyed.
   After this call, the token cannot be used again. This ensures that the
   underlying read reference is released and the blocks become eligible for
   eviction.

**Current limitations and future work:**

- **Tensor parallelism (TP)**
- **Partial write failures**
- **Timeout configuration**

**Performance impact:**

- For a typical 10 MiB multimodal tensor, ZMQ IPC would take about 1–2 ms
  normally, but can spike to ~10 ms under load. SHM transfer reduces this to
  sub‑millisecond copying, with the main overhead being allocation and
  synchronization. The overall benefit is most pronounced when multiple large
  tensors are batched together.

The module is enabled only when the model configuration enables paged SHM
(`multimodal_config.is_paged_shm_enabled()`).
"""

import logging
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
        start = time.perf_counter()
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

        end = time.perf_counter()
        elapsed_time = end - start
        print(
            f"PagedShmTensorIPC.write {elapsed_time * 1000} ms",
        )

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

        ⚠️ **Tensor Parallelism**: As noted in the module docstring, the token
        is single‑use. In a TP setup, only the first rank that calls `read()`
        will succeed; subsequent ranks will receive a token‑not‑found error.
        Coordination is required (e.g., rank 0 reads and broadcasts).

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
                # 1. wait for the write operation to complete.
                # 2. reads the data from the shared memory.
                # 3. release the shared memory.
                # todo: support tp:
                #  wait & reads
                #  But 'release' shouldn't be here.
                tensor_gpu = self.client.read(
                    pshm_tensor.uuid,
                    device=device,
                    timeout=self.read_timeout,
                )
                tensor_gpu = tensor_gpu.view(torch_dtype).view(pshm_tensor.shape)
                pixel_values.data = tensor_gpu

    def shutdown(self) -> None:
        """Release all resources (client connection, background threads, etc.)."""
        if not self.is_paged_shm_enabled:
            return
        self._resources.close()
