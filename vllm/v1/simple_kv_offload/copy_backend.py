# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable copy backends for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.simple_kv_offload.copy_ops import BatchMemcpyParams
    from vllm.v1.simple_kv_offload.copy_ops_kernel import LaunchParams

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class CopyBackend(ABC):
    """Abstract interface for GPU<->CPU block copy backends."""

    @abstractmethod
    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        """One-time setup after KV caches are allocated.

        Builds TWO sets of internal params (store: gpu->cpu, load: cpu->gpu).
        Caches must not be reallocated after this call.
        """

    @abstractmethod
    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
    ) -> None:
        """Launch a block copy. Appends (event_idx, Event) to events_list."""

    def shutdown(self) -> None:  # noqa: B027
        """Clean up resources. Idempotent."""


class KernelCopyBackend(CopyBackend):
    """Triton kernel copy backend (non-blocking, no thread)."""

    def __init__(self) -> None:
        self._store_params: LaunchParams | None = None
        self._load_params: LaunchParams | None = None
        self._gpu_caches: dict[str, torch.Tensor] | None = None
        self._cpu_caches: dict[str, torch.Tensor] | None = None
        self._device: torch.device | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        from vllm.v1.simple_kv_offload import copy_ops_kernel

        self._gpu_caches = gpu_caches
        self._cpu_caches = cpu_caches
        self._device = device
        self._load_stream = load_stream
        self._store_stream = store_stream

        # Store direction: gpu -> cpu
        self._store_params = copy_ops_kernel.build_params(gpu_caches, cpu_caches)
        # Load direction: cpu -> gpu
        self._load_params = copy_ops_kernel.build_params(cpu_caches, gpu_caches)

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
    ) -> None:
        from vllm.v1.simple_kv_offload import copy_ops_kernel

        if is_store:
            params = self._store_params
            stream = self._store_stream
            src_caches = self._gpu_caches
            dst_caches = self._cpu_caches
        else:
            params = self._load_params
            stream = self._load_stream
            src_caches = self._cpu_caches
            dst_caches = self._gpu_caches

        assert params is not None and stream is not None
        assert src_caches is not None and dst_caches is not None

        with torch.cuda.stream(stream):
            # block_mapping must be created on the copy stream so the
            # Triton kernel (also on this stream) doesn't read it before
            # the GPU allocation/copy from the default stream completes.
            block_mapping = torch.tensor(
                list(zip(src_blocks, dst_blocks)),
                dtype=torch.int64,
                device=self._device,
            )
            copy_ops_kernel.copy_blocks(
                src_caches,
                dst_caches,
                block_mapping,
                launch_params=params,
            )
            # Prevent the caching allocator from recycling block_mapping
            # after it goes out of scope — the kernel is still reading it
            # asynchronously on the copy stream.
            block_mapping.record_stream(stream)

        event = torch.Event()
        event.record(stream)
        events_list.append((event_idx, event))


class DmaCopyBackend(CopyBackend):
    """cuMemcpyBatchAsync copy backend (background thread)."""

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        from vllm.v1.simple_kv_offload import copy_ops

        self._load_stream = load_stream
        self._store_stream = store_stream

        # Store direction: gpu -> cpu (on store_stream)
        self._store_params = copy_ops.build_params(
            gpu_caches,
            cpu_caches,
            stream=store_stream,
        )
        # Load direction: cpu -> gpu (on load_stream)
        self._load_params = copy_ops.build_params(
            cpu_caches,
            gpu_caches,
            stream=load_stream,
        )

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put(
            (src_blocks, dst_blocks, params, is_store, event_idx, events_list)
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        from vllm.v1.simple_kv_offload import copy_ops

        current_platform.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            src_blocks, dst_blocks, params, is_store, event_idx, events_list = item
            copy_ops.copy_blocks(src_blocks, dst_blocks, params)
            stream = store_stream if is_store else load_stream
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))


def get_copy_backend(name: str = "kernel") -> CopyBackend:
    """Factory for copy backends."""
    if name == "kernel":
        return KernelCopyBackend()
    elif name == "dma":
        return DmaCopyBackend()
    else:
        raise ValueError(f"Unknown copy backend: {name!r}. Choose 'kernel' or 'dma'.")
