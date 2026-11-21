# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading
from typing import Protocol

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


class CopyEvent(Protocol):
    """Completion signal for one block transfer, either backend.

    The worker submits transfers without waiting and learns they are done
    by polling these events in get_finished(): DmaCopyBackend posts real
    torch events that flip when the DMA stream reaches them;
    TorchCopyBackend posts a _CompletedEvent once its background thread
    finishes the copy, so the same poll loop serves both backends.
    """

    def query(self) -> bool: ...
    def synchronize(self) -> None: ...


class DmaCopyBackend:
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
        self._load_stream = load_stream
        self._store_stream = store_stream

        # Stores read the live KV cache -> STREAM (paired with the compute-done
        # wait in get_finished); loads read stable pinned host memory -> ANY.
        self._store_params = build_params(
            gpu_caches,
            cpu_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            cpu_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
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
        events_list: list[tuple[int, CopyEvent]],
        wait_event: torch.Event | None = None,
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put(
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            )
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
        current_platform.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            ) = item
            stream = store_stream if is_store else load_stream
            if wait_event is not None:
                stream.wait_event(wait_event)
            copy_blocks(src_blocks, dst_blocks, params)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))


class _CompletedEvent:
    """Completed-event stand-in for a copy that already finished."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class TorchCopyBackend:
    """Generic tensor-copy backend for platforms without CUDA DMA.

    Used where host pages stay pageable and there is no cuMemcpyBatchAsync
    engine: transfers are plain block-indexed copies run on a background
    thread, so the worker thread never stalls per transfer.
    """

    def __init__(self) -> None:
        self._gpu_caches: dict[str, torch.Tensor] = {}
        self._cpu_caches: dict[str, torch.Tensor] = {}
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._error: BaseException | None = None

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        # Unused: kept for signature parity with DmaCopyBackend.init, so
        # the worker can construct either backend the same way.
        device: torch.device,
    ) -> None:
        assert list(gpu_caches.keys()) == list(cpu_caches.keys())
        self._gpu_caches = gpu_caches
        self._cpu_caches = cpu_caches
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, CopyEvent]],
        wait_event: torch.Event | None = None,
    ) -> None:
        if self._error is not None:
            raise RuntimeError("TorchCopyBackend copy thread failed") from self._error
        assert self._queue is not None
        self._queue.put(
            (
                src_blocks,
                dst_blocks,
                is_store,
                event_idx,
                events_list,
                wait_event,
            )
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._error is not None:
            raise RuntimeError("TorchCopyBackend copy thread failed") from self._error

    def _copy_loop(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            if item is None:
                return
            (
                src_blocks,
                dst_blocks,
                is_store,
                event_idx,
                events_list,
                wait_event,
            ) = item
            try:
                if wait_event is not None:
                    # Stores read the live KV cache; the compute stream may
                    # still be writing it. Wait here so the worker thread
                    # never stalls per transfer.
                    wait_event.synchronize()
                if is_store:
                    src, dst = self._gpu_caches, self._cpu_caches
                else:
                    src, dst = self._cpu_caches, self._gpu_caches
                for name, src_tensor in src.items():
                    dst_tensor = dst[name]
                    # Gather, move, then write: torch only moves the source
                    # to the destination device on its own for scalar values,
                    # so a block copy across devices has to do it explicitly.
                    src_idx = torch.as_tensor(
                        src_blocks, dtype=torch.long, device=src_tensor.device
                    )
                    dst_idx = torch.as_tensor(
                        dst_blocks, dtype=torch.long, device=dst_tensor.device
                    )
                    gathered = src_tensor.index_select(0, src_idx)
                    dst_tensor.index_copy_(0, dst_idx, gathered.to(dst_tensor.device))
                events_list.append((event_idx, _CompletedEvent()))
            except Exception as e:
                # A failed copy never posts its event, so the poll loop
                # could not report it: keep the first failure and surface it
                # at shutdown instead of dying silently in the thread.
                if self._error is None:
                    self._error = e
                return
