# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk I/O backend for GPU<->NVMe block transfers via pinned double buffer."""

from __future__ import annotations

import os
import queue
import threading

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
    pin_tensor,
)

logger = init_logger(__name__)

O_DIRECT = getattr(os, "O_DIRECT", 0)
_ALIGNMENT = 4096


class DiskBackend:
    """Async disk offload backend with GPU DMA pipeline and double buffering.

    Same launch_copy interface as DmaCopyBackend so the worker can swap
    backends without changing calling code.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._fd: int = -1
        self._total_block_bytes: int = 0
        self._buffer_caches: dict[str, torch.Tensor] = {}
        self._per_tensor_bpb: list[int] = []
        self._tensor_names: list[str] = []

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        disk_path: str,
        num_disk_slots: int,
        total_block_bytes: int,
        num_buffer_slots: int = 2,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._total_block_bytes = total_block_bytes
        self._num_buffer_slots = num_buffer_slots
        self._tensor_names = list(gpu_caches.keys())
        self._per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in gpu_caches.values()
        ]

        assert total_block_bytes % _ALIGNMENT == 0, (
            f"total_block_bytes={total_block_bytes} not aligned to {_ALIGNMENT}"
        )

        self._buffer_caches = {}
        for name, gpu_t in gpu_caches.items():
            bpb = gpu_t.stride(0) * gpu_t.element_size()
            buf = torch.zeros(num_buffer_slots, bpb, dtype=torch.int8, device="cpu")
            pin_tensor(buf)
            assert buf.data_ptr() % _ALIGNMENT == 0, (
                f"Buffer for {name} not {_ALIGNMENT}-byte aligned for O_DIRECT"
            )
            self._buffer_caches[name] = buf

        self._store_params = build_params(
            gpu_caches,
            self._buffer_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            self._buffer_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

        os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        self._fd = os.open(disk_path, os.O_RDWR | os.O_CREAT | O_DIRECT, 0o600)
        os.ftruncate(self._fd, num_disk_slots * total_block_bytes)

        logger.info(
            "DiskBackend: path=%s, slots=%d, total=%.2f GB, buf=%dx%d bytes",
            disk_path,
            num_disk_slots,
            (num_disk_slots * total_block_bytes) / (1024**3),
            num_buffer_slots,
            total_block_bytes,
        )

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._io_loop,
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
        wait_event: torch.Event | None = None,
    ) -> None:
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
            self._thread.join(timeout=10.0)
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def _io_loop(
        self,
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
            (src_blocks, dst_blocks, is_store, event_idx, events_list, wait_event) = (
                item
            )
            stream = store_stream if is_store else load_stream
            if wait_event is not None:
                stream.wait_event(wait_event)
            if is_store:
                self._do_store(src_blocks, dst_blocks, stream)
            else:
                self._do_load(src_blocks, dst_blocks, stream)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _writev_slot(self, buf_slot: int, file_offset: int) -> None:
        bufs = [
            self._buffer_caches[name][buf_slot].numpy() for name in self._tensor_names
        ]
        os.pwritev(self._fd, bufs, file_offset)

    def _readv_slot(self, buf_slot: int, file_offset: int) -> None:
        views = [
            memoryview(self._buffer_caches[name][buf_slot].numpy())
            for name in self._tensor_names
        ]
        os.preadv(self._fd, views, file_offset)

    def _do_store(
        self,
        gpu_blocks: list[int],
        disk_slots: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """GPU → buffer (DMA) → disk (writev), double-buffered.

        Pipeline: launch DMA into slot A, then while writing slot B to
        disk the GPU can DMA into slot A in parallel.
        """
        assert self._store_params is not None
        n = self._num_buffer_slots
        dma_events: list[torch.Event | None] = [None] * n
        pending_writes: list[int | None] = [None] * n

        for i, (gpu_blk, disk_slot) in enumerate(zip(gpu_blocks, disk_slots)):
            buf_slot = i % n
            if dma_events[buf_slot] is not None:
                dma_events[buf_slot].synchronize()
            if pending_writes[buf_slot] is not None:
                self._writev_slot(buf_slot, pending_writes[buf_slot])

            copy_blocks([gpu_blk], [buf_slot], self._store_params)
            ev = torch.Event()
            ev.record(stream)
            dma_events[buf_slot] = ev
            pending_writes[buf_slot] = disk_slot * self._total_block_bytes

        for slot in range(n):
            if dma_events[slot] is not None and pending_writes[slot] is not None:
                dma_events[slot].synchronize()
                self._writev_slot(slot, pending_writes[slot])

    def _do_load(
        self,
        disk_slots: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """Disk (preadv) → buffer → GPU (DMA), double-buffered."""
        assert self._load_params is not None
        n = self._num_buffer_slots
        prev_dma_events: list[torch.Event | None] = [None] * n

        for i, (disk_slot, gpu_blk) in enumerate(zip(disk_slots, gpu_blocks)):
            buf_slot = i % n
            if prev_dma_events[buf_slot] is not None:
                prev_dma_events[buf_slot].synchronize()

            self._readv_slot(buf_slot, disk_slot * self._total_block_bytes)

            copy_blocks([buf_slot], [gpu_blk], self._load_params)
            ev = torch.Event()
            ev.record(stream)
            prev_dma_events[buf_slot] = ev
