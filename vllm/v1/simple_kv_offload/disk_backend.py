# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk I/O backend for GPU<->NVMe block transfers via pinned staging buffers.

Uses separate IO threads for store and load so that loads (latency-critical)
never block behind stores (background work). Each thread owns its own pinned
staging buffers to avoid contention.
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.host_memory import pin_tensor, unpin_tensor
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)

O_DIRECT = getattr(os, "O_DIRECT", 0)
_ALIGNMENT = 4096


def _alloc_aligned(num_slots: int, bpb: int) -> torch.Tensor:
    """Allocate a staging buffer whose base address is O_DIRECT aligned.

    The CPU allocator only guarantees 64-byte alignment, so over-allocate by
    one alignment unit and return an aligned view. The view keeps the backing
    storage alive.
    """
    nbytes = num_slots * bpb
    raw = torch.zeros(nbytes + _ALIGNMENT, dtype=torch.int8, device="cpu")
    offset = -raw.data_ptr() % _ALIGNMENT
    return raw[offset : offset + nbytes].view(num_slots, bpb)


class DiskBackend:
    """Async disk offload backend with pipelined GPU DMA and interleaved IO.

    Architecture:
    - Separate coordinator threads for store and load (never block each other)
    - Interleaved pipeline: DMA slot N while preadv/pwritev slot N-1
    - O_DIRECT by default; page cache is opt-in via use_page_cache

    Same launch_copy interface as DmaCopyBackend so the worker can swap
    backends without changing calling code.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._store_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._load_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._store_thread: threading.Thread | None = None
        self._load_thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._fd: int = -1
        self._disk_path: str = ""
        self._total_block_bytes: int = 0
        self._store_buffer_caches: dict[str, torch.Tensor] = {}
        self._load_buffer_caches: dict[str, torch.Tensor] = {}
        self._store_slot_views: list[list[memoryview]] = []
        self._load_slot_views: list[list[memoryview]] = []
        self._per_tensor_bpb: list[int] = []
        self._tensor_names: list[str] = []
        self._hot_caches: dict[str, torch.Tensor] = {}
        self._hot_blocks: OrderedDict[int, int] = OrderedDict()
        self._free_hot_rows: list[int] = []
        self._hot_refs: list[int] = []
        self._hot_lock = threading.Lock()
        self._hot_load_params: BatchMemcpyParams | None = None
        self._read_executor: ThreadPoolExecutor | None = None
        self._error: Exception | None = None

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
        use_page_cache: bool = False,
        num_hot_slots: int = 0,
        max_load_slots: int | None = None,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._total_block_bytes = total_block_bytes
        self._num_buffer_slots = num_buffer_slots
        self._max_load_slots = min(
            max_load_slots or num_buffer_slots, num_buffer_slots + num_hot_slots
        )
        self._peak_load_slots = num_buffer_slots
        self._tensor_names = list(gpu_caches.keys())
        self._per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in gpu_caches.values()
        ]

        # The stride alignment is an O_DIRECT kernel-DMA requirement; buffered
        # I/O through the page cache has no alignment constraint.
        if not use_page_cache:
            assert total_block_bytes % _ALIGNMENT == 0, (
                f"total_block_bytes={total_block_bytes} not aligned to {_ALIGNMENT}"
            )

        # Separate buffer pools for store and load threads
        self._store_buffer_caches = {}
        self._load_buffer_caches = {}
        for name, gpu_t in gpu_caches.items():
            bpb = gpu_t.stride(0) * gpu_t.element_size()
            store_buf = _alloc_aligned(num_buffer_slots, bpb)
            pin_tensor(store_buf)
            self._store_buffer_caches[name] = store_buf
            load_buf = _alloc_aligned(num_buffer_slots, bpb)
            pin_tensor(load_buf)
            self._load_buffer_caches[name] = load_buf

        # Pre-built iovec views per slot (avoid per-transfer .numpy() calls)
        self._store_slot_views = [
            [
                memoryview(self._store_buffer_caches[name][slot].numpy())
                for name in self._tensor_names
            ]
            for slot in range(num_buffer_slots)
        ]
        self._load_slot_views = [
            [
                memoryview(self._load_buffer_caches[name][slot].numpy())
                for name in self._tensor_names
            ]
            for slot in range(num_buffer_slots)
        ]

        self._store_params = build_params(
            gpu_caches,
            self._store_buffer_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            self._load_buffer_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )
        if num_hot_slots:
            for name, gpu_t in gpu_caches.items():
                buf = _alloc_aligned(
                    num_hot_slots, gpu_t.stride(0) * gpu_t.element_size()
                )
                pin_tensor(buf)
                self._hot_caches[name] = buf
            self._free_hot_rows = list(range(num_hot_slots))
            self._hot_refs = [0] * num_hot_slots
            self._hot_load_params = build_params(
                self._hot_caches,
                gpu_caches,
                load_stream,
                src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            )
        if num_hot_slots:
            self._read_executor = ThreadPoolExecutor(
                max_workers=self._max_load_slots, thread_name_prefix="kv_read"
            )

        os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        # Slot contents never outlive the process, so unlink then O_EXCL rather
        # than reopening: a pre-existing file would otherwise keep its own
        # (possibly world-readable) mode, and blocks may encode user prompts.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(disk_path)
        # O_DIRECT by default: page cache would consume the very host DRAM this
        # backend exists to conserve, and doubles the copy on the store path.
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if not use_page_cache:
            flags |= O_DIRECT
        self._fd = os.open(disk_path, flags, 0o600)
        self._disk_path = disk_path
        os.ftruncate(self._fd, num_disk_slots * total_block_bytes)

        logger.info(
            "DiskBackend: path=%s, slots=%d, total=%.2f GB, buf=%dx%d bytes"
            " (page_cache=%s)",
            disk_path,
            num_disk_slots,
            (num_disk_slots * total_block_bytes) / (1024**3),
            num_buffer_slots,
            total_block_bytes,
            use_page_cache,
        )

        self._store_thread = threading.Thread(
            target=self._store_loop,
            args=(device, store_stream),
            daemon=True,
        )
        self._load_thread = threading.Thread(
            target=self._load_loop,
            args=(device, load_stream),
            daemon=True,
        )
        self._store_thread.start()
        self._load_thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
        hot_blocks: list[int] | None = None,
    ) -> None:
        self.check_error()
        q = self._store_queue if is_store else self._load_queue
        q.put((src_blocks, dst_blocks, event_idx, events_list, wait_event, hot_blocks))

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Disk KV transfer failed") from self._error

    def synchronize(self) -> None:
        """Drain queued work before the worker reuses preempted GPU blocks."""
        self.check_error()
        for q in (self._store_queue, self._load_queue):
            done = threading.Event()
            q.put(done)
            while not done.wait(0.1):
                self.check_error()
        self.check_error()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._store_queue.put(None)
        self._load_queue.put(None)
        if self._store_thread is not None:
            self._store_thread.join(timeout=10.0)
        if self._load_thread is not None:
            self._load_thread.join(timeout=10.0)
        # Slot contents can encode user prompts, so drop the name now rather
        # than leaving them readable until the next run overwrites the file.
        # Unlinking only removes the directory entry: any thread still holding
        # the fd keeps writing to the (now anonymous) inode, which the kernel
        # frees once the last fd goes away.
        with contextlib.suppress(OSError):
            os.unlink(self._disk_path)
        # Closing under a still-running IO thread would let the fd number be
        # reused by an unrelated open(), turning its next pwritev into a write
        # into that file. Leaking one fd for the remaining process lifetime is
        # the cheaper failure.
        if any(
            t is not None and t.is_alive()
            for t in (self._store_thread, self._load_thread)
        ):
            logger.warning(
                "IO thread still running after shutdown timeout; leaking fd %d",
                self._fd,
            )
            return
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        if self._read_executor is not None:
            self._read_executor.shutdown(wait=True)
        for pool in (
            self._store_buffer_caches,
            self._load_buffer_caches,
            self._hot_caches,
        ):
            for tensor in pool.values():
                unpin_tensor(tensor)
            pool.clear()
        self._store_slot_views.clear()
        self._load_slot_views.clear()

    def _take_hot_rows(self, count: int) -> list[int]:
        """Caller holds _hot_lock; reserved rows leave the cache until released."""
        rows: list[int] = []
        while self._free_hot_rows and len(rows) < count:
            rows.append(self._free_hot_rows.pop())
        if len(rows) == count:
            return rows
        for block, row in list(self._hot_blocks.items()):
            if len(rows) >= count:
                break
            if not self._hot_refs[row]:
                del self._hot_blocks[block]
                rows.append(row)
        return rows

    def _invalidate_hot(self, disk_block: int) -> None:
        with self._hot_lock:
            row = self._hot_blocks.pop(disk_block, None)
            if row is not None:
                # SimpleCPU pins logical disk slots until loads complete.
                assert not self._hot_refs[row]
                self._free_hot_rows.append(row)

    def _retain_hot(self, buf_slot: int, disk_block: int) -> None:
        with self._hot_lock:
            rows = self._take_hot_rows(1)
        if not rows:
            return
        row = rows[0]
        try:
            for name, tensor in self._hot_caches.items():
                tensor[row].copy_(self._store_buffer_caches[name][buf_slot])
        except Exception:
            with self._hot_lock:
                self._free_hot_rows.append(row)
            raise
        with self._hot_lock:
            self._hot_blocks[disk_block] = row

    def _store_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._store_queue.get()
            if item is None:
                return
            if not self._run_job(item, stream, is_store=True):
                return

    def _run_job(
        self, item: tuple | threading.Event, stream: torch.cuda.Stream, is_store: bool
    ) -> bool:
        try:
            if isinstance(item, threading.Event):
                stream.synchronize()
                item.set()
                return True
            src, dst, idx, events, fence, hot_blocks = item
            if fence is not None:
                stream.wait_event(fence)
            if is_store:
                self._do_store(src, dst, stream, set(hot_blocks or ()))
            elif self._hot_caches:
                self._do_hybrid_load(src, dst, stream)
            else:
                self._do_load(src, dst, stream)
            event = torch.Event()
            event.record(stream)
            events.append((idx, event))
            return True
        except Exception as exc:
            # Never report completion or recycle memory before outstanding DMA.
            try:
                stream.synchronize()
            finally:
                self._error = exc
            return False

    def _writev_slot(self, buf_slot: int, file_offset: int) -> None:
        written = os.pwritev(self._fd, self._store_slot_views[buf_slot], file_offset)
        if written < self._total_block_bytes:
            raise OSError(
                f"Short write: expected {self._total_block_bytes} bytes, "
                f"wrote {written}"
            )

    def _readv_slot(self, buf_slot: int, file_offset: int) -> None:
        self._read_into(self._load_slot_views[buf_slot], file_offset)

    def _read_into(self, views: list[memoryview], file_offset: int) -> None:
        bytes_read = os.preadv(self._fd, views, file_offset)
        if bytes_read < self._total_block_bytes:
            raise OSError(
                f"Short read: expected {self._total_block_bytes} bytes, "
                f"read {bytes_read}"
            )

    def _load_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            if not self._run_job(item, stream, is_store=False):
                return

    def _do_store(
        self,
        gpu_blocks: list[int],
        disk_slots: list[int],
        stream: torch.cuda.Stream,
        hot_blocks: set[int],
    ) -> None:
        """GPU -> buffer (DMA) -> disk (pwritev), interleaved double-buffer."""
        assert self._store_params is not None
        n = self._num_buffer_slots
        # (DMA event, disk block) of the block already staged in each slot.
        pending: list[tuple[torch.Event, int] | None] = [None] * n

        def flush(slot: int, transfer: tuple[torch.Event, int]) -> None:
            event, block = transfer
            event.synchronize()
            self._writev_slot(slot, block * self._total_block_bytes)
            if block in hot_blocks:
                self._retain_hot(slot, block)

        for i, (gpu_blk, disk_slot) in enumerate(zip(gpu_blocks, disk_slots)):
            buf_slot = i % n
            prev = pending[buf_slot]
            if prev is not None:
                flush(buf_slot, prev)

            self._invalidate_hot(disk_slot)
            copy_blocks([gpu_blk], [buf_slot], self._store_params)
            ev = torch.Event()
            ev.record(stream)
            pending[buf_slot] = (ev, disk_slot)

        for slot, last in enumerate(pending):
            if last is not None:
                flush(slot, last)

    def _do_load(
        self,
        disk_slots: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
        while_read: Callable[[Future[None]], bool] | None = None,
    ) -> None:
        """Disk (preadv) -> buffer -> GPU (DMA), interleaved double-buffer."""
        assert self._load_params is not None
        n = self._num_buffer_slots
        prev_dma_events: list[torch.Event | None] = [None] * n

        for i, (disk_slot, gpu_blk) in enumerate(zip(disk_slots, gpu_blocks)):
            buf_slot = i % n
            prev = prev_dma_events[buf_slot]
            if prev is not None:
                prev.synchronize()

            offset = disk_slot * self._total_block_bytes
            if while_read is not None:
                assert self._read_executor is not None
                if not while_read(
                    self._read_executor.submit(self._readv_slot, buf_slot, offset)
                ):
                    while_read = None
            else:
                self._readv_slot(buf_slot, offset)

            copy_blocks([buf_slot], [gpu_blk], self._load_params)
            ev = torch.Event()
            ev.record(stream)
            prev_dma_events[buf_slot] = ev

        for ev in prev_dma_events:
            if ev is not None:
                ev.synchronize()

    def _do_hybrid_load(
        self, disk_blocks: list[int], gpu_blocks: list[int], stream: torch.cuda.Stream
    ) -> None:
        hot, cold = [], []
        with self._hot_lock:
            for disk, gpu in zip(disk_blocks, gpu_blocks):
                row = self._hot_blocks.get(disk)
                if row is None:
                    cold.append((disk, gpu))
                else:
                    self._hot_refs[row] += 1
                    self._hot_blocks.move_to_end(disk)
                    hot.append((row, gpu))
            target = min(self._max_load_slots, (len(cold) + 31) // 32)
            leased = self._take_hot_rows(max(0, target - self._num_buffer_slots))
        active = min(len(cold), self._num_buffer_slots + len(leased))
        if active > self._peak_load_slots:
            self._peak_load_slots = active
            logger.info("Hybrid KV load-slot peak: %d", active)
        hot_index = 0
        previous_hot: torch.Event | None = None

        def hot_batch(read: Future[None] | None = None) -> bool:
            nonlocal hot_index, previous_hot
            if hot_index == len(hot):
                return False
            if previous_hot is not None:
                previous_hot.synchronize()
            # A cold read may finish while the preceding hot DMA is draining.
            if read is not None and read.done():
                return False
            assert self._hot_load_params is not None
            row, gpu = hot[hot_index]
            copy_blocks([row], [gpu], self._hot_load_params)
            hot_index += 1
            previous_hot = torch.Event()
            previous_hot.record(stream)
            return True

        def while_read(read: Future[None]) -> bool:
            try:
                while not read.done() and hot_batch(read):
                    pass
            finally:
                # Even a failed hot copy must drain the read before releasing rows.
                read.result()
            return hot_index < len(hot)

        try:
            if leased:
                self._do_load_parallel(cold, leased, stream, while_read)
            else:
                self._do_load(
                    [disk for disk, _ in cold],
                    [gpu for _, gpu in cold],
                    stream,
                    while_read if hot else None,
                )
            while hot_batch():
                pass
        finally:
            stream.synchronize()
            with self._hot_lock:
                for row, _ in hot:
                    self._hot_refs[row] -= 1
                self._free_hot_rows.extend(leased)

    def _do_load_parallel(
        self,
        cold: list[tuple[int, int]],
        leased: list[int],
        stream: torch.cuda.Stream,
        while_read: Callable[[Future[None]], bool],
    ) -> None:
        assert self._read_executor is not None
        executor = self._read_executor
        assert self._load_params is not None and self._hot_load_params is not None
        slots = [
            (views, self._load_params, i)
            for i, views in enumerate(self._load_slot_views)
        ] + [
            (
                [memoryview(t[row].numpy()) for t in self._hot_caches.values()],
                self._hot_load_params,
                row,
            )
            for row in leased
        ]
        slots = slots[: len(cold)]
        pending = []

        def read(i: int):
            return executor.submit(
                self._read_into,
                slots[i % len(slots)][0],
                cold[i][0] * self._total_block_bytes,
            )

        try:
            for i in range(len(slots)):
                pending.append(read(i))
            for i, (_, gpu) in enumerate(cold):
                index = i % len(slots)
                while_read(pending[index])
                _, params, row = slots[index]
                copy_blocks([row], [gpu], params)
                if i + len(slots) < len(cold):
                    event = torch.Event()
                    event.record(stream)
                    event.synchronize()
                    pending[index] = read(i + len(slots))
        finally:
            for future in pending:
                future.cancel()
            wait(pending)
