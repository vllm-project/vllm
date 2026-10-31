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
import time

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
    unpin_tensor,
)

logger = init_logger(__name__)

O_DIRECT = getattr(os, "O_DIRECT", 0)
_ALIGNMENT = 4096
# Total wait for both IO threads at shutdown. The file is unlinked before this
# wait, so the bound only decides whether the fd can be closed and the staging
# buffers unregistered safely.
_SHUTDOWN_JOIN_TIMEOUT_S = 2.0


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
        # (st_dev, st_ino) of the file this backend created; None once removed.
        self._file_id: tuple[int, int] | None = None
        self._pinned_buffers: list[torch.Tensor] = []
        self._total_block_bytes: int = 0
        self._store_buffer_caches: dict[str, torch.Tensor] = {}
        self._load_buffer_caches: dict[str, torch.Tensor] = {}
        self._store_slot_views: list[list[memoryview]] = []
        self._load_slot_views: list[list[memoryview]] = []
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
        use_page_cache: bool = False,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._total_block_bytes = total_block_bytes
        self._num_buffer_slots = num_buffer_slots
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
            self._pin(store_buf)
            self._store_buffer_caches[name] = store_buf
            load_buf = _alloc_aligned(num_buffer_slots, bpb)
            self._pin(load_buf)
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
        # Remember which inode we created so shutdown() only ever removes this
        # file, even if another process has since reused the same path.
        st = os.fstat(self._fd)
        self._file_id = (st.st_dev, st.st_ino)
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

    def _pin(self, buf: torch.Tensor) -> None:
        """Pin (cudaHostRegister) a staging buffer and record it for shutdown()."""
        pin_tensor(buf)
        self._pinned_buffers.append(buf)

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        q = self._store_queue if is_store else self._load_queue
        q.put((src_blocks, dst_blocks, event_idx, events_list, wait_event))

    def shutdown(self) -> None:
        """Remove the backing file, stop the IO threads and release the
        pinned staging buffers.

        Idempotent, and a no-op for the file if init() never created one.
        Never raises: it runs on the engine teardown path.
        """
        if self._shutdown:
            return
        self._shutdown = True
        # Slot contents can encode user prompts, so drop the name first: it is
        # O(1), and the process may be killed before the joins below return.
        # Unlinking only removes the directory entry: any thread still holding
        # the fd keeps writing to the (now anonymous) inode, which the kernel
        # frees once the last fd goes away.
        self._unlink_own_file()
        # The loops check _shutdown, so queued transfers are dropped instead of
        # drained: their results would only land in the unlinked file.
        self._store_queue.put(None)
        self._load_queue.put(None)
        deadline = time.monotonic() + _SHUTDOWN_JOIN_TIMEOUT_S
        for t in (self._store_thread, self._load_thread):
            if t is not None and t.is_alive():
                t.join(timeout=max(0.0, deadline - time.monotonic()))
        # A still-running IO thread may yet pwritev/preadv through the fd and
        # DMA into its staging buffers. Closing the fd would let its number be
        # reused by an unrelated open(), turning the next pwritev into a write
        # into that file; unregistering the buffers would let the allocator
        # hand that memory out again under the DMA. Leaking both for the
        # remaining process lifetime is the cheaper failure.
        if any(
            t is not None and t.is_alive()
            for t in (self._store_thread, self._load_thread)
        ):
            logger.warning(
                "DiskBackend: IO thread still running after the %.1f s shutdown "
                "timeout; leaking fd %d and keeping %d staging buffers registered",
                _SHUTDOWN_JOIN_TIMEOUT_S,
                self._fd,
                len(self._pinned_buffers),
            )
            return
        if self._fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = -1
        self._release_staging_buffers()

    def _release_staging_buffers(self) -> None:
        """Unregister every staging buffer init() pinned, then drop the
        references so the memory can be freed.

        Only call this once both IO threads have exited. Without it, freed
        buffers stay registered, and the next cudaHostRegister of a range the
        allocator hands out again (e.g. another DiskBackend in the same
        process) fails with cudaErrorHostMemoryAlreadyRegistered.
        """
        # The iovecs and DMA params point into the buffers; drop them with the
        # buffers so nothing can reach the memory once it is freed.
        self._store_slot_views = []
        self._load_slot_views = []
        self._store_params = None
        self._load_params = None
        self._store_buffer_caches = {}
        self._load_buffer_caches = {}
        still_pinned: list[torch.Tensor] = []
        error: Exception | None = None
        for buf in self._pinned_buffers:
            try:
                unpin_tensor(buf)
            except Exception as e:
                still_pinned.append(buf)
                error = e
        # Keep (leak) any buffer that is still registered: freeing it would
        # recreate the stale registration this method exists to avoid.
        self._pinned_buffers = still_pinned
        if still_pinned:
            logger.warning(
                "DiskBackend: %d staging buffers could not be unregistered and "
                "stay allocated: %s",
                len(still_pinned),
                error,
            )

    def _unlink_own_file(self) -> None:
        """Unlink disk_path only if it is still the inode init() created."""
        file_id, self._file_id = self._file_id, None
        if file_id is None:
            return
        try:
            st = os.lstat(self._disk_path)
            if (st.st_dev, st.st_ino) != file_id:
                # Another process (e.g. the next start on this node) has
                # replaced the path with its own file; leave that one alone.
                logger.info(
                    "DiskBackend: %s no longer refers to this process's file; "
                    "not removing it",
                    self._disk_path,
                )
                return
            os.unlink(self._disk_path)
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning("DiskBackend: failed to remove %s: %s", self._disk_path, e)
            return
        logger.info("DiskBackend: removed %s", self._disk_path)

    def _store_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._store_queue.get()
            if item is None or self._shutdown:
                return
            (src_blocks, dst_blocks, event_idx, events_list, wait_event) = item
            if wait_event is not None:
                stream.wait_event(wait_event)
            self._do_store(src_blocks, dst_blocks, stream)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _writev_slot(self, buf_slot: int, file_offset: int) -> None:
        written = os.pwritev(self._fd, self._store_slot_views[buf_slot], file_offset)
        if written < self._total_block_bytes:
            raise OSError(
                f"Short write: expected {self._total_block_bytes} bytes, "
                f"wrote {written}"
            )

    def _readv_slot(self, buf_slot: int, file_offset: int) -> None:
        bytes_read = os.preadv(self._fd, self._load_slot_views[buf_slot], file_offset)
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
            if item is None or self._shutdown:
                return
            (src_blocks, dst_blocks, event_idx, events_list, wait_event) = item
            if wait_event is not None:
                stream.wait_event(wait_event)
            self._do_load(src_blocks, dst_blocks, stream)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _do_store(
        self,
        gpu_blocks: list[int],
        disk_slots: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """GPU -> buffer (DMA) -> disk (pwritev), interleaved double-buffer."""
        assert self._store_params is not None
        n = self._num_buffer_slots
        # (DMA event, file offset) of the block already staged in each slot.
        pending: list[tuple[torch.Event, int] | None] = [None] * n

        for i, (gpu_blk, disk_slot) in enumerate(zip(gpu_blocks, disk_slots)):
            if self._shutdown:
                break
            buf_slot = i % n
            prev = pending[buf_slot]
            if prev is not None:
                prev[0].synchronize()
                self._writev_slot(buf_slot, prev[1])

            copy_blocks([gpu_blk], [buf_slot], self._store_params)
            ev = torch.Event()
            ev.record(stream)
            pending[buf_slot] = (ev, disk_slot * self._total_block_bytes)

        # Wait for every DMA into the staging buffers even when the writes are
        # dropped: shutdown() unregisters the buffers once this thread exits.
        for slot, last in enumerate(pending):
            if last is not None:
                last[0].synchronize()
                if not self._shutdown:
                    self._writev_slot(slot, last[1])

    def _do_load(
        self,
        disk_slots: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """Disk (preadv) -> buffer -> GPU (DMA), interleaved double-buffer."""
        assert self._load_params is not None
        n = self._num_buffer_slots
        prev_dma_events: list[torch.Event | None] = [None] * n

        for i, (disk_slot, gpu_blk) in enumerate(zip(disk_slots, gpu_blocks)):
            if self._shutdown:
                break
            buf_slot = i % n
            prev = prev_dma_events[buf_slot]
            if prev is not None:
                prev.synchronize()

            self._readv_slot(buf_slot, disk_slot * self._total_block_bytes)

            copy_blocks([buf_slot], [gpu_blk], self._load_params)
            ev = torch.Event()
            ev.record(stream)
            prev_dma_events[buf_slot] = ev

        # Also on shutdown: the DMAs read the staging buffers, which shutdown()
        # unregisters once this thread exits.
        for ev in prev_dma_events:
            if ev is not None:
                ev.synchronize()
