# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    _resolve_max_batch_descriptors,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


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
        # Set by init(staging_bytes > 0): copies go through a staging buffer.
        self._staging: _StagedCopies | None = None

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        staging_bytes: int = 0,
    ) -> None:
        """With `staging_bytes` > 0 the GPU never addresses `cpu_caches`: every
        transfer goes through a GPU-accessible staging buffer of that size (see
        `_StagedCopies`), so `cpu_caches` can be ordinary pageable memory."""
        self._load_stream = load_stream
        self._store_stream = store_stream

        if staging_bytes > 0:
            self._staging = _StagedCopies(
                gpu_caches, cpu_caches, load_stream, store_stream, staging_bytes
            )
            # launch_copy() only checks these are set; staged transfers pick
            # the params of their staging half.
            self._store_params = self._staging.store_params[0]
            self._load_params = self._staging.load_params[0]
        else:
            # Stores read the live KV cache -> STREAM (paired with the
            # compute-done wait in get_finished); loads read stable pinned
            # host memory -> ANY.
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
            args=(self._queue, device, load_stream, store_stream, self._staging),
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
        if self._staging is not None:
            self._staging.shutdown()

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        staging: _StagedCopies | None,
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
            if staging is not None:
                # A staged transfer returns once its last copy is complete
                # (the staging buffer is reused), so its event is already done.
                if is_store:
                    staging.store(src_blocks, dst_blocks, stream)
                else:
                    staging.load(src_blocks, dst_blocks, stream)
            else:
                copy_blocks(src_blocks, dst_blocks, params)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))


# Host-side copy threads of a staged backend (torch's CPU index ops release the
# GIL while copying; one thread moves ~15-25 GB/s).
_STAGING_COPY_THREADS = 4


class _StagedCopies:
    """GPU <-> CPU block transfers through a small GPU-accessible staging buffer.

    The CPU caches are never registered with the GPU runtime. On ROCm a
    registered (`hipHostRegister`) range stays movable: amdgpu mirrors it as a
    userptr through an MMU notifier, and when the kernel migrates any of its
    pages (memory compaction, NUMA balancing, `move_pages(2)`) KFD stops every
    queue of the process until the whole range is valid again, which takes
    about 0.25-0.5 s a GiB of 4 KiB pages: tens of seconds for a CPU KV cache
    of 64-128 GiB, with the worker's GPU and its TP peers stalled. Here only
    the staging buffer (two halves of `staging_bytes / 2`) is GPU-visible.

    A store DMAs a chunk of GPU blocks into one half and copies it into the CPU
    caches on the host while the next chunk's DMA fills the other half; a load
    does the reverse. At most one batch DMA is in flight, and a transfer returns
    once its last copy is complete.
    """

    def __init__(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        staging_bytes: int,
    ) -> None:
        names = list(gpu_caches)
        assert names == list(cpu_caches)
        bpb = {n: t.stride(0) * t.element_size() for n, t in gpu_caches.items()}
        block_bytes = sum(bpb.values())
        slots = max(1, staging_bytes // 2 // block_bytes)
        # One batch call per chunk, so a chunk's DMA is a single op in flight.
        max_desc = _resolve_max_batch_descriptors()
        if max_desc > 0:
            slots = max(1, min(slots, max_desc // len(names)))
        self.slots = slots

        # Two halves, each [slots, block bytes] per layer (block-strided, as the
        # batch copies address them).
        buf = torch.empty(2 * slots * block_bytes, dtype=torch.int8, pin_memory=True)
        self.halves: list[dict[str, torch.Tensor]] = []
        off = 0
        for _ in range(2):
            half = {}
            for n in names:
                half[n] = buf[off : off + slots * bpb[n]].view(slots, bpb[n])
                off += slots * bpb[n]
            self.halves.append(half)
        self.store_params = [
            build_params(
                gpu_caches,
                half,
                store_stream,
                src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
            )
            for half in self.halves
        ]
        self.load_params = [
            build_params(
                half,
                gpu_caches,
                load_stream,
                src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            )
            for half in self.halves
        ]

        # The CPU caches and the halves as [blocks, words] in the widest
        # integer type that divides every row (CPU index_copy_ moves int8 at
        # ~2 GB/s, int64 at ~15 GB/s).
        self._cpu: dict[str, torch.Tensor] = {}
        self._halves_w: list[dict[str, torch.Tensor]] = [{}, {}]
        for n, t in cpu_caches.items():
            assert t.is_contiguous() and t.stride(0) * t.element_size() == bpb[n]
            rows = [t.view(torch.int8).view(t.shape[0], bpb[n])]
            rows += [half[n] for half in self.halves]
            dtype = _widest_word(rows)
            self._cpu[n] = rows[0].view(dtype)
            for h in range(2):
                self._halves_w[h][n] = rows[1 + h].view(dtype)
        self._pool = ThreadPoolExecutor(
            _STAGING_COPY_THREADS, thread_name_prefix="vllm_kv_offload_staging"
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def _host_copy(self, h: int, cpu_ids: list[int], to_cpu: bool):
        """Start copying `len(cpu_ids)` slots of half `h` to (`to_cpu`) or
        from the CPU blocks `cpu_ids` on the copy threads; returns the futures.

        Each thread gets about the same bytes: whole layers when there are at
        least as many layers as threads (fewer, larger index ops: the ops hold
        the GIL while dispatching), otherwise slot ranges of each layer.
        """
        idx = torch.tensor(cpu_ids, dtype=torch.long)
        k = len(cpu_ids)
        half = self._halves_w[h]
        pieces = max(1, min(k, _STAGING_COPY_THREADS // len(self._cpu)))
        ops = []  # (bytes, layer, first slot, end slot)
        for n, cpu in self._cpu.items():
            row = cpu.shape[1] * cpu.element_size()
            bounds = [k * i // pieces for i in range(pieces + 1)]
            ops += [(row * (hi - lo), n, lo, hi) for lo, hi in zip(bounds, bounds[1:])]
        work: list[list[tuple[str, int, int]]] = [
            [] for _ in range(_STAGING_COPY_THREADS)
        ]
        load = [0] * _STAGING_COPY_THREADS
        for nbytes, n, lo, hi in sorted(ops, reverse=True):
            t = load.index(min(load))
            work[t].append((n, lo, hi))
            load[t] += nbytes

        def run(items: list[tuple[str, int, int]]) -> None:
            for n, lo, hi in items:
                if to_cpu:
                    self._cpu[n].index_copy_(0, idx[lo:hi], half[n][lo:hi])
                else:
                    torch.index_select(self._cpu[n], 0, idx[lo:hi], out=half[n][lo:hi])

        return [self._pool.submit(run, items) for items in work if items]

    def store(
        self,
        gpu_ids: list[int],
        cpu_ids: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        chunks = [
            (gpu_ids[i : i + self.slots], cpu_ids[i : i + self.slots])
            for i in range(0, len(gpu_ids), self.slots)
        ]
        if not chunks:
            return
        g0, _ = chunks[0]
        copy_blocks(g0, list(range(len(g0))), self.store_params[0])
        stream.synchronize()
        for i, (_, c) in enumerate(chunks):
            h = i % 2
            if i + 1 < len(chunks):
                g1, _ = chunks[i + 1]
                copy_blocks(g1, list(range(len(g1))), self.store_params[1 - h])
            futures = self._host_copy(h, c, to_cpu=True)
            if i + 1 < len(chunks):
                stream.synchronize()
            for f in futures:
                f.result()

    def load(
        self,
        cpu_ids: list[int],
        gpu_ids: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        chunks = [
            (cpu_ids[i : i + self.slots], gpu_ids[i : i + self.slots])
            for i in range(0, len(cpu_ids), self.slots)
        ]
        if not chunks:
            return
        for f in self._host_copy(0, chunks[0][0], to_cpu=False):
            f.result()
        for i, (_, g) in enumerate(chunks):
            h = i % 2
            copy_blocks(list(range(len(g))), g, self.load_params[h])
            futures = []
            if i + 1 < len(chunks):
                futures = self._host_copy(1 - h, chunks[i + 1][0], to_cpu=False)
            stream.synchronize()
            for f in futures:
                f.result()


def _widest_word(rows: list[torch.Tensor]) -> torch.dtype:
    """The widest integer type whose size divides the row length and the base
    address of every `[n, bytes]` int8 tensor in `rows`."""
    for dtype in (torch.int64, torch.int32, torch.int16):
        size = dtype.itemsize
        if all(r.shape[1] % size == 0 and r.data_ptr() % size == 0 for r in rows):
            return dtype
    return torch.int8
