# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import functools
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    prepare_copy_blocks,
    submit_prepared_copy,
)

logger = init_logger(__name__)

# Number of (start, end) event pairs preallocated by _EventPairPool.
_EVENT_POOL_INITIAL_SIZE = 16


@dataclass
class DmaCopyEvent:
    """A single queued copy op's timing/completion events.

    ``end_event`` doubles as the completion event: both events are recorded
    on the same stream, so FIFO stream order means ``end_event.query()``
    becoming true implies the op (and everything queued before it on that
    stream) has completed. A separate completion event would be redundant.
    """

    event_idx: int
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    is_store: bool
    release: Callable[[], None]


class _EventPairPool:
    """Reusable pool of (start, end) timing-event pairs.

    Preallocates `_EVENT_POOL_INITIAL_SIZE` pairs up front so the hot copy
    loop never calls `torch.Event(enable_timing=True)` per-op: timing events
    hold CUDA driver resources, and per-op creation/GC of them pressures the
    driver when many small transfers are launched. Grows on demand if
    exhausted by a burst of in-flight ops; growth is bounded by however many
    ops are concurrently in flight, not per-op.
    """

    def __init__(self, size: int = _EVENT_POOL_INITIAL_SIZE) -> None:
        self._pairs: list[tuple[torch.Event, torch.Event]] = [
            (torch.Event(enable_timing=True), torch.Event(enable_timing=True))
            for _ in range(size)
        ]

    def acquire(self) -> tuple[torch.Event, torch.Event]:
        if self._pairs:
            return self._pairs.pop()
        return (torch.Event(enable_timing=True), torch.Event(enable_timing=True))

    def release(self, pair: tuple[torch.Event, torch.Event]) -> None:
        self._pairs.append(pair)


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
        self._event_pool: _EventPairPool | None = None

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
        self._event_pool = _EventPairPool()

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
            args=(self._queue, device, load_stream, store_stream, self._event_pool),
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[DmaCopyEvent],
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
        event_pool: _EventPairPool,
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
            # Host prep (pure NumPy) happens first, outside the timed region,
            # so DMA timing below excludes it.
            prepared = prepare_copy_blocks(src_blocks, dst_blocks, params)
            start_event, end_event = event_pool.acquire()
            release = functools.partial(event_pool.release, (start_event, end_event))
            if wait_event is not None:
                # #46278: stream-order the copy after compute-done, before
                # start_event is recorded, so timing excludes the wait.
                stream.wait_event(wait_event)
            start_event.record(stream)
            if prepared is not None:
                submit_prepared_copy(prepared, params)
            end_event.record(stream)
            events_list.append(
                DmaCopyEvent(
                    event_idx=event_idx,
                    start_event=start_event,
                    end_event=end_event,
                    num_bytes=prepared.total_bytes if prepared is not None else 0,
                    is_store=is_store,
                    release=release,
                )
            )
