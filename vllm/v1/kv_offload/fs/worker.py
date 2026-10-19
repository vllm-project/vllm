# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base for filesystem-backed offloading workers."""

import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import (
    DevicePointers,
    IOCompletion,
    LoadStoreSpec,
    OffloadingWorker,
    OffloadKey,
    TransferResult,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.pointer_utils import iter_pointer_groups

logger = init_logger(__name__)


class FSLoadStoreSpec(LoadStoreSpec):
    """Spec for loading/storing KV blocks from a filesystem tier.

    Carries content-addressed OffloadKeys that the worker resolves to
    file paths via a FileMapper.
    """

    def __init__(self, keys: list[OffloadKey]):
        self.keys = keys


DEFAULT_MAX_THREADS = 400


@dataclass
class _Transfer:
    job_id: int
    completions: list[IOCompletion]
    num_bytes: int
    start_time: float


class FSOffloadingWorker(OffloadingWorker):
    """Abstract base for device <-> filesystem offloading workers.

    Handles: key->path resolution, per-key pointer grouping, and completion
    tracking.  Subclasses control how I/O is dispatched by implementing
    submit_io().
    """

    def __init__(self, file_mapper: FileMapper, block_size_factor: int):
        self._file_mapper = file_mapper
        self._block_size_factor = block_size_factor
        self._transfers: dict[int, _Transfer] = {}

    @abstractmethod
    def submit_io(
        self,
        file_ops: list[tuple[str, list[tuple[int, int, int]]]],
        is_store: bool,
        ready_event: torch.Event,
    ) -> list[IOCompletion]:
        """Submit a batch of per-file I/O operations.

        Args:
            file_ops: list of (file_path, ops) where each ops is a list
                of (device_ptr, size_bytes, file_offset) tuples.
            is_store: True for device->file, False for file->device.
            ready_event: device data is not valid until this event
                completes. Honor it with ready_event.synchronize() on an
                I/O thread, or stream.wait_event(ready_event) on a
                backend stream. Never wait on it on the calling thread.

        Returns:
            IOCompletions whose completion signals the I/O is done.

        """

    def shutdown_dispatch(self) -> None:
        """Stop accepting I/O and wait for in-flight work."""
        try:
            for t in self._transfers.values():
                for c in t.completions:
                    c.result()
        finally:
            self._transfers.clear()

    def shutdown_backend(self) -> None:
        """Release backend resources (e.g. cuFileDriverClose)."""

    def submit_store(
        self, job_id: int, device_ptrs: DevicePointers, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(dst_spec, FSLoadStoreSpec)
        completions, num_bytes = self._submit_io(
            device_ptrs, dst_spec.keys, is_store=True
        )
        self._transfers[job_id] = _Transfer(
            job_id=job_id,
            completions=completions,
            num_bytes=num_bytes,
            start_time=time.perf_counter(),
        )
        return True

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, device_ptrs: DevicePointers
    ) -> bool:
        assert isinstance(src_spec, FSLoadStoreSpec)
        completions, num_bytes = self._submit_io(
            device_ptrs, src_spec.keys, is_store=False
        )
        self._transfers[job_id] = _Transfer(
            job_id=job_id,
            completions=completions,
            num_bytes=num_bytes,
            start_time=time.perf_counter(),
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        finished_ids: list[int] = []
        for job_id, t in self._transfers.items():
            if all(c.done() for c in t.completions):
                success = True
                for c in t.completions:
                    try:
                        c.result()
                    except Exception:
                        logger.exception("I/O failed for job %d", job_id)
                        success = False
                elapsed = time.perf_counter() - t.start_time
                results.append(
                    TransferResult(
                        job_id=t.job_id,
                        success=success,
                        transfer_size=t.num_bytes,
                        transfer_time=elapsed,
                    )
                )
                finished_ids.append(job_id)
        for job_id in finished_ids:
            del self._transfers[job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            t = self._transfers.get(job_id)
            if t is not None:
                for c in t.completions:
                    c.result()

    def shutdown(self) -> None:
        try:
            self.shutdown_dispatch()
        finally:
            self.shutdown_backend()

    def _submit_io(
        self,
        device_ptrs: DevicePointers,
        keys: list[OffloadKey],
        is_store: bool,
    ) -> tuple[list[IOCompletion], int]:
        """Build per-file ops from device pointers and delegate to submit_io."""
        file_ops: list[tuple[str, list[tuple[int, int, int]]]] = []
        total_bytes = 0
        key_idx = 0

        for g in iter_pointer_groups(device_ptrs, self._block_size_factor):
            dev_blk = 0
            for i in range(g.n_chunks):
                key = keys[key_idx]
                key_idx += 1

                file_blk_start = g.skip if i == 0 else 0
                capacity = self._block_size_factor - file_blk_start
                n_blks = min(capacity, g.group_size - dev_blk)

                ops: list[tuple[int, int, int]] = []
                for d in range(g.n_data_refs):
                    base = g.dev_ptr_offset + d * g.group_size + dev_blk
                    blk_size = int(device_ptrs.sizes[base])
                    for b in range(n_blks):
                        idx = base + b
                        file_offset = (
                            d * self._block_size_factor + file_blk_start + b
                        ) * blk_size
                        ops.append(
                            (
                                int(device_ptrs.ptrs[idx]),
                                int(device_ptrs.sizes[idx]),
                                file_offset,
                            )
                        )
                        total_bytes += int(device_ptrs.sizes[idx])

                file_path = self._file_mapper.get_file_name(key)
                file_ops.append((file_path, ops))
                dev_blk += n_blks

        assert key_idx == len(keys)
        ready_event = torch.Event()
        ready_event.record()
        completions = self.submit_io(file_ops, is_store, ready_event)
        return completions, total_bytes


class ThreadPoolFSWorker(FSOffloadingWorker):
    """FSOffloadingWorker that dispatches per-file I/O to a thread pool.

    Subclasses implement write_block / read_block for the actual I/O.
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        block_size_factor: int,
        device: torch.device | None = None,
        max_io_threads: int = DEFAULT_MAX_THREADS,
        pool_initializer=None,
    ):
        super().__init__(file_mapper=file_mapper, block_size_factor=block_size_factor)

        def _init() -> None:
            if device is not None:
                current_platform.set_device(device)
            if pool_initializer is not None:
                pool_initializer()

        self._pool = ThreadPoolExecutor(max_workers=max_io_threads, initializer=_init)

    @abstractmethod
    def write_block(self, file_path: str, ops: list[tuple[int, int, int]]) -> None:
        """Write device data to a file.

        Args:
            file_path: destination file path.
            ops: list of (device_ptr, size_bytes, file_offset) tuples.

        """

    @abstractmethod
    def read_block(self, file_path: str, ops: list[tuple[int, int, int]]) -> None:
        """Read file data into device memory.

        Args:
            file_path: source file path.
            ops: list of (device_ptr, size_bytes, file_offset) tuples.

        """

    def submit_io(
        self,
        file_ops: list[tuple[str, list[tuple[int, int, int]]]],
        is_store: bool,
        ready_event: torch.Event,
    ) -> list[IOCompletion]:
        io_fn = self.write_block if is_store else self.read_block

        def _do_io(
            event: torch.Event,
            fn,
            path: str,
            ops: list[tuple[int, int, int]],
        ):
            event.synchronize()
            return fn(path, ops)

        return [
            self._pool.submit(_do_io, ready_event, io_fn, path, ops)
            for path, ops in file_ops
        ]

    def shutdown_backend(self) -> None:
        self._pool.shutdown(wait=True)
