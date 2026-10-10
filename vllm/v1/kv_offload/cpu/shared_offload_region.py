# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import errno
import mmap
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import numpy as np
import torch

from vllm.distributed.device_communicators.shm_broadcast import (
    check_shm_free_space,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.host_register import host_unregister

logger = init_logger(__name__)

# MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if os.fstat(fd).st_nlink == 0:
            raise RuntimeError(
                "Shared offload region creator failed during initialization."
            )
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


def _madvise_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    mmap_obj.madvise(_MADV_POPULATE_WRITE, offset, length)


def _fallback_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    # Touch one byte per page via a read-modify-write so existing bytes are
    # preserved — a peer worker may have already written KV data into this
    # shared mmap by the time we run on a kernel without MADV_POPULATE_WRITE.
    arr = np.frombuffer(mmap_obj, dtype=np.uint8)
    arr[offset : offset + length : mmap.PAGESIZE] |= 0


def _get_populate_write_fn(
    mmap_obj: mmap.mmap,
) -> Callable[[mmap.mmap, int, int], None]:
    """Select the pre-faulting method once for this mmap."""
    try:
        _madvise_populate_write(mmap_obj, 0, mmap.PAGESIZE)
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        logger.warning(
            "MADV_POPULATE_WRITE is not supported; falling back to per-page "
            "writes for mmap pre-population. Startup may be slower."
        )
        return _fallback_populate_write
    return _madvise_populate_write


class SharedOffloadRegion(ABC):
    """Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{engine_id}.mmap.  When a barrier is
    given, the path is unlinked once every worker has mapped the file, so
    the kernel reclaims the memory when the last worker exits, no matter
    how it exits; mappings taken before the unlink stay valid.

    Creator-only population pre-faults the entire region before the barrier
    and requires that barrier to keep joiners from using unpopulated pages.
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        engine_id: str,
        num_chunks: int,
        kv_bytes_per_chunk: int,
        barrier: Callable[[], None] | None = None,
        *,
        creator_memory_check: Callable[[int], None] | None = None,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_chunk % self.page_size == 0

        self.num_chunks = num_chunks
        self._row_stride = kv_bytes_per_chunk
        self.total_size_bytes = self.num_chunks * self._row_stride

        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        self._creator = False  # set True only if this worker creates the file
        self.rank = getattr(self, "rank", None)
        self._views: list[torch.Tensor] = []
        self.is_pinned = False
        self.pinned_addresses: list[int] = []
        try:
            try:
                self.fd: int | None = os.open(
                    self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
            except FileExistsError:
                # Joiner path — another worker won O_EXCL. Reopen and wait
                # for the file to reach expected size.
                self.fd = os.open(self.mmap_path, os.O_RDWR)
                _wait_for_file_size(self.fd, self.total_size_bytes)
                logger.info("Opened existing mmap file %s", self.mmap_path)
            else:
                # Creator path. We won O_EXCL, so we own the file: any
                # failure here must clean up so concurrent joiners don't
                # land on a 0-byte stub and spin in _wait_for_file_size
                # for the full 30 s timeout.
                self._creator = True
                if creator_memory_check is not None:
                    creator_memory_check(self.total_size_bytes)
                check_shm_free_space(
                    self.total_size_bytes,
                    allocation_name="CPU KV offload shared region in /dev/shm",
                )
                os.ftruncate(self.fd, self.total_size_bytes)
                logger.info(
                    "Created mmap file %s (%.2f GB)",
                    self.mmap_path,
                    self.total_size_bytes / 1e9,
                )

            self.mmap_obj: mmap.mmap | None = mmap.mmap(
                self.fd,
                self.total_size_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )

            # HiSparse needs creator-only population before the mapping
            # barrier. Other region types leave population to their explicit
            # populate() method after construction.
            self._populate_before_mapping_barrier()
        except Exception:
            if self._creator:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(self.mmap_path)
                self._creator = False
            if hasattr(self, "mmap_obj") and self.mmap_obj is not None:
                self.mmap_obj.close()
                self.mmap_obj = None
            if hasattr(self, "fd") and self.fd is not None:
                os.close(self.fd)
                self.fd = None
            # Peers block inside the barrier until the collective times out if
            # we die before reaching it.  Arrive anyway so every worker calls
            # barrier() exactly once and they fail on their own errors instead
            # of hanging; a failure here must not replace ours.
            if barrier is not None:
                try:
                    barrier()
                except Exception:
                    logger.warning(
                        "Failed to release peers waiting at the mmap barrier",
                        exc_info=True,
                    )
            raise

        if barrier is not None:
            # Every worker has mapped the file once the barrier releases, so
            # its name is no longer needed and dropping it here means no exit
            # path — including SIGKILL — can leak the file.
            try:
                barrier()
            except Exception:
                if self._creator:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(self.mmap_path)
                    self._creator = False
                self.mmap_obj.close()
                os.close(self.fd)
                self.mmap_obj = None
                self.fd = None
                raise
            if self._creator:
                os.unlink(self.mmap_path)
                self._creator = False
                logger.info("Unlinked mmap file %s", self.mmap_path)

        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)

    def _populate_before_mapping_barrier(self) -> None:
        """Hook for region types that populate before the mapping barrier."""
        return None

    @abstractmethod
    def populate(self) -> None:
        """Pre-fault this region according to its layout."""

    @abstractmethod
    def pin(self) -> None:
        """Register this region's host memory for GPU access."""

    def _populate_ranges(
        self, ranges: Iterable[tuple[int, int]], description: str
    ) -> None:
        """Populate the supplied byte ranges with one selected madvise path."""
        assert self.mmap_obj is not None
        populate_write_fn = _get_populate_write_fn(self.mmap_obj)
        _t0 = time.perf_counter()
        count = 0
        for offset, length in ranges:
            populate_write_fn(self.mmap_obj, offset, length)
            count += 1
        logger.debug(
            "MADV_POPULATE_WRITE %s: %d range(s) in %.3f s",
            description,
            count,
            time.perf_counter() - _t0,
        )

    @property
    def base_tensor(self) -> torch.Tensor:
        if self._base is None:
            raise RuntimeError("Shared offload region has been released.")
        return self._base

    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            base_ptr = self._base.data_ptr()
            addresses = self.pinned_addresses or [base_ptr]
            for address in reversed(addresses):
                host_unregister(address)
            self.pinned_addresses.clear()
            self.is_pinned = False
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        if self._creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False


class TensorViewRegion(SharedOffloadRegion, ABC):
    """Shared region whose logical views are tensor-sized slices."""

    @abstractmethod
    def get_view(self, tensor_page_size: int) -> torch.Tensor:
        """Return the next tensor view in this region's layout."""


class MemoryViewRegion(SharedOffloadRegion, ABC):
    """Shared region exposed as one complete row-major memoryview."""

    @abstractmethod
    def get_view(self) -> memoryview:
        """Return the complete zero-copy memoryview."""


class DirectRankRegion(TensorViewRegion):
    """Shared region with one private strided slot per rank."""

    def __init__(
        self,
        engine_id: str,
        num_chunks: int,
        rank: int,
        kv_bytes_per_chunk: int,
        cpu_page_size: int,
        barrier: Callable[[], None] | None = None,
        *,
        creator_memory_check: Callable[[int], None] | None = None,
    ) -> None:
        self.rank = rank
        self._cpu_page_size = cpu_page_size
        self._worker_offset = rank * cpu_page_size
        self._worker_area_end = (rank + 1) * cpu_page_size
        super().__init__(
            engine_id=engine_id,
            num_chunks=num_chunks,
            kv_bytes_per_chunk=kv_bytes_per_chunk,
            barrier=barrier,
            creator_memory_check=creator_memory_check,
        )

    def get_view(self, tensor_page_size: int) -> torch.Tensor:
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        view = torch.as_strided(
            self.base_tensor,
            size=(self.num_chunks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(view)
        return view

    def populate(self) -> None:
        assert self.rank is not None
        page_size = self.page_size
        ranges = []
        for chunk in range(self.num_chunks):
            raw_offset = chunk * self._row_stride + self.rank * self._cpu_page_size
            aligned_offset = (raw_offset // page_size) * page_size
            end = raw_offset + self._cpu_page_size
            ranges.append((aligned_offset, end - aligned_offset))
        self._populate_ranges(ranges, "rank slot")

    def pin(self) -> None:
        from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region

        pin_mmap_region(self)


class ReplicatedRegion(TensorViewRegion):
    """Shared region containing one worker-visible replicated copy."""

    def __init__(
        self,
        engine_id: str,
        num_chunks: int,
        kv_bytes_per_chunk: int,
        cpu_page_size: int,
        barrier: Callable[[], None] | None = None,
        *,
        creator_memory_check: Callable[[int], None] | None = None,
    ) -> None:
        self.rank = 0
        self._cpu_page_size = cpu_page_size
        self._worker_offset = 0
        self._worker_area_end = cpu_page_size
        super().__init__(
            engine_id=engine_id,
            num_chunks=num_chunks,
            kv_bytes_per_chunk=kv_bytes_per_chunk,
            barrier=barrier,
            creator_memory_check=creator_memory_check,
        )

    def get_view(self, tensor_page_size: int) -> torch.Tensor:
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Replicated offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        view = torch.as_strided(
            self.base_tensor,
            size=(self.num_chunks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(view)
        return view

    def populate(self) -> None:
        page_size = self.page_size
        ranges = []
        for chunk in range(self.num_chunks):
            raw_offset = chunk * self._row_stride
            aligned_offset = (raw_offset // page_size) * page_size
            end = raw_offset + self._cpu_page_size
            ranges.append((aligned_offset, end - aligned_offset))
        self._populate_ranges(ranges, "replicated slot")

    def pin(self) -> None:
        from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region

        pin_mmap_region(self)


class GlobalRegion(MemoryViewRegion):
    """Shared region exposed as a complete row-major CPU memoryview."""

    def get_view(self) -> memoryview:
        kv_tensor = self.base_tensor.view(self.num_chunks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self.base_tensor.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def populate(self) -> None:
        self._populate_ranges(((0, self.total_size_bytes),), "entire region")

    def pin(self) -> None:
        # Scheduler-side CPU access does not use CUDA host registration.
        return


class CanonicalRegion(TensorViewRegion):
    """Shared region exposing canonical tensor views."""

    def __init__(
        self,
        engine_id: str,
        num_chunks: int,
        rank: int,
        kv_bytes_per_chunk: int,
        cpu_page_size: int,
        barrier: Callable[[], None] | None = None,
        *,
        creator_memory_check: Callable[[int], None] | None = None,
    ) -> None:
        self.rank = rank
        self._cpu_page_size = cpu_page_size
        self._canonical_offset = 0
        super().__init__(
            engine_id=engine_id,
            num_chunks=num_chunks,
            kv_bytes_per_chunk=kv_bytes_per_chunk,
            barrier=barrier,
            creator_memory_check=creator_memory_check,
        )

    def get_view(self, tensor_page_size: int) -> torch.Tensor:
        new_offset = self._canonical_offset + tensor_page_size
        assert new_offset <= self._row_stride
        view = torch.as_strided(
            self.base_tensor,
            size=(self.num_chunks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._canonical_offset,
        )
        self._canonical_offset = new_offset
        self._views.append(view)
        return view

    def populate(self) -> None:
        assert self.rank is not None
        page_size = self.page_size
        ranges = []
        for chunk in range(self.num_chunks):
            raw_offset = chunk * self._row_stride + self.rank * self._cpu_page_size
            aligned_offset = (raw_offset // page_size) * page_size
            end = raw_offset + self._cpu_page_size
            ranges.append((aligned_offset, end - aligned_offset))
        self._populate_ranges(ranges, "canonical rank slot")

    def pin(self) -> None:
        from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region

        pin_mmap_region(self)


class HiSparseRegion(TensorViewRegion):
    """Shared HiSparse host pool with creator-only population and custom pinning."""

    def __init__(
        self,
        engine_id: str,
        num_chunks: int,
        kv_bytes_per_chunk: int,
        cpu_page_size: int,
        barrier: Callable[[], None],
        registration_ranges: tuple[tuple[int, int], ...],
        *,
        creator_memory_check: Callable[[int], None] | None = None,
    ) -> None:
        self.rank = 0
        self._registration_ranges = registration_ranges
        self._canonical_offset = 0
        super().__init__(
            engine_id=engine_id,
            num_chunks=num_chunks,
            kv_bytes_per_chunk=kv_bytes_per_chunk,
            barrier=barrier,
            creator_memory_check=creator_memory_check,
        )

    def _populate_before_mapping_barrier(self) -> None:
        if self._creator:
            self._populate_ranges(
                ((0, self.total_size_bytes),), "HiSparse creator region"
            )

    def get_view(self, tensor_page_size: int) -> torch.Tensor:
        new_offset = self._canonical_offset + tensor_page_size
        assert new_offset <= self._row_stride
        view = torch.as_strided(
            self.base_tensor,
            size=(self.num_chunks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._canonical_offset,
        )
        self._canonical_offset = new_offset
        self._views.append(view)
        return view

    def populate(self) -> None:
        # Creator-only population already happened before the mapping barrier.
        return

    def pin(self) -> None:
        from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

        for start, end in self._registration_ranges:
            tensor = self.base_tensor[start:end]
            pin_tensor(tensor)
            self.pinned_addresses.append(tensor.data_ptr())
            self.is_pinned = True
