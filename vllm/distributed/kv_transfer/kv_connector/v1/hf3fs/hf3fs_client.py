# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import multiprocessing
import os
import threading
from functools import cache, wraps
from pathlib import Path
from types import ModuleType

import torch

from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

HF3FS_AVAILABLE = True
try:
    from hf3fs_fuse.io import (
        deregister_fd,
        extract_mount_point,
        make_ioring,
        make_iovec,
        register_fd,
    )
except ImportError:
    HF3FS_AVAILABLE = False


@cache
def _load_cuda_shm_ops() -> ModuleType:
    """JIT-build the CUDA shm copy extension (utils/hf3fs_utils.cpp).

    CUDA only. Built on first use rather than at import time so that importing
    this module works on every platform.
    """
    from torch.utils.cpp_extension import CUDA_HOME, load

    assert CUDA_HOME is not None, "CUDA_HOME is required to build hf3fs_utils"
    root = Path(__file__).parent.resolve()
    return load(
        name="hf3fs_utils",
        sources=[str(root / "utils" / "hf3fs_utils.cpp")],
        extra_include_paths=[os.path.join(CUDA_HOME, "include")],
    )


def rsynchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.rlock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


def wsynchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.wlock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


class Hf3fsClient:
    def __init__(self, path: str, size: int, bytes_per_page: int, entries: int):
        """Initialize the HF3FS client with hf3fs_fuse.

        Args:
            path: Path to the file used for storage
            size: Total size of the storage file in bytes
            bytes_per_page: Size of each page in bytes
            entries: Maximum number of concurrent operations

        """
        if not HF3FS_AVAILABLE:
            raise ImportError(
                "hf3fs_fuse.io is not available. Please install the hf3fs_fuse package."
            )

        self.path = path
        self.size = size
        self.bytes_per_page = bytes_per_page
        self.entries = entries

        self._closed = False

        self.file = None
        self.shm_r = None
        self.shm_w = None
        self.ior_r = None
        self.ior_w = None
        self.iov_r = None
        self.iov_w = None
        try:
            # Create the file if it doesn't exist and set its size
            self.file = os.open(self.path, os.O_RDWR | os.O_CREAT)
            os.ftruncate(self.file, size)
            register_fd(self.file)

            self.hf3fs_mount_point = extract_mount_point(path)
            self.bs = self.bytes_per_page
            self.shm_r = multiprocessing.shared_memory.SharedMemory(
                size=self.bs * self.entries, create=True
            )
            self.shm_w = multiprocessing.shared_memory.SharedMemory(
                size=self.bs * self.entries, create=True
            )

            self.shm_r_tensor = torch.frombuffer(self.shm_r.buf, dtype=torch.uint8)
            self.shm_w_tensor = torch.frombuffer(self.shm_w.buf, dtype=torch.uint8)

            numel = self.bs * self.entries
            self.r_pinned = torch.empty(
                numel,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            self.w_pinned = torch.empty(
                numel,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )

            self.numa = -1
            self.ior_r = make_ioring(
                self.hf3fs_mount_point,
                self.entries,
                for_read=True,
                timeout=1,
                numa=self.numa,
            )
            self.ior_w = make_ioring(
                self.hf3fs_mount_point,
                self.entries,
                for_read=False,
                timeout=1,
                numa=self.numa,
            )
            self.iov_r = make_iovec(self.shm_r, self.hf3fs_mount_point)
            self.iov_w = make_iovec(self.shm_w, self.hf3fs_mount_point)
            self.shm_r.unlink()
            self.shm_w.unlink()

            self.rlock = threading.RLock()
            self.wlock = threading.RLock()

            # shm <-> device copies: native extension on CUDA, generic torch
            # copies on every other accelerator.
            self._cuda_ops: ModuleType | None = None
            if current_platform.is_cuda():
                self._cuda_ops = _load_cuda_shm_ops()
                self.stream = torch.cuda.Stream()
            else:
                self.stream = torch.Stream()

        except Exception:
            self._release_resources()
            raise

        logger.debug(
            "Initialized HF3FS client with file: %s, size: %s bytes", path, size
        )

    def _release_resources(self) -> None:
        """Release all acquired resources safely."""
        # iov must be released before ioring and shm
        for attr in ("iov_r", "iov_w", "ior_r", "ior_w"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)

        for attr in ("shm_r", "shm_w"):
            shm = getattr(self, attr, None)
            if shm is not None:
                try:
                    shm.close()
                except Exception as e:
                    logger.warning("Failed to close %s: %s", attr, e)
                setattr(self, attr, None)

        if self.file is not None:
            try:
                deregister_fd(self.file)
            except Exception as e:
                logger.warning("deregister_fd failed: %s", e)
            try:
                os.close(self.file)
            except OSError as e:
                logger.warning("os.close failed: %s", e)
            self.file = None

    @rsynchronized()
    def batch_read(self, offsets: list[int], tensors: list[torch.Tensor]) -> list[int]:
        """Read data from the file at specified offsets into tensors.

        Args:
            offsets: List of byte offsets to read from
            tensors: List of tensors to read data into

        Returns:
            List of operation results (0 for success, non-zero for error)

        """
        self.check(offsets, tensors)
        assert self.ior_r is not None
        assert self.iov_r is not None

        # prepare
        current = 0
        for offset, tensor in zip(offsets, tensors):
            size = tensor.numel() * tensor.itemsize
            self.ior_r.prepare(
                self.iov_r[current : current + size], True, self.file, offset
            )
            current += size

        # submit
        ionum = len(offsets)
        resv = self.ior_r.submit().wait(min_results=ionum)

        # results
        self._read_shm(tensors)
        results = [res.result for res in resv]

        return results

    @wsynchronized()
    def batch_write(
        self, offsets: list[int], tensors: list[torch.Tensor], event: torch.Event
    ) -> list[int]:
        """Write data from tensors to the file at specified offsets.

        Args:
            offsets: List of byte offsets to write to
            tensors: List of tensors containing data to write
            event: Accelerator event to wait on before reading the tensors

        Returns:
            List of operation results (0 for success, non-zero for error)

        """
        self.check(offsets, tensors)
        assert self.ior_w is not None
        assert self.iov_w is not None

        # prepare
        self._write_shm(tensors, event)

        current = 0
        for offset, tensor in zip(offsets, tensors):
            size = tensor.numel() * tensor.itemsize
            self.ior_w.prepare(
                self.iov_w[current : current + size], False, self.file, offset
            )
            current += size

        # submit
        ionum = len(offsets)
        resv = self.ior_w.submit().wait(min_results=ionum)

        # results
        results = [res.result for res in resv]

        return results

    @staticmethod
    def _byte_view(t: torch.Tensor) -> torch.Tensor:
        """Flat uint8 view over a contiguous tensor's storage."""
        assert t.is_contiguous(), "HF3FS I/O buffers must be contiguous"
        return t.view(-1).view(torch.uint8)

    def _read_shm(self, tensors: list[torch.Tensor]) -> None:
        """Copy packed pages from the read shm into ``tensors``.

        shm -> pinned host buffer -> device (async on ``self.stream``).
        ``self.stream`` does not wait on other streams: the caller must ensure
        no pending device work still touches ``tensors``.
        """
        if self._cuda_ops is not None:
            with torch.cuda.stream(self.stream):
                self._cuda_ops.read_shm(
                    self.shm_r_tensor,
                    self.r_pinned,
                    tensors,
                    self.stream.cuda_stream,
                )
            return

        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        self.r_pinned[:nbytes].copy_(self.shm_r_tensor[:nbytes])
        cur = 0
        with self.stream:
            for t in tensors:
                n = t.numel() * t.element_size()
                self._byte_view(t).copy_(
                    self.r_pinned[cur : cur + n], non_blocking=True
                )
                cur += n
        self.stream.synchronize()

    def _write_shm(self, tensors: list[torch.Tensor], event: torch.Event) -> None:
        """Pack ``tensors`` into the write shm once ``event`` has completed.

        device -> pinned host buffer (async on ``self.stream``) -> shm.
        """
        if self._cuda_ops is not None:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(event)
                self._cuda_ops.write_shm(
                    tensors,
                    self.shm_w_tensor,
                    self.w_pinned,
                    self.stream.cuda_stream,
                )
            return

        cur = 0
        with self.stream:
            self.stream.wait_event(event)
            for t in tensors:
                n = t.numel() * t.element_size()
                self.w_pinned[cur : cur + n].copy_(
                    self._byte_view(t), non_blocking=True
                )
                cur += n
        self.stream.synchronize()
        self.shm_w_tensor[:cur].copy_(self.w_pinned[:cur])

    def check(self, offsets: list[int], tensors: list[torch.Tensor]) -> None:
        sizes = [t.numel() * t.itemsize for t in tensors]
        if any(
            [
                len(offsets) > self.entries,
                len(offsets) != len(sizes),
                any(
                    offset < 0 or offset + size > self.size
                    for offset, size in zip(offsets, sizes)
                ),
                any(size > self.bytes_per_page for size in sizes),
            ]
        ):
            self.close()
            raise ValueError("Hf3fsClient.check Failed")

    def get_size(self) -> int:
        """Get the total size of the storage file.

        Returns:
            Size of the file in bytes

        """
        return self.size

    def close(self) -> None:
        """Close the client and clean up resources."""
        if self._closed:
            return
        self._closed = True
        self._release_resources()

    def flush(self) -> None:
        """Flush any pending writes to disk."""
        if not self._closed and self.file is not None:
            os.fsync(self.file)
