# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Raw-tensor filesystem I/O for encoder-cache entries."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import suppress
from itertools import islice
from pathlib import Path

_DEFAULT_IOV_MAX = 1024


def _get_iov_max() -> int:
    """Return the system limit for buffers in one readv/writev call."""
    try:
        return max(1, int(os.sysconf("SC_IOV_MAX")))
    except (AttributeError, OSError, ValueError):
        return _DEFAULT_IOV_MAX


_IOV_MAX = _get_iov_max()


def _prepare_buffers(
    buffers: Sequence[memoryview],
    file_size_bytes: int,
    *,
    writable: bool,
) -> deque[memoryview]:
    """Validate buffers and return zero-copy views measured in bytes."""
    if file_size_bytes <= 0:
        raise ValueError("file_size_bytes must be positive")

    prepared: deque[memoryview] = deque()
    total_size = 0
    for buffer in buffers:
        try:
            byte_view = buffer.cast("B")
        except TypeError as exc:
            raise ValueError("buffers must be contiguous") from exc
        if writable and byte_view.readonly:
            raise ValueError("destination buffers must be writable")
        if not byte_view:
            continue

        prepared.append(byte_view)
        total_size += len(byte_view)

    if total_size != file_size_bytes:
        raise ValueError(
            f"buffer size is {total_size} bytes, "
            f"expected exactly {file_size_bytes}"
        )
    return prepared


def _consume(buffers: deque[memoryview], completed_size_bytes: int) -> None:
    """Advance pending views past bytes completed by the last I/O call."""
    remaining = completed_size_bytes
    while remaining:
        buffer = buffers[0]
        if remaining < len(buffer):
            buffers[0] = buffer[remaining:]
            return
        remaining -= len(buffer)
        buffers.popleft()


def _transfer_all(
    fd: int,
    buffers: deque[memoryview],
    transfer: Callable[[int, Sequence[memoryview]], int],
    operation: str,
) -> None:
    """Run readv/writev until all buffers are transferred."""
    while buffers:
        batch = list(islice(buffers, 0, _IOV_MAX))
        transferred = transfer(fd, batch)
        if transferred <= 0:
            raise OSError(f"short {operation}: no forward progress")
        _consume(buffers, transferred)


def store_entry(
    entry_path: Path,
    temporary_path: Path,
    sources: Sequence[memoryview],
    file_size_bytes: int,
    *,
    fsync: bool = False,
) -> None:
    """Write a complete entry to a temporary file, then publish atomically."""
    if temporary_path == entry_path or temporary_path.parent != entry_path.parent:
        raise ValueError("temporary file must be beside and differ from final file")

    buffers = _prepare_buffers(sources, file_size_bytes, writable=False)
    entry_path.parent.mkdir(parents=True, exist_ok=True)

    fd: int | None = None
    try:
        fd = os.open(
            temporary_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
        _transfer_all(fd, buffers, os.writev, "write")
        if fsync:
            os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary_path, entry_path)
    except BaseException:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
        with suppress(OSError):
            temporary_path.unlink()
        raise


def load_entry(
    entry_path: Path,
    destinations: Sequence[memoryview],
    file_size_bytes: int,
) -> None:
    """Read a complete entry into writable CPU destination buffers."""
    buffers = _prepare_buffers(destinations, file_size_bytes, writable=True)
    fd = os.open(entry_path, os.O_RDONLY)
    try:
        actual_size = os.fstat(fd).st_size
        if actual_size != file_size_bytes:
            raise OSError(
                f"encoder-cache file size is {actual_size} bytes, "
                f"expected {file_size_bytes}"
            )
        _transfer_all(fd, buffers, os.readv, "read")
    finally:
        os.close(fd)


def remove_entry(entry_path: Path) -> None:
    """Remove an entry file, treating an already absent file as success."""
    entry_path.unlink(missing_ok=True)
