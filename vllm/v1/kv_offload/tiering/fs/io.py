# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import logging
import mmap
import os
import random
import threading

logger = logging.getLogger(__name__)

# O_DIRECT is Linux-specific and not available on macOS
O_DIRECT = getattr(os, "O_DIRECT", 0)

# Thread-local storage for unique temporary file suffixes
_thread_local = threading.local()


def _get_tmp_suffix() -> str:
    """Generate a thread-local unique suffix for temporary files."""
    try:
        return _thread_local.tmp_suffix
    except AttributeError:
        _thread_local.tmp_suffix = f"_{random.randint(0, 2**63 - 1)}.tmp"
        return _thread_local.tmp_suffix


def probe_o_direct(directory: str) -> bool:
    """Return whether ``O_DIRECT`` I/O works in *directory*.

    ``O_DIRECT`` is unsupported on some filesystems (e.g. the overlayfs backing
    a container ``/tmp``, older tmpfs, or some NFS mounts), where opening or
    writing a file with it fails with ``EINVAL``. Probe once with an aligned
    single-page write so callers can fall back to buffered I/O instead of
    failing on every block.
    """
    if not O_DIRECT:
        return False
    path = os.path.join(directory, f".o_direct_probe{_get_tmp_suffix()}")
    page = mmap.mmap(-1, mmap.PAGESIZE)
    try:
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | O_DIRECT, 0o644)
        try:
            os.write(fd, page)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False
    finally:
        page.close()
        with contextlib.suppress(OSError):
            os.remove(path)


def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
    use_o_direct: bool = True,
) -> None:
    """
    Store callback: Writes to a temp file then atomically replaces the destination.
    """
    # Check if block already exists to avoid redundant writes
    if os.path.exists(dest_path):
        return

    tmp_path = dest_path + _get_tmp_suffix()
    # Ensure parent directories exist
    _ensure_dirs(dest_path)

    # Write block atomically. Cast to a flat byte view so the slice uses byte
    # indices; the raw memoryview may be multi-dimensional with itemsize > 1.
    view_slice = buffer.cast("B")[offset : offset + block_size]
    o_direct = O_DIRECT if use_o_direct else 0
    try:
        fd = os.open(
            tmp_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | o_direct,
            0o644,
        )
        try:
            written = os.write(fd, view_slice)
            if written < len(view_slice):
                raise OSError(
                    f"Short write: expected {len(view_slice)} bytes, wrote {written}"
                )
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp file %s: %s", tmp_path, cleanup_exc)
        raise


def load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    use_o_direct: bool = True,
) -> None:
    """
    Load callback: read one KV block from disk. Remove the file on failure.
    """
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    o_direct = O_DIRECT if use_o_direct else 0
    try:
        fd = os.open(source_path, os.O_RDONLY | o_direct)
        bytes_read = os.readv(fd, [view_slice])
        if bytes_read < block_size:
            raise OSError(f"Short read: expected {block_size} bytes, read {bytes_read}")
    except Exception:
        try:
            os.remove(source_path)
        except OSError as cleanup_exc:
            logger.warning(
                "Failed to remove unreadable file %s: %s", source_path, cleanup_exc
            )
        raise
    finally:
        if fd is not None:
            os.close(fd)
