# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import random
import threading

try:
    from vllm.fs_io_C import (  # pyright: ignore[reportMissingImports]
        batch_load_block as batch_load_block_C,
    )
    from vllm.fs_io_C import (
        batch_store_block as batch_store_block_C,
    )

    _HAS_FSIO_C = True
except ImportError:
    _HAS_FSIO_C = False

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


def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _validate_offsets(view: memoryview, offsets: list[int], block_size: int) -> None:
    """Raise if any block would read/write past the bounds of `view`.

    Without this, an out-of-range offset silently clips to a shorter (or
    empty) slice instead of failing, since memoryview slicing follows
    Python's slice-clamping semantics rather than raising.
    """
    total_len = len(view.cast("B"))
    for offset in offsets:
        if offset < 0 or offset + block_size > total_len:
            raise ValueError(
                f"block offset {offset} (block_size {block_size}) is out of "
                f"bounds for a buffer of size {total_len}"
            )


def _store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
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
    try:
        fd = os.open(
            tmp_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | O_DIRECT,
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


def _load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """
    Load callback: read one KV block from disk. Remove the file on failure.
    """
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]

    try:
        fd = os.open(source_path, os.O_RDONLY | O_DIRECT)
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


def batch_store_block(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    block_size: int,
) -> None:
    """
    Store a batch of KV blocks from a shared buffer to disk in one call.

    Each block buffer[offsets[i] : offsets[i]+block_size] is written atomically
    to dest_paths[i] via a temp-file rename.  Raises on first error.
    """
    _validate_offsets(view, offsets, block_size)

    if _HAS_FSIO_C:
        view_B = view.cast("B")
        view_slices = [view_B[x : x + block_size] for x in offsets]
        tmp_paths = [p + _get_tmp_suffix() for p in paths]
        return batch_store_block_C(tmp_paths, paths, view_slices)
    else:
        for path, offset in zip(paths, offsets):
            _store_block(path, view, offset, block_size)


def batch_load_block(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    block_size: int,
) -> None:
    """
    Load a batch of KV blocks from disk into a shared buffer in one call.

    Block i is read from source_paths[i] into view[offsets[i] : offsets[i]+block_size].
    Raises on first error and removes the offending file.
    """
    _validate_offsets(view, offsets, block_size)

    if _HAS_FSIO_C:
        view_B = view.cast("B")
        view_slices = [view_B[x : x + block_size] for x in offsets]
        return batch_load_block_C(paths, view_slices)
    else:
        for path, offset in zip(paths, offsets):
            _load_block(path, view, offset, block_size)
