# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import logging
import mmap
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
    use_o_direct: bool = True,
) -> None:
    """Store callback: write to a temp file then atomically replace the target."""
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


def _load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    use_o_direct: bool = True,
) -> None:
    """Read one KV block from disk; remove the file only on a provable short
    read (a too-short file is genuine corruption) and leave it untouched on any
    other error.
    """
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    o_direct = O_DIRECT if use_o_direct else 0

    try:
        fd = os.open(source_path, os.O_RDONLY | o_direct)
        bytes_read = os.readv(fd, [view_slice])
        if bytes_read < block_size:
            # A failure to remove must not mask the short-read error below.
            try:
                os.remove(source_path)
            except OSError as cleanup_exc:
                logger.warning(
                    "Failed to remove short-read file %s: %s",
                    source_path,
                    cleanup_exc,
                )
            raise OSError(f"Short read: expected {block_size} bytes, read {bytes_read}")
    finally:
        if fd is not None:
            os.close(fd)


def batch_store_block(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    block_size: int,
    use_o_direct: bool = True,
) -> None:
    """Store a batch of KV blocks from a shared buffer to disk in one call.

    Each block buffer[offsets[i] : offsets[i]+block_size] is written atomically
    to dest_paths[i] via a temp-file rename.  Raises on first error.
    """
    _validate_offsets(view, offsets, block_size)

    if _HAS_FSIO_C:
        view_B = view.cast("B")
        view_slices = [view_B[x : x + block_size] for x in offsets]
        tmp_paths = [p + _get_tmp_suffix() for p in paths]
        return batch_store_block_C(tmp_paths, paths, view_slices, use_o_direct)
    else:
        for path, offset in zip(paths, offsets):
            _store_block(path, view, offset, block_size, use_o_direct)


def batch_load_block(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    block_size: int,
    use_o_direct: bool = True,
) -> None:
    """Load a batch of KV blocks from disk into a shared buffer in one call.

    Block i is read from source_paths[i] into view[offsets[i] : offsets[i]+block_size].
    Raises on first error (see _load_block for the delete-on-short-read policy).
    On failure the raised OSError carries ``num_succeeded`` = the number of
    blocks loaded before the failing one, so the tier can keep them.
    """
    _validate_offsets(view, offsets, block_size)

    if _HAS_FSIO_C:
        view_B = view.cast("B")
        view_slices = [view_B[x : x + block_size] for x in offsets]
        return batch_load_block_C(paths, view_slices, use_o_direct)
    else:
        for i, (path, offset) in enumerate(zip(paths, offsets)):
            try:
                _load_block(path, view, offset, block_size, use_o_direct)
            except OSError as exc:
                # Blocks 0..i-1 loaded fine; record the count for partial keep.
                # The C path sets the same attribute via PyObject_SetAttrString.
                exc.num_succeeded = i  # type: ignore[attr-defined]
                raise
