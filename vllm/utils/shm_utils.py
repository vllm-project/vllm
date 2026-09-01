# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared /dev/shm region files: creation, joining and orphan reclamation.

Every process that maps a region holds a shared flock on its fd for the life
of the mapping, and the kernel releases that lock however the process exits,
SIGKILL included. A file whose exclusive lock is free is therefore mapped by
nobody and can be reclaimed. Creators publish atomically from a temporary
name, so a file only ever appears at its final path once it is fully sized
and already locked: joiners never observe a partial region, and the sweep
can never mistake a region that is starting up for an orphan.
"""

import contextlib
import fcntl
import glob
import os
import tempfile

from vllm.logger import init_logger

logger = init_logger(__name__)


def _path_backed_by_fd(path: str, fd: int) -> bool:
    try:
        st = os.stat(path)
    except OSError:
        return False
    fst = os.fstat(fd)
    return (st.st_ino, st.st_dev) == (fst.st_ino, fst.st_dev)


def _create_region_file(path: str, size: int) -> int | None:
    """Build a sized, locked region file and publish it atomically at `path`.

    Returns its fd, or None if another process published first and this one
    should join instead.
    """
    from vllm.distributed.device_communicators.shm_broadcast import (
        check_shm_free_space,
    )

    directory, name = os.path.split(path)
    # mkstemp keeps the name unique even between threads of one process, and
    # the .tmp suffix keeps it inside the sweep's glob so a creator killed
    # before publishing does not leak it.
    fd, tmp_path = tempfile.mkstemp(prefix=f"{name}.", suffix=".tmp", dir=directory)
    published = False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        check_shm_free_space(size)
        os.ftruncate(fd, size)
        try:
            # link() is atomic and fails if the name is taken, so exactly one
            # racing creator wins and the losers join its region instead of
            # silently mapping a second one.
            os.link(tmp_path, path)
        except FileExistsError:
            return None
        except FileNotFoundError:
            # A sweep reclaimed our still-unpublished temp file; retry.
            return None
        published = True
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        if not published:
            os.close(fd)
    return fd


def open_region_file(path: str, size: int) -> tuple[int, bool]:
    """Create or join the shared region file at `path`, sized `size` bytes.

    Returns (fd, is_creator). The fd holds a shared flock for its lifetime,
    which is what reap_orphaned_region_files keys liveness on; the caller
    owns it and must close it.
    """
    while True:
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            fd_or_none = _create_region_file(path, size)
            if fd_or_none is None:
                continue
            logger.info(
                "Created shared region file %s (%.2f MiB)", path, size / (1 << 20)
            )
            return fd_or_none, True

        joined = False
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            # The name may have been reclaimed between the open and the lock.
            joined = _path_backed_by_fd(path, fd)
            if joined:
                actual_size = os.fstat(fd).st_size
                if actual_size < size:
                    raise RuntimeError(
                        f"Shared region file {path} is {actual_size} bytes, "
                        f"expected at least {size}; a process is using a "
                        f"different offloading configuration."
                    )
        finally:
            if not joined:
                os.close(fd)
        if joined:
            logger.info("Opened existing shared region file %s", path)
            return fd, False


def reap_orphaned_region_files(pattern: str) -> None:
    """Unlink region files matching `pattern` that no live process maps.

    The unlink is not atomic with the liveness check, so the inode re-check
    narrows, but cannot fully close, the window in which another process
    republishes the same name in between.
    """
    for path in glob.glob(pattern):
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue  # a live process still maps this region
            if not _path_backed_by_fd(path, fd):
                continue
            try:
                os.unlink(path)
            except OSError:
                logger.warning("Failed to reclaim %s", path, exc_info=True)
            else:
                logger.info("Reclaimed orphaned shared region file %s", path)
        finally:
            os.close(fd)
