# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fcntl
import glob
import os
import time

from vllm.logger import init_logger

logger = init_logger(__name__)


def wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while os.fstat(fd).st_size < expected_size:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


def _path_backed_by_fd(path: str, fd: int) -> bool:
    try:
        st = os.stat(path)
    except OSError:
        return False
    fst = os.fstat(fd)
    return (st.st_ino, st.st_dev) == (fst.st_ino, fst.st_dev)


def open_region_file(path: str, size: int) -> tuple[int, bool]:
    """Create or join a shared region file of `size` bytes.

    Returns (fd, is_creator). The fd holds a shared flock for its lifetime,
    which is what reap_orphaned_region_files keys liveness on.
    """
    from vllm.distributed.device_communicators.shm_broadcast import (
        check_shm_free_space,
    )

    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            try:
                fd = os.open(path, os.O_RDWR)
            except FileNotFoundError:
                continue
            fcntl.flock(fd, fcntl.LOCK_SH)
            if not _path_backed_by_fd(path, fd):
                os.close(fd)
                continue
            try:
                wait_for_file_size(fd, size)
            except (TimeoutError, OSError):
                os.close(fd)
                raise
            logger.info("Opened existing shared region file %s", path)
            return fd, False
        fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            check_shm_free_space(size)
            os.ftruncate(fd, size)
        except (RuntimeError, OSError):
            os.unlink(path)
            os.close(fd)
            raise
        logger.info("Created shared region file %s (%.2f MiB)", path, size / (1 << 20))
        return fd, True


def reap_orphaned_region_files(pattern: str) -> None:
    """Unlink files matching `pattern` that no live process holds a flock on."""
    for path in glob.glob(pattern):
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue
            if os.fstat(fd).st_size == 0 or not _path_backed_by_fd(path, fd):
                continue
            try:
                os.unlink(path)
            except OSError:
                logger.warning("Failed to reclaim %s", path, exc_info=True)
            else:
                logger.info("Reclaimed orphaned shared region file %s", path)
        finally:
            os.close(fd)
