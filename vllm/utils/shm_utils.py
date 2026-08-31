# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fcntl
import glob
import os

from vllm.logger import init_logger

logger = init_logger(__name__)


def hold_region_file_lock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_SH)


def path_backed_by_fd(path: str, fd: int) -> bool:
    try:
        st = os.stat(path)
    except OSError:
        return False
    fst = os.fstat(fd)
    return (st.st_ino, st.st_dev) == (fst.st_ino, fst.st_dev)


def reap_orphaned_region_files(pattern: str) -> None:
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
            if os.fstat(fd).st_size == 0:
                continue
            if not path_backed_by_fd(path, fd):
                continue
            try:
                os.unlink(path)
            except OSError:
                logger.warning("Failed to reclaim %s", path, exc_info=True)
            else:
                logger.info("Reclaimed orphaned shared region file %s", path)
        finally:
            os.close(fd)
