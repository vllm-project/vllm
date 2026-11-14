# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ownership of temporary payloads shared by filesystem-tier instances."""

import fcntl
import os
import re
from contextlib import suppress
from pathlib import Path
from stat import S_ISREG
from uuid import uuid4

from vllm.logger import init_logger

logger = init_logger(__name__)
_OWNER = re.compile(r"[0-9a-f]{32}\Z")
_TEMPORARY = re.compile(r"\.([0-9a-f]{64})\.ec\.([0-9a-f]{32})\.[0-9a-f]{16}\.tmp\Z")


class ECTemporaryFileOwner:
    """Hold an instance lease until its writes and cleanup have finished."""

    def __init__(self, namespace_path: Path) -> None:
        self._directory = namespace_path / ".owners"
        self._directory.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = namespace_path / ".tmp"
        self._locks: dict[str, int] = {}
        self.orphaned: dict[Path, int] = {}
        while True:
            self.owner_id = uuid4().hex
            fd = self._try_lock(self._directory / f"{self.owner_id}.lock", create=True)
            if fd is not None:
                self._locks[self.owner_id] = fd
                break
        try:
            with os.scandir(self._directory) as entries:
                owners = [
                    Path(entry.path)
                    for entry in entries
                    if entry.name.endswith(".lock")
                ]
            for path in owners:
                owner = path.stem
                if not _OWNER.fullmatch(owner) or owner == self.owner_id:
                    continue
                fd = self._try_lock(path)
                if fd is None:
                    continue
                self._locks[owner] = fd
                directory = self._temporary_directory / owner
                if directory.is_symlink():
                    raise OSError(
                        f"encoder-cache temporary directory is a symlink: {directory}"
                    )
                try:
                    entries = os.scandir(directory)
                except FileNotFoundError:
                    entries = None
                if entries is not None:
                    with entries:
                        for item in entries:
                            match = _TEMPORARY.fullmatch(item.name)
                            if match is None or match.group(2) != owner:
                                continue
                            path = Path(item.path)
                            try:
                                info = path.lstat()
                            except FileNotFoundError:
                                continue
                            if S_ISREG(info.st_mode):
                                path.unlink(missing_ok=True)
                self._release(owner, remove=True, strict=True)
        except BaseException:
            for owner in tuple(self._locks):
                self._release(owner, remove=False)
            raise

    @staticmethod
    def _try_lock(path: Path, *, create: bool = False) -> int | None:
        flags = os.O_RDWR | os.O_NOFOLLOW
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(path, flags, 0o600)
        except FileNotFoundError:
            if create:
                raise
            return None
        except FileExistsError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            info = os.fstat(fd)
            current = path.lstat()
            if S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == (
                current.st_dev,
                current.st_ino,
            ):
                return fd
        except (BlockingIOError, FileNotFoundError):
            pass
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)
        return None

    def _release(self, owner: str, *, remove: bool, strict: bool = False) -> None:
        fd = self._locks.pop(owner)
        try:
            if remove:
                with suppress(FileNotFoundError):
                    (self._temporary_directory / owner).rmdir()
                (self._directory / f"{owner}.lock").unlink(missing_ok=True)
        except OSError:
            if strict:
                raise
            logger.warning("Failed to remove encoder-cache owner files %s", owner)
        finally:
            os.close(fd)

    def close(self, remaining: dict[Path, int]) -> None:
        unfinished = {
            match.group(2)
            for path in remaining
            if (match := _TEMPORARY.fullmatch(path.name)) is not None
        }
        for owner in tuple(self._locks):
            self._release(owner, remove=owner not in unfinished)
