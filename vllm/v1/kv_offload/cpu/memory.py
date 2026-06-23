# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

DEFAULT_SHM_PATH = "/dev/shm"
HUGEPAGE_2MB = 2 * 1024 * 1024
HUGEPAGE_1GB = 1024 * 1024 * 1024
HUGEPAGE_BLOCK_SIZES = {
    "2M": HUGEPAGE_2MB,
    "2MB": HUGEPAGE_2MB,
    "1G": HUGEPAGE_1GB,
    "1GB": HUGEPAGE_1GB,
}


class CPUOffloadMemoryBackend(str, Enum):
    DEFAULT = "default"
    SHM = "shm"
    HUGETLBFS = "hugetlbfs"


@dataclass(frozen=True)
class MountInfo:
    mount_point: str
    fs_type: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class SharedMemoryAllocation:
    path: str
    fd: int
    creator: bool
    logical_size_bytes: int
    mapped_size_bytes: int


@dataclass(frozen=True)
class CPUOffloadMemoryConfig:
    backend: CPUOffloadMemoryBackend = CPUOffloadMemoryBackend.DEFAULT
    path: str | None = None
    hugepage_block_size: int = HUGEPAGE_2MB

    @classmethod
    def from_extra_config(
        cls, extra_config: Mapping[str, Any] | None
    ) -> "CPUOffloadMemoryConfig":
        extra_config = extra_config or {}
        backend = _parse_backend(extra_config.get("cpu_memory_backend", "default"))
        path = _parse_memory_path(extra_config.get("cpu_memory_path"))
        hugepage_block_size = _parse_hugepage_block_size(
            extra_config.get("cpu_hugepage_block_size", "2MB")
        )
        if backend == CPUOffloadMemoryBackend.HUGETLBFS and path is None:
            raise ValueError(
                "cpu_memory_path is required when "
                "cpu_memory_backend='hugetlbfs'"
            )
        return cls(
            backend=backend,
            path=path,
            hugepage_block_size=hugepage_block_size,
        )

    @property
    def effective_backend(self) -> CPUOffloadMemoryBackend:
        if self.backend == CPUOffloadMemoryBackend.DEFAULT:
            return CPUOffloadMemoryBackend.SHM
        return self.backend

    def mmap_path(self, instance_id: str) -> str:
        directory = self.path or DEFAULT_SHM_PATH
        return os.path.join(directory, f"vllm_offload_{instance_id}.mmap")

    def mapped_size(self, logical_size_bytes: int) -> int:
        if self.effective_backend == CPUOffloadMemoryBackend.HUGETLBFS:
            return round_up(logical_size_bytes, self.hugepage_block_size)
        return logical_size_bytes

    def validate(self) -> None:
        directory = self.path or DEFAULT_SHM_PATH
        _validate_directory(directory)
        if self.effective_backend == CPUOffloadMemoryBackend.HUGETLBFS:
            _validate_hugetlbfs_directory(directory, self.hugepage_block_size)


def _parse_backend(value: Any) -> CPUOffloadMemoryBackend:
    if isinstance(value, CPUOffloadMemoryBackend):
        return value
    try:
        return CPUOffloadMemoryBackend(str(value).lower())
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in CPUOffloadMemoryBackend)
        raise ValueError(
            f"Invalid cpu_memory_backend {value!r}; supported values are: "
            f"{supported}"
        ) from exc


def _parse_memory_path(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("cpu_memory_path must be a non-empty string")
    return value


def _parse_hugepage_block_size(value: Any) -> int:
    if isinstance(value, int):
        if value in (HUGEPAGE_2MB, HUGEPAGE_1GB):
            return value
    else:
        parsed = HUGEPAGE_BLOCK_SIZES.get(str(value).upper())
        if parsed is not None:
            return parsed
    raise ValueError("cpu_hugepage_block_size must be either '2MB' or '1GB'")


def _validate_directory(directory: str) -> None:
    if not os.path.isdir(directory):
        raise ValueError(f"cpu_memory_path {directory!r} must be an existing directory")
    if not os.access(directory, os.W_OK):
        raise ValueError(f"cpu_memory_path {directory!r} must be writable")


def _validate_hugetlbfs_directory(
    directory: str, expected_page_size: int
) -> None:
    real_path = os.path.realpath(directory)
    if _path_is_under(real_path, DEFAULT_SHM_PATH):
        raise ValueError(
            "cpu_memory_path for cpu_memory_backend='hugetlbfs' must not be "
            f"under {DEFAULT_SHM_PATH}"
        )

    mount_info = _get_mount_info(real_path)
    if mount_info is None:
        logger.warning(
            "Could not determine filesystem type for cpu_memory_path=%s; "
            "continuing with hugetlbfs allocation attempt.",
            directory,
        )
        return
    if mount_info.fs_type != "hugetlbfs":
        raise ValueError(
            "cpu_memory_path for cpu_memory_backend='hugetlbfs' must be backed "
            f"by hugetlbfs; found filesystem type {mount_info.fs_type!r} at "
            f"{mount_info.mount_point!r}"
        )
    mount_page_size = _get_hugetlbfs_page_size(mount_info)
    if mount_page_size is not None and mount_page_size != expected_page_size:
        raise ValueError(
            "cpu_hugepage_block_size does not match hugetlbfs mount page size: "
            f"expected {expected_page_size} bytes, mount uses "
            f"{mount_page_size} bytes"
        )


def _path_is_under(path: str, parent: str) -> bool:
    try:
        real_parent = os.path.realpath(parent)
        return os.path.commonpath([os.path.realpath(path), real_parent]) == real_parent
    except ValueError:
        return False


def _decode_mountinfo_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _get_mount_info(path: str) -> MountInfo | None:
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as f:
            mount_lines = f.readlines()
    except OSError:
        return None

    best_match: MountInfo | None = None
    best_match_len = -1
    real_path = os.path.realpath(path)
    for line in mount_lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator + 3 >= len(fields) or separator < 6:
            continue

        mount_point = os.path.realpath(_decode_mountinfo_path(fields[4]))
        if not _path_is_under(real_path, mount_point):
            continue
        if len(mount_point) <= best_match_len:
            continue

        options: list[str] = []
        options.extend(fields[5].split(","))
        options.extend(fields[separator + 3].split(","))
        best_match = MountInfo(
            mount_point=mount_point,
            fs_type=fields[separator + 1],
            options=tuple(options),
        )
        best_match_len = len(mount_point)
    return best_match


def _get_hugetlbfs_page_size(mount_info: MountInfo) -> int | None:
    for option in mount_info.options:
        key, separator, value = option.partition("=")
        if key == "pagesize" and separator:
            try:
                return _parse_hugepage_block_size(value)
            except ValueError:
                logger.warning(
                    "Could not parse hugetlbfs pagesize mount option %r",
                    value,
                )
                return None
    return None


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


def create_shared_memory_allocation(
    instance_id: str,
    logical_size_bytes: int,
    memory_config: CPUOffloadMemoryConfig,
) -> SharedMemoryAllocation:
    memory_config.validate()
    path = memory_config.mmap_path(instance_id)
    mapped_size_bytes = memory_config.mapped_size(logical_size_bytes)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    except FileExistsError:
        fd = os.open(path, os.O_RDWR)
        _wait_for_file_size(fd, mapped_size_bytes)
        logger.info("Opened existing mmap file %s", path)
        return SharedMemoryAllocation(
            path=path,
            fd=fd,
            creator=False,
            logical_size_bytes=logical_size_bytes,
            mapped_size_bytes=mapped_size_bytes,
        )
    except OSError as exc:
        raise OSError(
            f"Failed to open offload mmap file {path!r} for "
            f"cpu_memory_backend={memory_config.effective_backend.value!r}"
        ) from exc

    try:
        os.ftruncate(fd, mapped_size_bytes)
    except OSError as exc:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            logger.warning("Failed to unlink incomplete mmap file %s", path)
        raise OSError(
            f"Failed to size offload mmap file {path!r} to "
            f"{mapped_size_bytes} bytes for "
            f"cpu_memory_backend={memory_config.effective_backend.value!r}; "
            "verify cpu_memory_path and available hugepages"
        ) from exc

    logger.info(
        "Created mmap file %s (logical %.2f GB, mapped %.2f GB)",
        path,
        logical_size_bytes / 1e9,
        mapped_size_bytes / 1e9,
    )
    return SharedMemoryAllocation(
        path=path,
        fd=fd,
        creator=True,
        logical_size_bytes=logical_size_bytes,
        mapped_size_bytes=mapped_size_bytes,
    )
