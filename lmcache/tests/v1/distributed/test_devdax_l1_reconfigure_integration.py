# SPDX-License-Identifier: Apache-2.0
"""Opt-in integration test for runtime Device-DAX L1 reconfiguration
(add/remove lifecycle against live mmap-backed devices). Run with:

    RUN_DEVDAX_L1_INTEGRATION=1 pytest -xvs \
        tests/v1/distributed/test_devdax_l1_reconfigure_integration.py

Uses DRAM-backed tmpfs stand-ins by default (same open/fstat/mmap code path);
point at real devices (>=3) with LMCACHE_TEST_DEVDAX_L1_PATHS=/dev/dax0.0,...
Slot size defaults to 2 MiB (DAX mapping granularity); override with
LMCACHE_TEST_DEVDAX_L1_SLOT_BYTES.
"""

# Standard
from collections.abc import Iterator
import gc
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxRemoveMode,
)

RUN_IT = os.environ.get("RUN_DEVDAX_L1_INTEGRATION") == "1"
REAL_DEVICE_PATHS = [
    path.strip()
    for path in os.environ.get("LMCACHE_TEST_DEVDAX_L1_PATHS", "").split(",")
    if path.strip()
]
USING_REAL_DEVICES = bool(REAL_DEVICE_PATHS)
SLOT_BYTES = int(os.environ.get("LMCACHE_TEST_DEVDAX_L1_SLOT_BYTES", str(2 << 20)))

pytestmark = pytest.mark.skipif(
    not RUN_IT,
    reason="Device-DAX L1 integration test (set RUN_DEVDAX_L1_INTEGRATION=1)",
)


class _DeviceProvider:
    """Hands out device paths: real ones from LMCACHE_TEST_DEVDAX_L1_PATHS in
    order, else tmpfs files created on demand and removed on close."""

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace
        self._created: list[str] = []
        self._real_index = 0

    def acquire(self, size_in_bytes: int) -> str:
        if USING_REAL_DEVICES:
            if self._real_index >= len(REAL_DEVICE_PATHS):
                pytest.skip("not enough real Device-DAX devices provided")
            path = REAL_DEVICE_PATHS[self._real_index]
            self._real_index += 1
            fd = os.open(path, os.O_RDWR)
            try:
                capacity = os.fstat(fd).st_size
            finally:
                os.close(fd)
            if capacity and capacity < size_in_bytes:
                pytest.skip(
                    f"device {path} capacity {capacity} < required {size_in_bytes}"
                )
            return path

        path = os.path.join(self._workspace, f"devdax-arena-{len(self._created)}")
        with open(path, "wb") as handle:
            handle.truncate(size_in_bytes)
        self._created.append(path)
        return path

    def close(self) -> None:
        for path in self._created:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


@pytest.fixture
def devices(tmp_path) -> Iterator[_DeviceProvider]:
    """Device provider; tmpfs workspace defaults to /dev/shm when writable
    (override with LMCACHE_TEST_DEVDAX_L1_DIR)."""
    workspace = os.environ.get("LMCACHE_TEST_DEVDAX_L1_DIR", "")
    if not workspace:
        workspace = "/dev/shm" if os.access("/dev/shm", os.W_OK) else str(tmp_path)
    provider = _DeviceProvider(workspace)
    try:
        yield provider
    finally:
        provider.close()


def _layout(num_bytes: int = SLOT_BYTES) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([num_bytes])], dtypes=[torch.uint8])


def _key(seed: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=seed.to_bytes(4, "big") + b"\0" * 28,
        model_name="devdax-l1-reconfig-it",
        kv_rank=0,
    )


def _open_fd_count(path: str) -> int:
    """Return how many of this process's open fds point at ``path``."""
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}") == path:
                count += 1
        except OSError:
            continue
    return count


def test_runtime_add_and_drain_remove_lifecycle(devices):
    primary = devices.acquire(SLOT_BYTES)
    manager = DevDaxL1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=SLOT_BYTES,
            use_lazy=False,
            shm_name="",
            align_bytes=4096,
            devdax_path=primary,
        )
    )
    try:
        # The pool starts as a single primary arena that is actually mapped.
        statuses = manager.get_arena_statuses()
        assert [status.device_path for status in statuses] == [primary]
        assert statuses[0].is_primary is True
        assert _open_fd_count(primary) > 0

        # Fill the primary arena; the mapping is live shared memory.
        error, primary_objs = manager.allocate(_layout(), count=1)
        assert error == L1Error.SUCCESS
        primary_objs[0].raw_tensor.fill_(0xAB)
        assert int(primary_objs[0].raw_tensor[0]) == 0xAB

        # Primary is full, so allocation fails until we add capacity.
        error, empty = manager.allocate(_layout(), count=1)
        assert error == L1Error.OUT_OF_MEMORY
        assert empty == []

        # Add a device at runtime and confirm it is mapped and serves overflow.
        overflow = devices.acquire(2 * SLOT_BYTES)
        added = manager.add_device(overflow, 2 * SLOT_BYTES)
        assert added.state == DevDaxArenaState.ACTIVE
        assert added.is_primary is False
        assert _open_fd_count(overflow) > 0

        error, overflow_objs = manager.allocate(_layout(), count=2)
        assert error == L1Error.SUCCESS
        assert len(overflow_objs) == 2
        overflow_objs[0].raw_tensor.fill_(0xCD)
        assert int(overflow_objs[0].raw_tensor[0]) == 0xCD

        used, total = manager.get_memory_usage()
        assert total == 3 * SLOT_BYTES
        assert used == 3 * SLOT_BYTES

        # Drain-remove the overflow arena while its allocations are still live.
        removing = manager.remove_device(overflow, DevDaxRemoveMode.DRAIN)
        assert removing.state == DevDaxArenaState.DRAINING
        assert removing.active_allocations == 2

        # A draining arena is excluded from new allocations.
        error, blocked = manager.allocate(_layout(), count=1)
        assert error == L1Error.OUT_OF_MEMORY

        # Freeing the arena's last allocation unmaps it automatically.
        manager.free(overflow_objs)
        del overflow_objs
        assert [status.device_path for status in manager.get_arena_statuses()] == [
            primary
        ]
        assert _open_fd_count(overflow) == 0

        # An empty added arena is unmapped immediately on removal.
        third = devices.acquire(SLOT_BYTES)
        manager.add_device(third, SLOT_BYTES)
        removed = manager.remove_device(third, DevDaxRemoveMode.DRAIN)
        assert removed.state == DevDaxArenaState.REMOVED
        assert _open_fd_count(third) == 0

        # The primary arena backs get_l1_memory_desc and cannot be removed.
        with pytest.raises(ValueError, match="primary"):
            manager.remove_device(primary)

        manager.free(primary_objs)
        del primary_objs
    finally:
        manager.close()

    # Every device is unmapped once the manager is closed.
    assert _open_fd_count(primary) == 0

    # tmpfs devices are real files, so MAP_SHARED write-through is observable on
    # media; real Device-DAX char devices do not support read()/write() syscalls.
    if not USING_REAL_DEVICES:
        with open(primary, "rb") as handle:
            assert handle.read(SLOT_BYTES) == bytes([0xAB]) * SLOT_BYTES


def test_kv_cache_drain_gates_device_removal(devices):
    """Removal sentinel via the KV-cache path: a device requested for removal
    stays DRAINING (and readable) while KV entries live on it, and unmaps only
    after the last one is deleted."""
    primary = devices.acquire(SLOT_BYTES)
    l1 = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=SLOT_BYTES,
                use_lazy=False,
                shm_name="",
                align_bytes=4096,
                devdax_path=primary,
            )
        )
    )
    memory_manager = l1._memory_manager
    assert isinstance(memory_manager, DevDaxL1MemoryManager)
    try:
        # KV entry A fills the primary device.
        key_a, key_b, key_c = _key(1), _key(2), _key(3)
        write = l1.reserve_write([key_a], [False], _layout())
        assert write[key_a][0] == L1Error.SUCCESS
        write[key_a][1].tensor.fill_(0xA1)
        assert l1.finish_write([key_a])[key_a] == L1Error.SUCCESS
        del write

        # Add a device at runtime; KV entries B and C land on it as overflow.
        overflow = devices.acquire(2 * SLOT_BYTES)
        added = memory_manager.add_device(overflow, 2 * SLOT_BYTES)
        assert added.state == DevDaxArenaState.ACTIVE
        for key, fill in ((key_b, 0xB2), (key_c, 0xC3)):
            write = l1.reserve_write([key], [False], _layout())
            assert write[key][0] == L1Error.SUCCESS
            write[key][1].tensor.fill_(fill)
            assert l1.finish_write([key])[key] == L1Error.SUCCESS
            del write
        gc.collect()

        # Per-device usage: the primary holds A, the added device holds B and C.
        statuses = {s.device_path: s for s in memory_manager.get_arena_statuses()}
        assert statuses[primary].used_bytes == SLOT_BYTES
        assert statuses[primary].active_allocations == 1
        assert statuses[overflow].used_bytes == 2 * SLOT_BYTES
        assert statuses[overflow].active_allocations == 2
        assert statuses[overflow].free_bytes == 0

        # Request removal while B and C are still cached: the device drains.
        removing = memory_manager.remove_device(overflow, DevDaxRemoveMode.DRAIN)
        assert removing.state == DevDaxArenaState.DRAINING
        assert removing.active_allocations == 2
        assert _open_fd_count(overflow) > 0

        # KV cached on a draining device stays readable.
        read = l1.reserve_read([key_b])
        assert read[key_b][0] == L1Error.SUCCESS
        assert int(read[key_b][1].tensor[0]) == 0xB2
        assert l1.finish_read([key_b])[key_b] == L1Error.SUCCESS
        del read
        gc.collect()

        # Deleting one of the two entries keeps the device mapped and draining.
        assert l1.delete([key_b])[key_b] == L1Error.SUCCESS
        gc.collect()
        statuses = {s.device_path: s for s in memory_manager.get_arena_statuses()}
        assert statuses[overflow].state == DevDaxArenaState.DRAINING
        assert statuses[overflow].active_allocations == 1
        assert _open_fd_count(overflow) > 0

        # Deleting the last entry on the device unmaps it automatically.
        assert l1.delete([key_c])[key_c] == L1Error.SUCCESS
        gc.collect()
        assert [s.device_path for s in memory_manager.get_arena_statuses()] == [primary]
        assert _open_fd_count(overflow) == 0

        # KV on the remaining device is untouched by the removal.
        read = l1.reserve_read([key_a])
        assert read[key_a][0] == L1Error.SUCCESS
        assert int(read[key_a][1].tensor[0]) == 0xA1
        assert l1.finish_read([key_a])[key_a] == L1Error.SUCCESS
        del read
        assert l1.delete([key_a])[key_a] == L1Error.SUCCESS
        gc.collect()
    finally:
        l1.close()

    assert _open_fd_count(primary) == 0
