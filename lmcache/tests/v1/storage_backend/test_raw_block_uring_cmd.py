# SPDX-License-Identifier: Apache-2.0

"""Tests for io_uring command (passthrough) support in Rust raw block backend."""

# Standard
from unittest.mock import MagicMock, patch
import asyncio
import mmap
import multiprocessing
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)
from lmcache.v1.storage_backend.raw_block.core import (
    RawBlockCore,
)

logger = init_logger(__name__)


# GLobal test device configuration with environment variables
TEST_DEVICES = {
    "block_device": os.environ.get("LMCACHE_TEST_BLOCK_DEVICE", "/dev/nvme0n1"),
    "char_device": os.environ.get("LMCACHE_TEST_CHAR_DEVICE", "/dev/ng0n1"),
    "null_device": os.environ.get("LMCACHE_TEST_NULL_DEVICE", "/dev/null"),
}


def _get_sysfs_path(device_path: str) -> str:
    """Derive sysfs path from device path."""

    device_name = os.path.basename(device_path)

    if device_name.startswith("ng"):
        parts = device_name[2:]
        device_name = f"nvme{parts}"

    return f"/sys/block/{device_name}"


def _has_ext() -> bool:
    """Check if the Rust raw block I/O extension is available."""
    try:
        # Third Party
        import lmcache_rust_raw_block_io  # noqa: F401

        return True
    except Exception:
        return False


def _run_uring_cmd_build_error_probe(device_path: str) -> None:
    """Verify that a middle SQE build failure completes without hanging."""
    # Third Party
    from lmcache_rust_raw_block_io import RawBlockDevice

    raw_device = RawBlockDevice(
        device_path,
        writable=False,
        use_odirect=False,
        alignment=4096,
        io_engine="io_uring",
        iouring_queue_depth=256,
        use_uring_cmd=True,
    )
    buffers: list[mmap.mmap] = []
    try:
        lba_size = raw_device.nvme_lba_size()
        buffers = [mmap.mmap(-1, lba_size) for _ in range(3)]
        batch_id = raw_device.batched_read(
            [0, 1, lba_size],
            buffers,
            [lba_size] * len(buffers),
        )

        try:
            raw_device.wait_iouring(batch_id)
        except ValueError as error:
            if "offset must be aligned to LBA size" not in str(error):
                raise
        else:
            raise AssertionError("expected the unaligned middle SQE build to fail")
    finally:
        raw_device.close()
        for buffer in buffers:
            buffer.close()


# Skip all tests in this file if the Rust extension is not available
pytestmark = pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not available"
)


@pytest.fixture
def loop_in_thread():
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


class MockConfig:
    """Mock configuration for testing."""

    def __init__(
        self,
        device_path: str,
        use_uring_cmd: bool = False,
        meta_total_bytes=4 * 1024 * 1024,
    ):
        self.extra_config = {
            "rust_raw_block.device_path": device_path,
            "rust_raw_block.use_odirect": False,
            "rust_raw_block.use_uring": True,
            "rust_raw_block.use_uring_cmd": use_uring_cmd,
            "rust_raw_block.capacity_bytes": 1024 * 1024 * 1024,  # 1GB
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": meta_total_bytes,
        }


class MockMetadata:
    """Mock metadata for testing."""

    def __init__(self, worker_id: int = 0, world_size: int = 1):
        self.worker_id = worker_id
        self.world_size = world_size


class MockLocalCPUBackend:
    """Mock local CPU backend for testing."""

    def __init__(self):
        pass

    def get_memory_allocator(self):
        return None

    def get_full_chunk_size_bytes(self) -> int:
        """return a default chunk size only for testing."""
        return 256 * 1024


def _build_rust_raw_block_metadata(
    worker_id: int = 0,
    world_size: int = 1,
) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=world_size,
        local_world_size=world_size,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


def _build_rust_raw_block_local_cpu_backend() -> MagicMock:
    local_cpu_backend = MagicMock()
    local_cpu_backend.get_full_chunk_size_bytes.return_value = 4096
    return local_cpu_backend


def _build_transfer_limit_backend(
    dev_path: str,
    max_data_transfer_size: int | None = None,
) -> RustRawBlockBackend:
    config = MockConfig(device_path=dev_path, use_uring_cmd=True)
    if max_data_transfer_size is not None:
        config.extra_config["rust_raw_block.max_data_transfer_size"] = (
            max_data_transfer_size
        )

    metadata = MockMetadata()
    loop = asyncio.new_event_loop()
    try:
        with (
            patch.object(RawBlockCore, "_rawdev", return_value=MagicMock()),
            patch.object(RawBlockCore, "_ensure_capacity_and_layout"),
            patch.object(RawBlockCore, "_load_checkpoint_from_device"),
        ):
            backend = RustRawBlockBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=MockLocalCPUBackend(),
                loop=loop,
                dst_device="cpu",
            )
            return backend
    finally:
        loop.close()


def test_uring_cmd_requires_character_device(loop_in_thread):
    """Test that io_uring_cmd requires a character device, not a block device."""
    # This test requires a block device device
    # Skip if this doesn't exist
    device_path = TEST_DEVICES["block_device"]

    if not os.path.exists(device_path):
        pytest.skip(f"Test device {device_path} not found.")

    config = MockConfig(device_path=device_path, use_uring_cmd=True)
    metadata = MockMetadata(worker_id=0, world_size=1)
    local_cpu_backend = MockLocalCPUBackend()

    # This should raise an error because the device is not a character device
    with pytest.raises(
        ValueError, match="use_uring_cmd requires an NVMe namespace character device"
    ):
        RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop_in_thread,
        )


def test_uring_cmd_get_nvme_info(loop_in_thread):
    """Test getting NVMe namespace ID and LBA size from character device."""
    # This test requires a nvme NS character device
    # Skip if this doesn't exist
    device_path = TEST_DEVICES["char_device"]

    if not os.path.exists(device_path):
        pytest.skip(f"Test device {device_path} not found.")

    config = MockConfig(device_path=device_path, use_uring_cmd=True)
    metadata = MockMetadata(worker_id=0, world_size=1)
    local_cpu_backend = MockLocalCPUBackend()

    try:
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop_in_thread,
        )

        # Get the raw device
        raw_device = backend._core.raw_device()

        # Test getting namespace ID
        nsid = raw_device.nvme_nsid()
        assert nsid > 0, f"Expected positive nsid, got {nsid}"
        logger.info(f"NVMe namespace ID: {nsid}")

        # Test getting LBA size
        lba_size = raw_device.nvme_lba_size()
        assert lba_size > 0, f"Expected positive lba_size, got {lba_size}"
        logger.info(f"NVMe LBA size: {lba_size} bytes")

    except Exception as e:
        pytest.fail(f"Failed to get NVMe info: {e}")


def test_uring_cmd_rejects_alignment_smaller_than_lba():
    """A 4Kn namespace must reject a 512-byte transfer alignment."""
    device_path = TEST_DEVICES["char_device"]
    if not os.path.exists(device_path):
        pytest.skip(f"Test device {device_path} not found.")

    lba_path = os.path.join(
        _get_sysfs_path(device_path),
        "queue",
        "logical_block_size",
    )
    try:
        with open(lba_path, encoding="utf-8") as file:
            lba_size = int(file.read().strip())
    except (OSError, ValueError):
        pytest.skip(f"Cannot determine the namespace LBA size from {lba_path}.")

    if lba_size <= 512:
        pytest.skip(
            "Test requires an NVMe namespace with an LBA larger than 512 bytes."
        )

    # Third Party
    from lmcache_rust_raw_block_io import RawBlockDevice

    expected_error = (
        rf"alignment \(512\) must be a non-zero multiple of "
        rf"NVMe LBA size \({lba_size}\)"
    )
    with pytest.raises(ValueError, match=expected_error):
        RawBlockDevice(
            device_path,
            writable=False,
            use_odirect=False,
            alignment=512,
            io_engine="io_uring",
            iouring_queue_depth=256,
            use_uring_cmd=True,
        )


def test_uring_cmd_middle_build_error_does_not_hang():
    """A rejected middle SQE must complete its batch with an error."""
    device_path = TEST_DEVICES["char_device"]
    if not os.path.exists(device_path):
        pytest.skip(f"Test device {device_path} not found.")

    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_uring_cmd_build_error_probe,
        args=(device_path,),
    )
    process.start()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("wait_iouring hung after a middle SQE build error")

    assert process.exitcode == 0


def test_uring_cmd_disabled(loop_in_thread):
    """Test that NVMe methods are not available when use_uring_cmd is disabled."""
    config = MockConfig(device_path=TEST_DEVICES["null_device"], use_uring_cmd=False)
    metadata = MockMetadata(worker_id=0, world_size=1)
    local_cpu_backend = MockLocalCPUBackend()
    raw_device = MagicMock()
    raw_device.nvme_nsid.side_effect = RuntimeError("use_uring_cmd not enabled")
    raw_device.nvme_lba_size.side_effect = RuntimeError("use_uring_cmd not enabled")

    with (
        patch.object(RawBlockCore, "_rawdev", return_value=MagicMock()),
        patch.object(RawBlockCore, "_ensure_capacity_and_layout"),
        patch.object(RawBlockCore, "_load_checkpoint_from_device"),
    ):
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop_in_thread,
        )
        backend._raw = raw_device

    # These should raise errors when use_uring_cmd is disabled
    with pytest.raises(RuntimeError, match="use_uring_cmd not enabled"):
        raw_device.nvme_nsid()

    with pytest.raises(RuntimeError, match="use_uring_cmd not enabled"):
        raw_device.nvme_lba_size()


def test_uring_cmd_auto_transfer_limit_from_sysfs_ng_device():
    expected_path = (
        f"{_get_sysfs_path(TEST_DEVICES['char_device'])}/queue/max_hw_sectors_kb"
    )
    with patch(
        "lmcache.v1.storage_backend.raw_block.core._read_sysfs_int",
        return_value=1024,
    ) as mock_read:
        backend = _build_transfer_limit_backend(
            TEST_DEVICES["char_device"], max_data_transfer_size=-1
        )

    mock_read.assert_called_once_with(expected_path)
    assert backend._core.max_data_transfer_size == 1024 * 1024


def test_uring_cmd_auto_transfer_limit_fails_when_sysfs_unavailable():
    expected_path = (
        f"{_get_sysfs_path(TEST_DEVICES['char_device'])}/queue/max_hw_sectors_kb"
    )
    with patch(
        "lmcache.v1.storage_backend.raw_block.core._read_sysfs_int",
        return_value=None,
    ) as mock_read:
        with pytest.raises(RuntimeError, match="failed to read max_hw_sectors_kb"):
            _build_transfer_limit_backend(
                TEST_DEVICES["char_device"], max_data_transfer_size=-1
            )

    mock_read.assert_called_once_with(expected_path)


def test_uring_cmd_explicit_transfer_limit_must_be_block_aligned():
    """Test that explicitly configured max_data_transfer_size must be block-aligned."""
    # Default block_align is 4096 bytes
    # 4096 is block-aligned (4096 % 4096 == 0)
    backend = _build_transfer_limit_backend(
        TEST_DEVICES["char_device"], max_data_transfer_size=4096
    )
    assert backend._core.max_data_transfer_size == 4096

    # 8192 is block-aligned (8192 % 4096 == 0)
    backend = _build_transfer_limit_backend(
        TEST_DEVICES["char_device"], max_data_transfer_size=8192
    )
    assert backend._core.max_data_transfer_size == 8192

    # 5000 is NOT block-aligned (5000 % 4096 != 0)
    with pytest.raises(
        ValueError,
        match=r"max_data_transfer_size \(5000\) must be a multiple of "
        r"block_align \(4096\)",
    ):
        _build_transfer_limit_backend(
            TEST_DEVICES["char_device"], max_data_transfer_size=5000
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
