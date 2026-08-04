# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for L1Manager.

These tests verify the behavior of L1Manager as described in the
interface docstrings. The tests focus on:

1. reserve_read() - Reserve read access for given keys
   - Returns KEY_NOT_EXIST if key does not exist
   - Returns KEY_NOT_READABLE if key exists but is write-locked
   - Returns SUCCESS and MemoryObj if key is readable

2. unsafe_read() - Unsafe read without acquiring new read locks
   - Returns KEY_NOT_EXIST if key does not exist
   - Returns KEY_NOT_READABLE if key is not read-locked
   - Returns SUCCESS and MemoryObj if key is read-locked

3. finish_read() - Finish read access for given keys
   - Returns KEY_NOT_EXIST if key does not exist
   - Returns KEY_IN_WRONG_STATE if key is write-locked or non-read-locked
   - Returns SUCCESS on successful unlock
   - Deletes temporary objects when read count reaches zero

4. reserve_write() - Reserve write access for given keys
   - Returns KEY_NOT_WRITABLE if key exists but cannot be written
   - Returns OUT_OF_MEMORY if allocation fails
   - Returns SUCCESS and MemoryObj on success

5. finish_write() - Finish write access for given keys
   - Returns KEY_NOT_EXIST if key does not exist
   - Returns KEY_IN_WRONG_STATE if not write-locked or read-locked
   - Returns SUCCESS on successful unlock

6. delete() - Delete keys from L1 cache
   - Returns KEY_NOT_EXIST if key does not exist
   - Returns KEY_IS_LOCKED if key is locked
   - Returns SUCCESS on successful deletion

7. get_object_state() - Debugging API to get internal state

8. close() - Close the L1Manager and free all resources
"""

# Standard
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from tests.v1.distributed.utils import should_use_lazy_alloc

try:
    # First Party
    from lmcache.v1.distributed.l1_manager import L1Manager
except ImportError:
    # Skip tests if L1Manager cannot be imported
    pytest.skip(
        "Skipping because L1 manager cannot be imported", allow_module_level=True
    )

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_memory_config():
    """Create a basic L1MemoryManagerConfig for testing."""
    return L1MemoryManagerConfig(
        size_in_bytes=128 * 1024 * 1024,  # 128MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,  # 64MB
        align_bytes=0x1000,  # 4KB
    )


@pytest.fixture
def small_memory_config():
    """Create a small L1MemoryManagerConfig to test memory exhaustion."""
    return L1MemoryManagerConfig(
        size_in_bytes=64 * 1024 * 1024,  # 64MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,  # 64MB
        align_bytes=0x1000,
    )


@pytest.fixture
def basic_l1_config(basic_memory_config):
    """Create a basic L1ManagerConfig for testing."""
    return L1ManagerConfig(
        memory_config=basic_memory_config,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


@pytest.fixture
def small_l1_config(small_memory_config):
    """Create a small L1ManagerConfig to test memory exhaustion."""
    return L1ManagerConfig(
        memory_config=small_memory_config,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


@pytest.fixture
def basic_layout():
    """Create a basic MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


@pytest.fixture
def large_layout():
    """Create a large MemoryLayoutDesc that will exhaust small memory.

    Each allocation is 8MB (2M elements * 4 bytes).
    """
    return MemoryLayoutDesc(
        shapes=[torch.Size([2048, 1024])],  # 2M elements * 4 bytes = 8MB
        dtypes=[torch.float32],
    )


def make_object_key(chunk_hash: int, model_name: str = "test_model", kv_rank: int = 0):
    """Helper to create ObjectKey instances."""
    hash_bytes = ObjectKey.IntHash2Bytes(chunk_hash)
    return ObjectKey(chunk_hash=hash_bytes, model_name=model_name, kv_rank=kv_rank)


# =============================================================================
# Tests for L1Manager.reserve_read()
# =============================================================================


class TestReserveRead:
    """
    Tests for L1Manager.reserve_read() method.

    Per the docstring:
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_NOT_READABLE: The key exists but is not readable.
    - Returns (L1Error, Optional[MemoryObj]) for each key.
    """

    def test_reserve_read_non_existing_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_read returns KEY_NOT_EXIST for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.reserve_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_NOT_EXIST
        assert mem_obj is None

        manager.close()

    def test_reserve_read_write_locked_key_returns_key_not_readable(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_read returns KEY_NOT_READABLE for write-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Reserve write (but don't finish) - key is now write-locked
        write_result = manager.reserve_write([key], [False], basic_layout)
        assert write_result[key][0] == L1Error.SUCCESS

        # Try to reserve read on a write-locked key
        read_result = manager.reserve_read([key])

        assert key in read_result
        error, mem_obj = read_result[key]
        assert error == L1Error.KEY_NOT_READABLE
        assert mem_obj is None

        manager.close()

    def test_reserve_read_ready_key_returns_success(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_read returns SUCCESS for ready (unlocked) keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create object: reserve write -> finish write
        write_result = manager.reserve_write([key], [False], basic_layout)
        assert write_result[key][0] == L1Error.SUCCESS
        finish_result = manager.finish_write([key])
        assert finish_result[key] == L1Error.SUCCESS

        # Now reserve read
        read_result = manager.reserve_read([key])

        assert key in read_result
        error, mem_obj = read_result[key]
        assert error == L1Error.SUCCESS
        assert mem_obj is not None
        assert mem_obj.is_valid()

        manager.close()

    def test_reserve_read_multiple_keys(self, basic_l1_config, basic_layout):
        """Test reserve_read with multiple keys in a single call."""
        manager = L1Manager(basic_l1_config)
        key1 = make_object_key(1)
        key2 = make_object_key(2)
        key3 = make_object_key(3)

        # Create key1 as ready object
        manager.reserve_write([key1], [False], basic_layout)
        manager.finish_write([key1])

        # key2 does not exist
        # key3 is write-locked
        manager.reserve_write([key3], [False], basic_layout)

        # Reserve read on all three
        result = manager.reserve_read([key1, key2, key3])

        assert result[key1][0] == L1Error.SUCCESS
        assert result[key1][1] is not None
        assert result[key2][0] == L1Error.KEY_NOT_EXIST
        assert result[key2][1] is None
        assert result[key3][0] == L1Error.KEY_NOT_READABLE
        assert result[key3][1] is None

        manager.close()

    def test_reserve_read_can_be_called_multiple_times(
        self, basic_l1_config, basic_layout
    ):
        """Test that multiple read reservations can be made on the same key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Multiple read reservations
        result1 = manager.reserve_read([key])
        result2 = manager.reserve_read([key])
        result3 = manager.reserve_read([key])

        assert result1[key][0] == L1Error.SUCCESS
        assert result2[key][0] == L1Error.SUCCESS
        assert result3[key][0] == L1Error.SUCCESS

        # Verify using get_object_state that read lock is held
        state = manager.get_object_state(key)
        assert state is not None
        # Check via available_for_read (should still be true since
        # read-locked is readable)
        assert state.available_for_read() is True

        manager.close()

    def test_reserve_read_with_extra_count(self, basic_l1_config, basic_layout):
        """Test reserve_read(extra_count=N) acquires 1+N locks."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve with extra_count=2 -> total 3 locks
        result = manager.reserve_read([key], extra_count=2)
        assert result[key][0] == L1Error.SUCCESS
        assert result[key][1] is not None

        # Need 3 finish_read() to fully release
        manager.finish_read([key])
        state = manager.get_object_state(key)
        assert state is not None
        assert state.read_lock.is_locked()

        manager.finish_read([key])
        state = manager.get_object_state(key)
        assert state is not None
        assert state.read_lock.is_locked()

        manager.finish_read([key])
        state = manager.get_object_state(key)
        assert state is not None
        assert not state.read_lock.is_locked()

        manager.close()

    def test_reserve_read_extra_count_default_is_zero(
        self, basic_l1_config, basic_layout
    ):
        """Default extra_count=0 acquires exactly 1 lock."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        manager.reserve_read([key])
        manager.finish_read([key])

        # Lock fully released after single finish_read
        state = manager.get_object_state(key)
        assert state is not None
        assert not state.read_lock.is_locked()

        manager.close()


# =============================================================================
# Tests for L1Manager.unsafe_read()
# =============================================================================


class TestUnsafeRead:
    """
    Tests for L1Manager.unsafe_read() method.

    Per the docstring:
    - This method does not acquire read locks.
    - Caller must ensure unsafe_read is called between reserve_read and finish_read.
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_NOT_READABLE: The key is not readable (not read-locked).
    - Returns (L1Error, Optional[MemoryObj]) for each key.
    """

    def test_unsafe_read_non_existing_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that unsafe_read returns KEY_NOT_EXIST for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.unsafe_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_NOT_EXIST
        assert mem_obj is None

        manager.close()

    def test_unsafe_read_non_read_locked_returns_key_not_readable(
        self, basic_l1_config, basic_layout
    ):
        """Test that unsafe_read returns KEY_NOT_READABLE if not read-locked."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object (not read-locked)
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Try unsafe_read without reserve_read
        result = manager.unsafe_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_NOT_READABLE
        assert mem_obj is None

        manager.close()

    def test_unsafe_read_write_locked_returns_key_not_readable(
        self, basic_l1_config, basic_layout
    ):
        """Test that unsafe_read returns KEY_NOT_READABLE for write-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create write-locked object
        manager.reserve_write([key], [False], basic_layout)

        # Try unsafe_read on write-locked key
        result = manager.unsafe_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_NOT_READABLE
        assert mem_obj is None

        manager.close()

    def test_unsafe_read_read_locked_returns_success(
        self, basic_l1_config, basic_layout
    ):
        """Test that unsafe_read returns SUCCESS for read-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object and reserve read
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        reserve_result = manager.reserve_read([key])
        assert reserve_result[key][0] == L1Error.SUCCESS

        # unsafe_read should succeed on read-locked key
        result = manager.unsafe_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.SUCCESS
        assert mem_obj is not None
        assert mem_obj.is_valid()

        manager.close()

    def test_unsafe_read_returns_same_memory_obj_as_reserve_read(
        self, basic_l1_config, basic_layout
    ):
        """Test that unsafe_read returns the same MemoryObj as reserve_read."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read and get memory object
        reserve_result = manager.reserve_read([key])
        assert reserve_result[key][0] == L1Error.SUCCESS
        reserved_mem_obj = reserve_result[key][1]

        # unsafe_read should return the same memory object
        unsafe_result = manager.unsafe_read([key])
        assert unsafe_result[key][0] == L1Error.SUCCESS
        unsafe_mem_obj = unsafe_result[key][1]

        # Should be the same object
        assert reserved_mem_obj is unsafe_mem_obj

        manager.close()

    def test_unsafe_read_multiple_times_without_adding_read_count(
        self, basic_l1_config, basic_layout
    ):
        """Test that multiple unsafe_reads don't add to read lock count."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # Reserve read once
        manager.reserve_read([key])

        # Multiple unsafe_reads
        for _ in range(5):
            result = manager.unsafe_read([key])
            assert result[key][0] == L1Error.SUCCESS

        # Single finish_read should release the lock and delete temp object
        manager.finish_read([key])

        # Object should be deleted (only 1 read lock was held, not 6)
        assert manager.get_object_state(key) is None

        manager.close()

    def test_unsafe_read_multiple_keys(self, basic_l1_config, basic_layout):
        """Test unsafe_read with multiple keys in a single call."""
        manager = L1Manager(basic_l1_config)
        key1 = make_object_key(1)
        key2 = make_object_key(2)
        key3 = make_object_key(3)

        # key1: read-locked
        manager.reserve_write([key1], [False], basic_layout)
        manager.finish_write([key1])
        manager.reserve_read([key1])

        # key2: does not exist

        # key3: ready but not read-locked
        manager.reserve_write([key3], [False], basic_layout)
        manager.finish_write([key3])

        # unsafe_read on all three
        result = manager.unsafe_read([key1, key2, key3])

        assert result[key1][0] == L1Error.SUCCESS
        assert result[key1][1] is not None
        assert result[key2][0] == L1Error.KEY_NOT_EXIST
        assert result[key2][1] is None
        assert result[key3][0] == L1Error.KEY_NOT_READABLE
        assert result[key3][1] is None

        manager.close()

    def test_unsafe_read_between_reserve_and_finish(
        self, basic_l1_config, basic_layout
    ):
        """Test proper usage: unsafe_read between reserve_read and finish_read."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Proper workflow: reserve_read -> unsafe_read -> finish_read
        reserve_result = manager.reserve_read([key])
        assert reserve_result[key][0] == L1Error.SUCCESS

        unsafe_result = manager.unsafe_read([key])
        assert unsafe_result[key][0] == L1Error.SUCCESS
        assert unsafe_result[key][1] is not None

        finish_result = manager.finish_read([key])
        assert finish_result[key] == L1Error.SUCCESS

        # After finish_read, unsafe_read should fail (not read-locked)
        result = manager.unsafe_read([key])
        assert result[key][0] == L1Error.KEY_NOT_READABLE

        manager.close()


# =============================================================================
# Tests for L1Manager.finish_read()
# =============================================================================


class TestFinishRead:
    """
    Tests for L1Manager.finish_read() method.

    Per the docstring:
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_IN_WRONG_STATE: The key is write-locked or non-read-locked.
    - Will delete the object if it is temporary and read count reaches zero.
    """

    def test_finish_read_non_existing_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_read returns KEY_NOT_EXIST for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.finish_read([key])

        assert key in result
        assert result[key] == L1Error.KEY_NOT_EXIST

        manager.close()

    def test_finish_read_success(self, basic_l1_config, basic_layout):
        """Test that finish_read returns SUCCESS after proper read reservation."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read, then finish read
        manager.reserve_read([key])
        result = manager.finish_read([key])

        assert result[key] == L1Error.SUCCESS

        manager.close()

    def test_finish_read_non_read_locked_returns_wrong_state(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_read returns KEY_IN_WRONG_STATE if not read-locked."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object (not read-locked)
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Try to finish read without reserving
        result = manager.finish_read([key])

        assert result[key] == L1Error.KEY_IN_WRONG_STATE

        manager.close()

    def test_finish_read_write_locked_returns_wrong_state(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_read returns KEY_IN_WRONG_STATE if write-locked."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create write-locked object
        manager.reserve_write([key], [False], basic_layout)

        # Try to finish read on write-locked key
        result = manager.finish_read([key])

        assert result[key] == L1Error.KEY_IN_WRONG_STATE

        manager.close()

    def test_finish_read_temporary_object_deleted_when_count_zero(
        self, basic_l1_config, basic_layout
    ):
        """Test that temporary objects are deleted when read count reaches zero."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)  # is_temporary=True
        manager.finish_write([key])

        # Reserve read
        manager.reserve_read([key])

        # Verify object exists
        assert manager.get_object_state(key) is not None

        # Finish read - should delete the temporary object
        result = manager.finish_read([key])
        assert result[key] == L1Error.SUCCESS

        # Verify object is deleted
        assert manager.get_object_state(key) is None

        manager.close()

    def test_finish_read_multiple_reads_count_down(self, basic_l1_config, basic_layout):
        """Test that multiple finish_reads count down properly."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # Reserve read three times
        manager.reserve_read([key])
        manager.reserve_read([key])
        manager.reserve_read([key])

        # Finish read twice - object should still exist
        manager.finish_read([key])
        manager.finish_read([key])
        assert manager.get_object_state(key) is not None

        # Finish read third time - temporary object should be deleted
        manager.finish_read([key])
        assert manager.get_object_state(key) is None

        manager.close()

    def test_finish_read_with_extra_count_releases_multiple(
        self, basic_l1_config, basic_layout
    ):
        """finish_read(extra_count=2) releases 3 locks at once."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Acquire 3 read locks (1 + extra_count=2)
        manager.reserve_read([key], extra_count=2)

        # Release all 3 at once
        result = manager.finish_read([key], extra_count=2)
        assert result[key] == L1Error.SUCCESS

        state = manager.get_object_state(key)
        assert state is not None
        assert not state.read_lock.is_locked()

        manager.close()

    def test_finish_read_extra_count_partial_release(
        self, basic_l1_config, basic_layout
    ):
        """Partial extra_count release leaves remaining locks."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # 3 locks total (1 + extra_count=2)
        manager.reserve_read([key], extra_count=2)

        # Release 2 of 3 (1 + extra_count=1)
        manager.finish_read([key], extra_count=1)
        state = manager.get_object_state(key)
        assert state is not None
        assert state.read_lock.is_locked()

        # Release last one (1 + extra_count=0)
        manager.finish_read([key])
        state = manager.get_object_state(key)
        assert state is not None
        assert not state.read_lock.is_locked()

        manager.close()

    def test_finish_read_extra_count_deletes_temporary(
        self, basic_l1_config, basic_layout
    ):
        """Temp objects deleted when extra_count releases all."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # Acquire 3 read locks at once (1 + extra_count=2)
        manager.reserve_read([key], extra_count=2)
        assert manager.get_object_state(key) is not None

        # Release all 3 at once -> temp deleted
        result = manager.finish_read([key], extra_count=2)
        assert result[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is None

        manager.close()

    def test_finish_read_extra_count_temp_survives_partial(
        self, basic_l1_config, basic_layout
    ):
        """Temp object survives partial extra_count release."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # 4 locks total (1 + extra_count=3)
        manager.reserve_read([key], extra_count=3)

        # Release 2 of 4 -> still locked
        manager.finish_read([key], extra_count=1)
        assert manager.get_object_state(key) is not None

        # Release remaining 2 -> deleted
        manager.finish_read([key], extra_count=1)
        assert manager.get_object_state(key) is None

        manager.close()


# =============================================================================
# Tests for L1Manager.reserve_write()
# =============================================================================


class TestReserveWrite:
    """
    Tests for L1Manager.reserve_write() method.

    Per the docstring:
    - KEY_NOT_WRITABLE: The key exists but is not writable.
    - OUT_OF_MEMORY: Not enough memory to allocate for the object.
    - Returns (L1Error, Optional[MemoryObj]) for each key.
    """

    def test_reserve_write_new_key_returns_success(self, basic_l1_config, basic_layout):
        """Test that reserve_write returns SUCCESS for new keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.reserve_write([key], [False], basic_layout)

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.SUCCESS
        assert mem_obj is not None
        assert mem_obj.is_valid()

        manager.close()

    def test_reserve_write_write_locked_key_returns_not_writable(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_write returns KEY_NOT_WRITABLE for write-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # First reserve write
        result1 = manager.reserve_write([key], [False], basic_layout)
        assert result1[key][0] == L1Error.SUCCESS

        # Try to reserve write again while still write-locked
        result2 = manager.reserve_write([key], [False], basic_layout)

        assert result2[key][0] == L1Error.KEY_NOT_WRITABLE
        assert result2[key][1] is None

        manager.close()

    def test_reserve_write_read_locked_key_returns_not_writable(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_write returns KEY_NOT_WRITABLE for read-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read
        manager.reserve_read([key])

        # Try to reserve write while read-locked
        result = manager.reserve_write([key], [False], basic_layout)

        assert result[key][0] == L1Error.KEY_NOT_WRITABLE
        assert result[key][1] is None

        manager.close()

    def test_reserve_write_temporary_key_returns_not_writable(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_write returns KEY_NOT_WRITABLE for temporary objects."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)  # is_temporary=True
        manager.finish_write([key])

        # Try to reserve write on temporary object
        result = manager.reserve_write([key], [False], basic_layout)

        assert result[key][0] == L1Error.KEY_NOT_WRITABLE

        manager.close()

    def test_reserve_write_multiple_keys(self, basic_l1_config, basic_layout):
        """Test reserve_write with multiple keys in a single call."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(5)]
        is_temporary = [False] * 5

        result = manager.reserve_write(keys, is_temporary, basic_layout)

        for key in keys:
            assert key in result
            error, mem_obj = result[key]
            assert error == L1Error.SUCCESS
            assert mem_obj is not None

        manager.close()

    def test_reserve_write_out_of_memory(self, small_l1_config, large_layout):
        """Test that reserve_write returns OUT_OF_MEMORY when allocation fails."""
        manager = L1Manager(small_l1_config)
        keys = [make_object_key(i) for i in range(10)]
        is_temporary = [False] * 10

        # Request more memory than available (8MB * 10 = 80MB > 64MB)
        result = manager.reserve_write(keys, is_temporary, large_layout)

        # All keys should return OUT_OF_MEMORY
        for key in keys:
            assert result[key][0] == L1Error.OUT_OF_MEMORY
            assert result[key][1] is None

        manager.close()

    def test_reserve_write_new_mode(self, basic_l1_config, basic_layout):
        """Test that reserve_write returns KEY_NOT_WRITABLE for existing keys."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(5)]
        is_temporary = [False] * 5

        result = manager.reserve_write(keys, is_temporary, basic_layout, mode="new")

        for key in keys:
            assert result[key][0] == L1Error.SUCCESS
            assert result[key][1] is not None

        # Commit the write
        result = manager.finish_write(keys)
        for key in keys:
            assert result[key] == L1Error.SUCCESS

        # Now try to reserve write again with mode="new"
        result = manager.reserve_write(keys, is_temporary, basic_layout, mode="new")
        for key in keys:
            assert result[key][0] == L1Error.KEY_NOT_WRITABLE
            assert result[key][1] is None

        manager.close()

    def test_reserve_write_update_mode(self, basic_l1_config, basic_layout):
        """Test that reserve_write returns KEY_NOT_WRITABLE for new keys."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(5)]
        is_temporary = [False] * 5

        result = manager.reserve_write(keys, is_temporary, basic_layout, mode="update")
        for key in keys:
            assert result[key][0] == L1Error.KEY_NOT_WRITABLE
            assert result[key][1] is None

        # Cannot finish write in update mode because keys not exist
        result = manager.finish_write(keys)
        for key in keys:
            assert result[key] == L1Error.KEY_NOT_EXIST

        # Now try to reserve write again with mode="new"
        result = manager.reserve_write(keys, is_temporary, basic_layout, mode="new")
        for key in keys:
            assert result[key][0] == L1Error.SUCCESS
            assert result[key][1] is not None

        # Commit the write
        result = manager.finish_write(keys)
        for key in keys:
            assert result[key] == L1Error.SUCCESS

        # Now try to reserve write again with mode="update"
        result = manager.reserve_write(keys, is_temporary, basic_layout, mode="update")
        for key in keys:
            assert result[key][0] == L1Error.SUCCESS
            assert result[key][1] is not None

        manager.close()


# =============================================================================
# Tests for L1Manager.finish_write()
# =============================================================================


class TestFinishWrite:
    """
    Tests for L1Manager.finish_write() method.

    Per the docstring:
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_IN_WRONG_STATE: The key is not write-locked, or it's read-locked.
    """

    def test_finish_write_non_existing_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_write returns KEY_NOT_EXIST for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.finish_write([key])

        assert key in result
        assert result[key] == L1Error.KEY_NOT_EXIST

        manager.close()

    def test_finish_write_success(self, basic_l1_config, basic_layout):
        """Test that finish_write returns SUCCESS after proper write reservation."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Reserve write
        manager.reserve_write([key], [False], basic_layout)

        # Finish write
        result = manager.finish_write([key])

        assert result[key] == L1Error.SUCCESS

        # Verify object is now ready (not write-locked)
        state = manager.get_object_state(key)
        assert state is not None
        assert state.available_for_read() is True
        assert state.available_for_write() is True

        manager.close()

    def test_finish_write_non_write_locked_returns_wrong_state(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_write returns KEY_IN_WRONG_STATE if not write-locked."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object (not write-locked)
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Try to finish write again
        result = manager.finish_write([key])

        assert result[key] == L1Error.KEY_IN_WRONG_STATE

        manager.close()


# =============================================================================
# Tests for L1Manager.finish_write_and_reserve_read()
# =============================================================================


class TestFinishWriteAndReserveRead:
    """
    Tests for L1Manager.finish_write_and_reserve_read() method.

    This method atomically finishes write and acquires read lock,
    preventing a race window where eviction could interfere.

    Per the docstring:
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_IN_WRONG_STATE: Not write-locked, or already read-locked.
    - SUCCESS: Write unlocked and read lock acquired atomically.
    """

    def test_normal_transition(self, basic_l1_config, basic_layout):
        """Test normal write-locked -> read-locked transition."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Reserve write (key is now write-locked)
        write_result = manager.reserve_write([key], [False], basic_layout)
        assert write_result[key][0] == L1Error.SUCCESS

        # Atomically finish write and reserve read
        result = manager.finish_write_and_reserve_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.SUCCESS
        assert mem_obj is not None

        # Verify state: write unlocked, read locked
        state = manager.get_object_state(key)
        assert state is not None
        assert not state.write_lock.is_locked()
        assert state.read_lock.is_locked()

        # Should be readable (not write-locked)
        assert state.available_for_read() is True
        # Should not be writable (read-locked)
        assert state.available_for_write() is False

        # Clean up read lock
        manager.finish_read([key])
        manager.close()

    def test_key_not_exist(self, basic_l1_config, basic_layout):
        """Test that non-existing key returns KEY_NOT_EXIST."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.finish_write_and_reserve_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_NOT_EXIST
        assert mem_obj is None

        manager.close()

    def test_not_write_locked(self, basic_l1_config, basic_layout):
        """Test that non-write-locked key returns KEY_IN_WRONG_STATE."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object (not write-locked)
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        result = manager.finish_write_and_reserve_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_IN_WRONG_STATE
        assert mem_obj is None

        manager.close()

    def test_already_read_locked(self, basic_l1_config, basic_layout):
        """Test that key with both write+read locks returns KEY_IN_WRONG_STATE.

        This is an unexpected state — normally a key shouldn't be both
        write-locked and read-locked simultaneously.
        """
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create write-locked object
        manager.reserve_write([key], [False], basic_layout)

        # Force a read lock via internal state (unusual state)
        state = manager.get_object_state(key)
        assert state is not None
        state.read_lock.lock()

        result = manager.finish_write_and_reserve_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_IN_WRONG_STATE
        assert mem_obj is None

        # Clean up
        state.read_lock.unlock()
        manager.close()

    def test_multiple_keys_mixed_results(self, basic_l1_config, basic_layout):
        """Test with multiple keys where some succeed and some fail."""
        manager = L1Manager(basic_l1_config)
        key1 = make_object_key(1)
        key2 = make_object_key(2)  # will not exist
        key3 = make_object_key(3)

        # key1: write-locked (should succeed)
        manager.reserve_write([key1], [False], basic_layout)

        # key3: ready, not write-locked (should fail)
        manager.reserve_write([key3], [False], basic_layout)
        manager.finish_write([key3])

        result = manager.finish_write_and_reserve_read([key1, key2, key3])

        # key1: SUCCESS
        assert result[key1][0] == L1Error.SUCCESS
        assert result[key1][1] is not None

        # key2: KEY_NOT_EXIST
        assert result[key2][0] == L1Error.KEY_NOT_EXIST
        assert result[key2][1] is None

        # key3: KEY_IN_WRONG_STATE (not write-locked)
        assert result[key3][0] == L1Error.KEY_IN_WRONG_STATE
        assert result[key3][1] is None

        # Clean up
        manager.finish_read([key1])
        manager.close()

    def test_can_unsafe_read_after_transition(self, basic_l1_config, basic_layout):
        """Test that unsafe_read works on the transitioned key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Write and transition
        manager.reserve_write([key], [False], basic_layout)
        result = manager.finish_write_and_reserve_read([key])
        assert result[key][0] == L1Error.SUCCESS

        # unsafe_read should work (key is read-locked)
        read_result = manager.unsafe_read([key])
        assert read_result[key][0] == L1Error.SUCCESS
        assert read_result[key][1] is not None

        manager.finish_read([key])
        manager.close()


# =============================================================================
# Tests for L1Manager.delete()
# =============================================================================


class TestDelete:
    """
    Tests for L1Manager.delete() method.

    Per the docstring:
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_IS_LOCKED: The key is locked (either write-locked or read-locked).
    """

    def test_delete_non_existing_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that delete returns KEY_NOT_EXIST for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.delete([key])

        assert key in result
        assert result[key] == L1Error.KEY_NOT_EXIST

        manager.close()

    def test_delete_success(self, basic_l1_config, basic_layout):
        """Test that delete returns SUCCESS for unlocked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Verify object exists
        assert manager.get_object_state(key) is not None

        # Delete
        result = manager.delete([key])

        assert result[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is None

        manager.close()

    def test_delete_write_locked_returns_key_is_locked(
        self, basic_l1_config, basic_layout
    ):
        """Test that delete returns KEY_IS_LOCKED for write-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create write-locked object
        manager.reserve_write([key], [False], basic_layout)

        # Try to delete
        result = manager.delete([key])

        assert result[key] == L1Error.KEY_IS_LOCKED

        manager.close()

    def test_delete_read_locked_returns_key_is_locked(
        self, basic_l1_config, basic_layout
    ):
        """Test that delete returns KEY_IS_LOCKED for read-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read
        manager.reserve_read([key])

        # Try to delete
        result = manager.delete([key])

        assert result[key] == L1Error.KEY_IS_LOCKED

        manager.close()

    def test_delete_multiple_keys(self, basic_l1_config, basic_layout):
        """Test delete with multiple keys in a single call."""
        manager = L1Manager(basic_l1_config)
        key1 = make_object_key(1)
        key2 = make_object_key(2)
        key3 = make_object_key(3)

        # key1: ready (unlocked)
        manager.reserve_write([key1], [False], basic_layout)
        manager.finish_write([key1])

        # key2: does not exist

        # key3: write-locked
        manager.reserve_write([key3], [False], basic_layout)

        result = manager.delete([key1, key2, key3])

        assert result[key1] == L1Error.SUCCESS
        assert result[key2] == L1Error.KEY_NOT_EXIST
        assert result[key3] == L1Error.KEY_IS_LOCKED

        manager.close()

    def test_force_delete_removes_write_locked_key(self, basic_l1_config, basic_layout):
        """force=True deletes a write-locked key that non-force refuses."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout)  # write-locked

        assert manager.delete([key]) == {key: L1Error.KEY_IS_LOCKED}
        assert manager.delete([key], force=True) == {key: L1Error.SUCCESS}
        assert manager.get_object_state(key) is None

        manager.close()

    def test_force_delete_removes_read_locked_key(self, basic_l1_config, basic_layout):
        """force=True deletes a read-locked key that non-force refuses."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_read([key])  # read-locked

        assert manager.delete([key]) == {key: L1Error.KEY_IS_LOCKED}
        assert manager.delete([key], force=True) == {key: L1Error.SUCCESS}
        assert manager.get_object_state(key) is None

        manager.close()

    def test_force_delete_missing_key_still_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """force does not invent keys: a missing key still reports KEY_NOT_EXIST."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(999)

        assert manager.delete([key], force=True) == {key: L1Error.KEY_NOT_EXIST}

        manager.close()


# =============================================================================
# Tests for L1Manager.get_object_state()
# =============================================================================


class TestGetObjectState:
    """
    Tests for L1Manager.get_object_state() method.

    Per the docstring:
    - Returns the L1ObjectState if the object exists, None otherwise.
    """

    def test_get_object_state_non_existing_returns_none(
        self, basic_l1_config, basic_layout
    ):
        """Test that get_object_state returns None for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        state = manager.get_object_state(key)

        assert state is None

        manager.close()

    def test_get_object_state_existing_returns_state(
        self, basic_l1_config, basic_layout
    ):
        """Test that get_object_state returns L1ObjectState for existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        state = manager.get_object_state(key)

        assert state is not None
        # Verify we can use the state's methods
        assert state.available_for_read() is True
        assert state.available_for_write() is True

        manager.close()

    def test_get_object_state_write_locked(self, basic_l1_config, basic_layout):
        """Test get_object_state for write-locked objects."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create write-locked object
        manager.reserve_write([key], [False], basic_layout)

        state = manager.get_object_state(key)

        assert state is not None
        assert state.available_for_read() is False
        assert state.available_for_write() is False

        manager.close()

    def test_get_object_state_read_locked(self, basic_l1_config, basic_layout):
        """Test get_object_state for read-locked objects."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object, then read lock it
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_read([key])

        state = manager.get_object_state(key)

        assert state is not None
        # Read-locked is still readable
        assert state.available_for_read() is True
        # But not writable
        assert state.available_for_write() is False

        manager.close()


# =============================================================================
# Tests for L1Manager.close()
# =============================================================================


class TestClose:
    """
    Tests for L1Manager.close() method.

    Per the docstring:
    - Close the L1Manager and free all resources.
    """

    def test_close_empty_manager(self, basic_l1_config):
        """Test that close works on an empty manager."""
        manager = L1Manager(basic_l1_config)

        # Should not raise any exceptions
        manager.close()

    def test_close_with_objects(self, basic_l1_config, basic_layout):
        """Test that close frees all objects in the manager."""
        manager = L1Manager(basic_l1_config)

        # Create multiple objects
        keys = [make_object_key(i) for i in range(5)]
        manager.reserve_write(keys, [False] * 5, basic_layout)

        # Close should free all objects
        manager.close()

    def test_close_clears_objects(self, basic_l1_config, basic_layout):
        """Test that close clears all objects from the manager."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Verify object exists before close
        assert manager.get_object_state(key) is not None

        # Close
        manager.close()

        # After close, get_object_state should return None
        # (objects dict should be cleared)
        assert manager.get_object_state(key) is None


# =============================================================================
# Tests for state machine transitions (integration)
# =============================================================================


class TestStateMachineTransitions:
    """
    Integration tests verifying the state machine transitions as described in the
    L1Manager class docstring.
    """

    def test_full_write_read_cycle(self, basic_l1_config, basic_layout):
        """Test full cycle: None -> write_locked -> ready -> read_locked -> ready."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # None state (key doesn't exist)
        assert manager.get_object_state(key) is None

        # reserve_write: None -> write_locked
        result = manager.reserve_write([key], [False], basic_layout)
        assert result[key][0] == L1Error.SUCCESS
        state = manager.get_object_state(key)
        assert state.available_for_read() is False
        assert state.available_for_write() is False

        # finish_write: write_locked -> ready
        result = manager.finish_write([key])
        assert result[key] == L1Error.SUCCESS
        state = manager.get_object_state(key)
        assert state.available_for_read() is True
        assert state.available_for_write() is True

        # reserve_read: ready -> read_locked
        result = manager.reserve_read([key])
        assert result[key][0] == L1Error.SUCCESS
        state = manager.get_object_state(key)
        assert state.available_for_read() is True
        assert state.available_for_write() is False

        # finish_read: read_locked -> ready
        result = manager.finish_read([key])
        assert result[key] == L1Error.SUCCESS
        state = manager.get_object_state(key)
        assert state.available_for_read() is True
        assert state.available_for_write() is True

        manager.close()

    def test_full_write_read_with_unsafe_read(self, basic_l1_config, basic_layout):
        """Test cycle with unsafe_read between reserve_read and finish_read."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read
        reserve_result = manager.reserve_read([key])
        assert reserve_result[key][0] == L1Error.SUCCESS

        # Multiple unsafe_reads should all succeed
        for _ in range(3):
            unsafe_result = manager.unsafe_read([key])
            assert unsafe_result[key][0] == L1Error.SUCCESS

        # Finish read
        finish_result = manager.finish_read([key])
        assert finish_result[key] == L1Error.SUCCESS

        # Object should still exist (not temporary)
        assert manager.get_object_state(key) is not None

        manager.close()

    def test_delete_from_ready_state(self, basic_l1_config, basic_layout):
        """Test deletion from ready state."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # delete: ready -> None
        result = manager.delete([key])
        assert result[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is None

        manager.close()

    def test_temporary_object_lifecycle(self, basic_l1_config, basic_layout):
        """Test temporary object lifecycle: deleted after last read."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # Read and release
        manager.reserve_read([key])
        manager.finish_read([key])

        # Object should be deleted
        assert manager.get_object_state(key) is None

        manager.close()

    def test_temporary_object_with_unsafe_read(self, basic_l1_config, basic_layout):
        """Test temporary object with unsafe_read doesn't affect deletion."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # Reserve read
        manager.reserve_read([key])

        # Multiple unsafe_reads
        for _ in range(5):
            result = manager.unsafe_read([key])
            assert result[key][0] == L1Error.SUCCESS

        # Single finish_read should delete the object
        manager.finish_read([key])
        assert manager.get_object_state(key) is None

        manager.close()

    def test_multi_reader_lifecycle_with_extra_count(
        self, basic_l1_config, basic_layout
    ):
        """Full lifecycle using extra_count (MLA TP>1 scenario).

        Simulates multiple workers sharing the same key:
        reserve_read(extra_count=N-1) acquires N locks,
        finish_read(extra_count=N-1) releases them all.
        """
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)
        extra = 3  # total locks = 1 + 3 = 4

        # write -> ready
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # reserve_read with extra_count
        result = manager.reserve_read([key], extra_count=extra)
        assert result[key][0] == L1Error.SUCCESS

        # unsafe_read should work while read-locked
        ur = manager.unsafe_read([key])
        assert ur[key][0] == L1Error.SUCCESS

        # finish_read with same extra_count
        fr = manager.finish_read([key], extra_count=extra)
        assert fr[key] == L1Error.SUCCESS

        # All locks released -> writable again
        state = manager.get_object_state(key)
        assert state is not None
        assert state.available_for_write() is True

        manager.close()

    def test_temp_object_multi_reader_deletion(self, basic_l1_config, basic_layout):
        """Temporary object deleted after extra_count release."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)
        extra = 2  # total locks = 1 + 2 = 3

        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        manager.reserve_read([key], extra_count=extra)
        assert manager.get_object_state(key) is not None

        manager.finish_read([key], extra_count=extra)
        assert manager.get_object_state(key) is None

        manager.close()


# =============================================================================
# Thread safety tests
# =============================================================================


class TestThreadSafety:
    """Tests verifying thread-safety of L1Manager operations."""

    def test_concurrent_reserve_write_different_keys(
        self, basic_l1_config, basic_layout
    ):
        """Test concurrent reserve_write on different keys."""
        manager = L1Manager(basic_l1_config)
        num_threads = 8
        keys_per_thread = 5
        results = []
        lock = threading.Lock()

        def worker(thread_id):
            thread_keys = [
                make_object_key(thread_id * 1000 + i) for i in range(keys_per_thread)
            ]
            is_temporary = [False] * keys_per_thread
            result = manager.reserve_write(thread_keys, is_temporary, basic_layout)
            with lock:
                results.append((thread_keys, result))

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All writes should succeed
        assert len(results) == num_threads
        for keys, result in results:
            for key in keys:
                assert result[key][0] == L1Error.SUCCESS

        manager.close()

    def test_concurrent_read_same_key(self, basic_l1_config, basic_layout):
        """Test concurrent reads on the same key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        num_threads = 10
        results = []
        lock = threading.Lock()

        def worker():
            result = manager.reserve_read([key])
            with lock:
                results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        assert len(results) == num_threads
        for result in results:
            assert result[key][0] == L1Error.SUCCESS

        manager.close()

    def test_concurrent_unsafe_read_same_key(self, basic_l1_config, basic_layout):
        """Test concurrent unsafe_reads on the same read-locked key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object and reserve read
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_read([key])

        num_threads = 10
        results = []
        lock = threading.Lock()

        def worker():
            result = manager.unsafe_read([key])
            with lock:
                results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All unsafe_reads should succeed
        assert len(results) == num_threads
        for result in results:
            assert result[key][0] == L1Error.SUCCESS

        manager.close()

    def test_concurrent_read_write_mixed_operations(
        self, basic_l1_config, basic_layout
    ):
        """Test concurrent mixed operations don't cause crashes."""
        manager = L1Manager(basic_l1_config)
        num_threads = 8
        operations_per_thread = 10
        errors = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for i in range(operations_per_thread):
                    key = make_object_key(thread_id * 1000 + i)

                    # Write cycle
                    write_result = manager.reserve_write([key], [False], basic_layout)
                    if write_result[key][0] == L1Error.SUCCESS:
                        manager.finish_write([key])

                        # Read cycle with unsafe_read
                        read_result = manager.reserve_read([key])
                        if read_result[key][0] == L1Error.SUCCESS:
                            # Do some unsafe_reads
                            manager.unsafe_read([key])
                            manager.unsafe_read([key])
                            manager.finish_read([key])

                        # Delete
                        manager.delete([key])
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should have occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"

        manager.close()


# =============================================================================
# Tests for L1Manager.is_key_evictable()
# =============================================================================


class TestIsKeyEvictable:
    """Tests for L1Manager.is_key_evictable() method."""

    def test_evictable_key_in_ready_state(self, basic_l1_config, basic_layout):
        """A key that has been written and finished should be evictable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Write and finish -> key is in ready state
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        assert manager.is_key_evictable(key) is True

        manager.close()

    def test_write_locked_key_is_not_evictable(self, basic_l1_config, basic_layout):
        """A write-locked key should not be evictable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Reserve write but don't finish -> key is write-locked
        manager.reserve_write([key], [False], basic_layout)

        assert manager.is_key_evictable(key) is False

        # Cleanup
        manager.finish_write([key])
        manager.close()

    def test_read_locked_key_is_not_evictable(self, basic_l1_config, basic_layout):
        """A read-locked key should not be evictable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Write, finish, then reserve read -> key is read-locked
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_read([key])

        assert manager.is_key_evictable(key) is False

        # Cleanup
        manager.finish_read([key])
        manager.close()

    def test_nonexistent_key_is_not_evictable(self, basic_l1_config):
        """A key that does not exist should not be evictable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(999)

        assert manager.is_key_evictable(key) is False

        manager.close()

    def test_key_becomes_evictable_after_read_unlock(
        self, basic_l1_config, basic_layout
    ):
        """A key should become evictable after all read locks are released."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Write and finish
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read -> not evictable
        manager.reserve_read([key])
        assert manager.is_key_evictable(key) is False

        # Finish read -> evictable again
        manager.finish_read([key])
        assert manager.is_key_evictable(key) is True

        manager.close()

    def test_key_becomes_evictable_after_write_unlock(
        self, basic_l1_config, basic_layout
    ):
        """A key should become evictable after write lock is released."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Reserve write -> not evictable
        manager.reserve_write([key], [False], basic_layout)
        assert manager.is_key_evictable(key) is False

        # Finish write -> evictable
        manager.finish_write([key])
        assert manager.is_key_evictable(key) is True

        manager.close()

    def test_multiple_keys_mixed_evictability(self, basic_l1_config, basic_layout):
        """Test evictability with multiple keys in different states."""
        manager = L1Manager(basic_l1_config)
        key_ready = make_object_key(1)
        key_write_locked = make_object_key(2)
        key_read_locked = make_object_key(3)

        # key_ready: write + finish -> ready state
        manager.reserve_write([key_ready], [False], basic_layout)
        manager.finish_write([key_ready])

        # key_write_locked: write only -> write-locked
        manager.reserve_write([key_write_locked], [False], basic_layout)

        # key_read_locked: write + finish + read -> read-locked
        manager.reserve_write([key_read_locked], [False], basic_layout)
        manager.finish_write([key_read_locked])
        manager.reserve_read([key_read_locked])

        assert manager.is_key_evictable(key_ready) is True
        assert manager.is_key_evictable(key_write_locked) is False
        assert manager.is_key_evictable(key_read_locked) is False

        # Cleanup
        manager.finish_write([key_write_locked])
        manager.finish_read([key_read_locked])
        manager.close()

    def test_deleted_key_is_not_evictable(self, basic_l1_config, basic_layout):
        """A key that has been deleted should not be evictable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Write, finish, then delete
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.delete([key])

        assert manager.is_key_evictable(key) is False

        manager.close()

    def test_key_with_multiple_read_locks_not_evictable(
        self, basic_l1_config, basic_layout
    ):
        """A key with extra_count read locks should not be evictable
        until all locks are released."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Write and finish
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read with extra_count=2 (total 3 locks)
        manager.reserve_read([key], extra_count=2)
        assert manager.is_key_evictable(key) is False

        # Release only 1 lock (extra_count=0) -> still locked
        manager.finish_read([key], extra_count=0)
        assert manager.is_key_evictable(key) is False

        # Release remaining 2 locks (extra_count=1)
        manager.finish_read([key], extra_count=1)
        assert manager.is_key_evictable(key) is True

        manager.close()
