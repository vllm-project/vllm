# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import random
import time
import unittest
from multiprocessing import Lock

# Assuming these are imported from your module
from vllm.distributed.device_communicators.shm_object_storage import (
    SingleWriterShmObjectStorage, SingleWriterShmRingBuffer)


class TestSingleWriterShmObjectStorage(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=1024 * 100,
            create=True,  # 10 MB buffer
            is_free_fn=SingleWriterShmObjectStorage.default_is_free_check,
        )
        self.storage = SingleWriterShmObjectStorage(
            max_object_size=1024 * 10,  # 10KB max object
            n_readers=2,
            ring_buffer=ring_buffer,
            reader_lock=Lock(),
        )

    def tearDown(self):
        """Clean up after each test."""
        if self.storage:
            del self.storage

    def test_minimal_put_get_cycle(self):
        """Test basic put and get operations."""
        key = "test_key"
        value = "hello world"

        # Put operation
        address, monotonic_id = self.storage.put(key, value)

        # Verify key is in index
        self.assertIn(key, self.storage.key_index)
        self.assertEqual(self.storage.key_index[key], (address, monotonic_id))
        self.assertEqual(self.storage.id_index[monotonic_id], key)

        # Get operation
        result = self.storage.get(address, monotonic_id)

        # Verify result
        self.assertEqual(result, value)

    def test_various_data_types(self):
        """Test serialization and storage of various Python data types."""
        test_cases = [
            ("string_key", "hello world"),
            ("int_key", 42),
            ("float_key", 3.14159),
            ("list_key", [1, 2, 3, "four", 5.0]),
            ("dict_key", {
                "name": "Alice",
                "age": 30,
                "scores": [95, 87, 92]
            }),
            ("tuple_key", (1, "two", 3.0, [4, 5])),
            (
                "nested_key",
                {
                    "users": [
                        {
                            "id": 1,
                            "name": "Alice",
                            "metadata": {
                                "active": True
                            }
                        },
                        {
                            "id": 2,
                            "name": "Bob",
                            "metadata": {
                                "active": False
                            }
                        },
                    ],
                    "config": {
                        "version": "1.0",
                        "settings": [1, 2, 3]
                    },
                },
            ),
            ("none_key", None),
            ("bool_key", True),
            ("bytes_key", b"binary data"),
        ]

        addresses_and_ids = []

        # Store all test data
        for key, value in test_cases:
            with self.subTest(operation="put", key=key):
                address, monotonic_id = self.storage.put(key, value)
                addresses_and_ids.append((key, value, address, monotonic_id))

        # Retrieve and verify all test data
        for key, original_value, address, monotonic_id in addresses_and_ids:
            with self.subTest(operation="get", key=key):
                retrieved_value = self.storage.get(address, monotonic_id)
                self.assertEqual(retrieved_value, original_value)

    def test_put_same_key_twice(self):
        """Test behavior when putting the same key multiple times."""
        key = "duplicate_key"
        value1 = "first value"
        value2 = "second value"

        # First put
        address1, id1 = self.storage.put(key, value1)
        retrieved1 = self.storage.get(address1, id1)
        self.assertEqual(retrieved1, value1)

        # Second put with same key but different value
        address2, id2 = self.storage.put(key, value2)

        # Should return same address and ID (cache hit)
        self.assertEqual(address1, address2)
        self.assertEqual(id1, id2)

        retrieved2 = self.storage.get(address2, id2)
        # data will not be overwritten for the same key,
        # should still be the first value
        self.assertEqual(retrieved2, value1)

    def test_large_object_rejection(self):
        """Test that objects exceeding max_object_size are rejected."""
        # Create an object larger than max_object_size
        large_data = "x" * (self.storage.max_object_size + 100)

        with self.assertRaises(ValueError) as context:
            self.storage.put("large_key", large_data)

        self.assertIn("exceeds max object size", str(context.exception))

    def test_buffer_overflow_and_cleanup(self):
        """Test behavior when buffer fills up and needs cleanup."""
        # Fill up the buffer with many small objects
        stored_items = []

        try:
            for i in range(1000):  # Try to store many items
                key = f"item_{i}"
                value = f"data_{i}" * 100  # Make it reasonably sized
                address, monotonic_id = self.storage.put(key, value)
                stored_items.append((key, value, address, monotonic_id))
        except MemoryError:
            print(f"Buffer filled after {len(stored_items)} items")

        # Verify that some items are still accessible
        accessible_count = 0
        for key, original_value, address, monotonic_id in stored_items:
            for i in range(self.storage.n_readers):
                retrieved = self.storage.get(address, monotonic_id)
            if retrieved == original_value:
                accessible_count += 1

        self.assertEqual(accessible_count, len(stored_items))

        try:
            for i in range(len(stored_items), 1000):  # Try to store many items
                key = f"item_{i}"
                value = f"data_{i}" * 100  # Make it reasonably sized
                address, monotonic_id = self.storage.put(key, value)
                stored_items.append((key, value, address, monotonic_id))
        except MemoryError:
            print(f"Buffer filled after {len(stored_items)} items")

        # Verify that some items are still accessibles
        for key, original_value, address, monotonic_id in stored_items:
            try:
                for i in range(self.storage.n_readers):
                    retrieved = self.storage.get(address, monotonic_id)
                if retrieved == original_value:
                    accessible_count += 1
            except ValueError as e:
                print(f"Error retrieving {key}: {e}")

        # some items from the first batch may still be accessible
        self.assertGreaterEqual(accessible_count, len(stored_items))

    def test_invalid_get_operations(self):
        """Test various invalid get operations."""
        # Test with non-existent address
        with self.assertRaises(ValueError):  # Could be various exceptions
            self.storage.get(99999, 1)

        # Store something first
        address, monotonic_id = self.storage.put("test", "value")

        # Test with wrong monotonic_id
        with self.assertRaises(ValueError) as context:
            self.storage.get(address, monotonic_id + 100)

        self.assertIn("has been modified or is invalid", \
            str(context.exception))

    def test_clear_storage(self):
        """Test clearing the storage."""
        # Store some items
        for i in range(5):
            self.storage.put(f"item_{i}", f"value_{i}")

        # Clear the storage
        self.storage.clear()

        # Verify that all indices are empty
        self.assertEqual(len(self.storage.key_index), 0)
        self.assertEqual(len(self.storage.id_index), 0)
        self.assertEqual(len(self.storage.ring_buffer.metadata), 0)

        # Verify that new items can be added after clearing
        address, monotonic_id = self.storage.put("new_item", "new_value")
        self.assertIn("new_item", self.storage.key_index)
        self.assertEqual((address, monotonic_id), (0, 0))


# Reader process function
def reader_process(process_id, storage_handle, items_to_read):
    """Reader process that connects to existing shared memory and reads data."""
    reader_storage = SingleWriterShmObjectStorage.create_from_handle(
        storage_handle)

    print(f"Reader {process_id} started")

    errors = []

    for key, original_value, address, monotonic_id in items_to_read:
        time.sleep(random.random() / 100)
        try:
            # Read data from shared memory
            retrieved_value = reader_storage.get(address, monotonic_id)

            # Verify data integrity
            assert retrieved_value == original_value
            print(f"Reader {process_id} retrieved {key}: {retrieved_value}")
        except Exception as e:
            errors.append((key, str(e), type(e).__name__))


def run_multiprocess_example():
    """Run a minimal working example with real shared memory."""
    print("=== Minimal Object Storage Example ===")

    try:
        # Create storage instance
        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=1024 * 100,
            create=True,  # 10 MB buffer
            is_free_fn=SingleWriterShmObjectStorage.default_is_free_check,
        )
        storage = SingleWriterShmObjectStorage(
            max_object_size=1024,
            n_readers=3,
            ring_buffer=ring_buffer,
            reader_lock=Lock(),
        )

        print(f"Created storage (writer: {storage.is_writer})")

        # Test basic data types
        test_data = [
            ("user_data", {
                "name": "Alice",
                "age": 30,
                "scores": [95, 87, 92]
            }),
            ("simple_string", "Hello, World!"),
            ("number", 42),
            ("list_data", [1, 2, 3, "four", 5.0]),
        ]

        stored_items = []

        # Store all data
        for key, value in test_data:
            print(f"Storing {key}: {value}")
            address, monotonic_id = storage.put(key, value)
            stored_items.append((key, value, address, monotonic_id))
            print(f"  -> Stored at address {address}, ID {monotonic_id}")

        print("\n--- Retrieving Data ---")
        processes = []
        handle = storage.handle()
        # initialize lock for reader processes
        handle.reader_lock = Lock()
        for i in range(storage.n_readers):
            p = multiprocessing.Process(target=reader_process,
                                        args=(i, handle, stored_items))
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()

    except Exception as e:
        print(f"Error in minimal example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the minimal example first
    run_multiprocess_example()
    print("\n" + "=" * 50 + "\n")

    # Run the test suite
    print("Running comprehensive test suite...")
    unittest.main(verbosity=2, exit=False)
