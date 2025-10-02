# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import traceback
import unittest

from vllm.distributed.device_communicators.shm_object_storage import (
    SingleWriterShmRingBuffer)


class TestSingleWriterShmRingBuffer(unittest.TestCase):
    """Test suite for the ring buffer implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.buffer_size = 4096
        self.ring_buffer = None

    def tearDown(self):
        """Clean up after tests"""
        if self.ring_buffer:
            del self.ring_buffer

    def test_buffer_opening(self):
        """Test opening an existing buffer"""
        # First create a buffer
        self.ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=self.buffer_size, create=True)

        # Then open it with another instance
        reader_buffer = SingleWriterShmRingBuffer(*self.ring_buffer.handle())
        self.assertFalse(reader_buffer.is_writer)
        self.assertEqual(reader_buffer.shared_memory.name,
                         self.ring_buffer.shared_memory.name)

    def test_buffer_access(self):
        """Test accessing allocated buffers"""
        self.ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=self.buffer_size, create=True)

        size = 100
        address, monotonic_id = self.ring_buffer.allocate_buf(size)

        # Write some test data
        test_data = b"Hello, World!" * 7  # 91 bytes
        with self.ring_buffer.access_buf(address) as (data_buf, metadata):
            data_buf[0:len(test_data)] = test_data

        # Read it back
        with self.ring_buffer.access_buf(address) as (data_buf2, metadata2):
            read_data = bytes(data_buf2[0:len(test_data)])
            read_id = metadata2[0]

        self.assertEqual(read_data, test_data)
        self.assertEqual(read_id, monotonic_id)

    def test_memory_error_on_full_buffer(self):
        """Test that MemoryError is raised when buffer is full"""
        small_buffer_size = 200
        self.ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=small_buffer_size, create=True)

        # Fill up the buffer
        self.ring_buffer.allocate_buf(100)
        self.ring_buffer.allocate_buf(80)  # Total: 196 bytes used

        # This should fail
        with self.assertRaises(MemoryError):
            self.ring_buffer.allocate_buf(1)  # Would exceed buffer capacity

    def test_allocation_and_free(self):
        """Test allocation and freeing of buffers"""
        small_buffer_size = 200
        self.ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=small_buffer_size, create=True)

        size = 80
        # Write some data
        test_data = b"Repeated test data"
        for i in range(5):
            address, monotonic_id = self.ring_buffer.allocate_buf(size)
            with self.ring_buffer.access_buf(address) as (data_buf, metadata):
                data_buf[0:4] = (0).to_bytes(4, "little")  # 0 for not in-use
                data_buf[4:len(test_data) + 4] = test_data
            print(self.ring_buffer.metadata)
            freed_ids = self.ring_buffer.free_buf(lambda *args: True)
            print(f"  Freed IDs: {freed_ids}")
            self.assertEqual(freed_ids[0], i)

    def test_clear_buffer(self):
        """Test clearing the buffer"""
        self.ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=self.buffer_size, create=True)

        # Allocate some buffers
        for _ in range(3):
            self.ring_buffer.allocate_buf(100)

        # Clear the buffer
        self.ring_buffer.clear()

        # Check that metadata is empty and IDs reset
        self.assertEqual(len(self.ring_buffer.metadata), 0)
        self.assertEqual(self.ring_buffer.monotonic_id_start, 0)
        self.assertEqual(self.ring_buffer.monotonic_id_end, 0)
        self.assertEqual(self.ring_buffer.data_buffer_start, 0)
        self.assertEqual(self.ring_buffer.data_buffer_end, 0)


def main():
    """Main function demonstrating usage and running tests"""
    print("=== SingleWriterShmRingBuffer Test Suite ===\n")

    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[""], exit=False, verbosity=2)

    print("\n" + "=" * 50)
    print("=== Manual Demo ===\n")

    # Manual demonstration
    try:
        print("Creating ring buffer...")
        writer_buffer = SingleWriterShmRingBuffer(data_buffer_size=2048,
                                                  create=True)
        reader_buffer = SingleWriterShmRingBuffer(*writer_buffer.handle())

        print(f"Buffer created with name: {writer_buffer.shared_memory.name}")

        # Allocate some buffers
        print("\nAllocating buffers...")
        address_array = []
        for i in range(3):
            size = 100 + i * 50
            try:
                writer_buffer.free_buf(lambda *args: True)
                address, monotonic_id = writer_buffer.allocate_buf(size)
                address_array.append((address, size, monotonic_id))

                # Write some test data
                with writer_buffer.access_buf(address) as (data_buf, metadata):
                    test_message = f"Test message {i}".encode()
                    data_buf[0:len(test_message)] = test_message

            except MemoryError as e:
                print(f"  Failed to allocate {size} bytes: {e}")

        print("\nBuffer state:")
        print(f"  Data buffer start: {writer_buffer.data_buffer_start}")
        print(f"  Data buffer end: {writer_buffer.data_buffer_end}")
        print(f"  Monotonic ID start: {writer_buffer.monotonic_id_start}")
        print(f"  Monotonic ID end: {writer_buffer.monotonic_id_end}")
        print(f"  Metadata entries: {len(writer_buffer.metadata)}")

        # Try to read back the data
        print("\nReading back data...")
        for address, size, monotonic_id in address_array:
            with reader_buffer.access_buf(address) as (data_buf, metadata):
                # Find null terminator or read first 50 chars
                data_bytes = bytes(data_buf[0:size])
                message = data_bytes.decode()
                print(f"  ID {monotonic_id}: '{message}'")

    except Exception as e:
        print(f"Demo error: {e}")
        traceback.print_exc()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
