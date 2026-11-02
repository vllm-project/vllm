# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Actual packaged process-shared Endpoint controls; no substituted queue objects."""

from __future__ import annotations

import multiprocessing
import struct
import sys
import threading
import time
import unittest
from multiprocessing import shared_memory
from typing import Any, ClassVar


def native_reader(name, readers, capacity, rank, count, connection):
    """Separate-process consumer checks publication of the shared payload ring."""
    from vllm._asa_shmwait import Endpoint

    shm = shared_memory.SharedMemory(name=name)
    endpoint = None
    try:
        endpoint = Endpoint(shm.buf, readers, capacity, rank, False, 128)
        connection.send("ready")
        payload = 128 + readers * 128
        for sequence in range(count):
            index = endpoint.acquire(timeout=5)
            if struct.unpack_from("Q", shm.buf, payload + index * 8)[0] != sequence:
                raise AssertionError("shared payload published out of order")
            endpoint.release()
        connection.send(("complete", count))
    except BaseException as error:
        connection.send(("error", type(error).__name__, str(error)))
    finally:
        if endpoint is not None:
            del endpoint
        shm.close()
        connection.close()


class NativeControlTests(unittest.TestCase):
    """Packaged process-shared controls; separate from MessageQueue integration."""

    Endpoint: ClassVar[Any]

    @classmethod
    def setUpClass(cls):
        from vllm._asa_shmwait import Endpoint

        cls.Endpoint = Endpoint

    def setUp(self):
        self.shm = shared_memory.SharedMemory(create=True, size=128 + 128 + 4 * 8)
        self.writer = self.Endpoint(self.shm.buf, 1, 4, -1, True, 128)
        self.reader = self.Endpoint(self.shm.buf, 1, 4, 0, False, 128)

    def tearDown(self):
        del self.reader, self.writer
        self.shm.close()
        self.shm.unlink()

    def round_trip(self, readers):
        ctx = multiprocessing.get_context("spawn")
        capacity = 4
        count = 64
        shm = shared_memory.SharedMemory(
            create=True, size=128 + readers * 128 + capacity * 8
        )
        writer = self.Endpoint(shm.buf, readers, capacity, -1, True, 128)
        processes = []
        pipes = []
        try:
            for rank in range(readers):
                parent, child = ctx.Pipe()
                pipes.append(parent)
                process = ctx.Process(
                    target=native_reader,
                    args=(shm.name, readers, capacity, rank, count, child),
                )
                process.start()
                child.close()
                processes.append(process)
            for pipe in pipes:
                self.assertTrue(pipe.poll(10), "reader startup did not complete")
                self.assertEqual(pipe.recv(), "ready")
            for sequence in range(count):
                index = writer.acquire(timeout=5)
                struct.pack_into(
                    "Q", shm.buf, 128 + readers * 128 + index * 8, sequence
                )
                writer.release()
            for pipe, process in zip(pipes, processes):
                self.assertTrue(pipe.poll(10), "reader completion did not arrive")
                self.assertEqual(pipe.recv(), ("complete", count))
                process.join(5)
                self.assertEqual(process.exitcode, 0)
        finally:
            writer.cancel()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
            for pipe in pipes:
                pipe.close()
            del writer
            shm.close()
            shm.unlink()

    def test_native_one_reader_publication(self):
        self.round_trip(1)

    def test_native_two_reader_publication(self):
        self.round_trip(2)

    def test_native_four_reader_publication(self):
        self.round_trip(4)

    def test_native_eight_reader_publication(self):
        self.round_trip(8)

    def test_native_deadline_unregisters_waiter(self):
        with self.assertRaises(TimeoutError):
            self.reader.acquire(timeout=0.02)
        self.assertEqual(self.reader.snapshot()["reader_waiters"], 0)

    def test_native_aborted_write_does_not_publish(self):
        self.writer.acquire(timeout=0)
        self.writer.release(False)
        self.assertEqual(self.writer.snapshot()["published"], 0)
        with self.assertRaises(TimeoutError):
            self.reader.acquire(timeout=0)

    def test_native_writer_close_drains_committed_payload(self):
        self.writer.acquire(timeout=0)
        self.writer.release()
        self.writer.cancel()
        self.assertEqual(self.reader.acquire(timeout=0), 0)
        self.reader.release()
        with self.assertRaises(BrokenPipeError):
            self.reader.acquire(timeout=0)

    def test_native_reader_close_rejects_writer(self):
        self.reader.cancel()
        with self.assertRaises(BrokenPipeError):
            self.writer.acquire(timeout=0)

    def test_native_cancel_wakes_blocked_reader(self):
        errors = []

        def wait():
            try:
                self.reader.acquire(timeout=2)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=wait)
        thread.start()
        deadline = time.monotonic() + 1
        while (
            self.reader.snapshot()["reader_waiters"] == 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        try:
            self.assertEqual(self.reader.snapshot()["reader_waiters"], 1)
        finally:
            self.reader.cancel()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)

    def test_native_invalid_geometry_cannot_reinitialize_shared_control(self):
        with self.assertRaises(ValueError):
            self.Endpoint(self.shm.buf, 1, 4, 0, True, 128)
        with self.assertRaises(ValueError):
            self.Endpoint(self.shm.buf, 1, 3, 0, False, 128)
        self.assertEqual(self.writer.snapshot()["published"], 0)

    def test_native_release_requires_acquisition(self):
        with self.assertRaises(RuntimeError):
            self.reader.release()

    def test_packaged_heap_type_survives_repeated_endpoint_destruction(self):
        before = sys.getrefcount(self.Endpoint)
        for _ in range(1000):
            endpoint = self.Endpoint(self.shm.buf, 1, 4, 0, False, 128)
            self.assertEqual(type(endpoint).__module__, "vllm._asa_shmwait")
            del endpoint
        self.assertEqual(sys.getrefcount(self.Endpoint), before)


if __name__ == "__main__":
    unittest.main()
