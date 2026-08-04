#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
IDC end-to-end verification script for shared memory
connector (TCP mode).

This script runs INSIDE the container.  The shm_file_worker
must be started on the HOST MACHINE first (see instructions
below).

Usage
-----

Step 1 (on the HOST machine):
  mkdir -p tests/v1/shm_allocator/csrc/build && cd tests/v1/shm_allocator/csrc/build
  cmake .. && make shm_file_worker
  ./shm_file_worker --listen 0.0.0.0:9800

Step 2 (INSIDE the container):
  python tests/v1/shm_allocator/verify_shmfile_tcp.py \
      --worker-addr <HOST_IP>:9800 \
      --shm-name /lmcache_verify \
      --shm-size 8388608 \
      --storage-dir /tmp/shm_verify_data
"""

# Standard
from multiprocessing import shared_memory
import argparse
import ctypes
import hashlib
import os
import random
import socket
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description="Verify SHM + TCP worker roundtrip")
    p.add_argument(
        "--worker-addr",
        required=True,
        help="host:port of the shm_file_worker",
    )
    p.add_argument(
        "--shm-name",
        default="/lmcache_verify",
        help="POSIX shm name (must start with /)",
    )
    p.add_argument(
        "--shm-size-gb",
        type=int,
        default=1,
        help="Size of shared memory in GB (default 1 GB)",
    )
    p.add_argument(
        "--storage-dir",
        default="/tmp/shm_verify_data",
        help="Directory for temporary file storage",
    )
    p.add_argument(
        "--data-size-gb",
        type=float,
        default=0.001,
        help="Size of test data in GB (default 0.001 GB)",
    )
    return p.parse_args()


class ShmRegion:
    """Manage a POSIX shared memory region."""

    def __init__(self, name: str, size: int):
        self.name = name.lstrip("/")
        self.size = size
        # Clean up stale segment if any
        try:
            old = shared_memory.SharedMemory(name=self.name, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass

        self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=size)
        arr_t = ctypes.c_uint8 * size
        self._arr = arr_t.from_buffer(self.shm.buf)
        self.base_addr = ctypes.addressof(self._arr)
        print(
            "[OK] Created shm /%s, size=%d, "
            "base_addr=%d" % (self.name, size, self.base_addr)
        )

    def write_bytes(self, offset: int, data: bytes):
        self.shm.buf[offset : offset + len(data)] = data

    def read_bytes(self, offset: int, length: int) -> bytes:
        return bytes(self.shm.buf[offset : offset + length])

    def zero(self, offset: int, length: int):
        self.shm.buf[offset : offset + length] = b"\x00" * length

    def close(self):
        del self._arr
        self.shm.close()
        self.shm.unlink()
        print("[OK] Cleaned up shm /%s" % self.name)


class TcpClient:
    """Simple TCP client for the shm_file_worker protocol."""

    def __init__(self, addr: str):
        host, port_str = addr.rsplit(":", 1)
        port = int(port_str)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self._buf = b""
        print("[OK] Connected to worker at %s" % addr)

    def send_cmd(self, cmd: str) -> str:
        self.sock.sendall((cmd + "\n").encode("utf-8"))
        while b"\n" not in self._buf:
            data = self.sock.recv(4096)
            if not data:
                return "ERROR connection closed"
            self._buf += data
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("utf-8").strip()

    def close(self):
        try:
            self.send_cmd("QUIT")
        except Exception:
            pass
        self.sock.close()


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def main():
    args = parse_args()
    os.makedirs(args.storage_dir, exist_ok=True)
    shm_size = int(args.shm_size_gb * 1024 * 1024 * 1024)
    data_size = int(args.data_size_gb * 1024 * 1024 * 1024)
    print("=" * 60)
    print("SHM File Connector - IDC Verification")
    print("=" * 60)
    print("Worker addr : %s" % args.worker_addr)
    print("SHM name    : %s" % args.shm_name)
    print("SHM size    : %d GB" % args.shm_size_gb)
    print("Data size   : %d GB" % args.data_size_gb)
    print("Storage dir : %s" % args.storage_dir)
    print()

    # 1. Create shared memory
    print("[Step 1] Creating POSIX shared memory...")
    shm = ShmRegion(args.shm_name, shm_size)

    # 2. Connect to remote worker
    print("[Step 2] Connecting to remote worker...")
    client = TcpClient(args.worker_addr)

    # 3. ATTACH
    print("[Step 3] Sending ATTACH command...")
    resp = client.send_cmd("ATTACH %s %d %d" % (args.shm_name, shm_size, shm.base_addr))
    if not resp.startswith("OK"):
        print("[FAIL] ATTACH failed: %s" % resp)
        sys.exit(1)
    print("[OK] ATTACH response: %s" % resp)

    # 4. Generate random test data and write to shm
    print("[Step 4] Generating random test data...")
    random.seed(42)
    test_data = random.randbytes(data_size)
    original_md5 = md5(test_data)
    print("  Original data MD5: %s (%d bytes)" % (original_md5, len(test_data)))

    offset = 0
    shm.write_bytes(offset, test_data)

    # 5. WRITE: worker reads from shm and writes to file
    file_path = os.path.join(args.storage_dir, "verify_test.data")
    data_ptr = shm.base_addr + offset
    print("[Step 5] WRITE: shm -> file (%s)..." % file_path)
    t0 = time.time()
    resp = client.send_cmd("WRITE %s %d %d" % (file_path, data_ptr, data_size))
    t1 = time.time()
    if not resp.startswith("OK"):
        print("[FAIL] WRITE failed: %s" % resp)
        sys.exit(1)
    print("[OK] WRITE response: %s (%.3f ms)" % (resp, (t1 - t0) * 1000))

    # Verify file content from disk
    with open(file_path, "rb") as f:
        file_data = f.read()
    file_md5 = md5(file_data)
    print("  File on disk MD5 : %s" % file_md5)
    if file_md5 != original_md5:
        print("[FAIL] WRITE data mismatch!")
        sys.exit(1)
    print("[OK] WRITE data verified!")

    # 6. Zero the shm region
    print("[Step 6] Zeroing shm region...")
    shm.zero(offset, data_size)
    zero_md5 = md5(shm.read_bytes(offset, data_size))
    print("  Zeroed shm MD5   : %s" % zero_md5)

    # 7. READ: worker reads from file and writes to shm
    print("[Step 7] READ: file -> shm (%s)..." % file_path)
    t0 = time.time()
    resp = client.send_cmd("READ %s %d %d" % (file_path, data_ptr, data_size))
    t1 = time.time()
    if not resp.startswith("OK"):
        print("[FAIL] READ failed: %s" % resp)
        sys.exit(1)
    print("[OK] READ response: %s (%.3f ms)" % (resp, (t1 - t0) * 1000))

    # 8. Verify data in shm matches original
    print("[Step 8] Verifying data in shm...")
    read_back = shm.read_bytes(offset, data_size)
    read_md5 = md5(read_back)
    print("  Read-back MD5    : %s" % read_md5)
    if read_md5 != original_md5:
        print("[FAIL] READ data mismatch!")
        print("  Expected: %s, Got: %s" % (original_md5, read_md5))
        sys.exit(1)
    print("[OK] READ data verified!")

    # 9. Byte-by-byte comparison
    print("[Step 9] Byte-by-byte comparison...")
    if read_back == test_data:
        print("[OK] All %d bytes match!" % data_size)
    else:
        # Find first mismatch
        for i in range(len(test_data)):
            if read_back[i] != test_data[i]:
                print(
                    "[FAIL] First mismatch at byte %d: "
                    "expected 0x%02x, got 0x%02x" % (i, test_data[i], read_back[i])
                )
                sys.exit(1)

    # Cleanup
    print()
    print("[Step 10] Cleaning up...")
    client.close()
    shm.close()
    os.unlink(file_path)

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
