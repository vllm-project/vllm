"""
Integration tests for PagedShmClient and PagedShmServer.

Covers basic write/read lifecycle, atomicity of write operations, pin/unpin,
delete, info queries, and protocol robustness.
"""

import multiprocessing as mp

import numpy as np
import pytest
import torch

from vllm.renderers.paged_shm.server import zmq_server
from vllm.renderers.paged_shm.client import PagedShmClient
from vllm.utils import random_uuid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def server_address():
    """
    Spawn a real PagedShmServer in a subprocess and return its IPC address.

    The server uses a 1 MB pool with 4 KB blocks. It is terminated and the
    shared memory is cleaned up after each test.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    stop_event = ctx.Event()

    proc = ctx.Process(
        target=zmq_server,
        args=(1024 * 1024, 4096, child_conn, stop_event),
    )
    proc.start()

    # Wait for the server to send back its address
    address = parent_conn.recv()
    parent_conn.close()

    yield address

    # Signal the server to stop and wait for clean exit
    stop_event.set()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join()


@pytest.fixture(scope="function")
def client(server_address):
    """
    Create a fresh PagedShmClient connected to the test server.

    Memory pinning is disabled by default; GPU tests create their own
    client with ``pin=True``.
    """
    c = PagedShmClient(address=server_address, pin=False)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _unique_uuid() -> str:
    """Return a short unique identifier for test items."""
    return f"test-{random_uuid()}"


# ---------------------------------------------------------------------------
# Basic write / read
# ---------------------------------------------------------------------------

class TestWriteRead:
    """Verify round‑trip correctness for various data types."""

    def test_write_read_bytes(self, client):
        uuid = _unique_uuid()
        data = b"Hello, shared memory!"
        client.write(uuid, data)
        result = client.read(uuid)
        assert result.tobytes() == data

    def test_write_read_numpy(self, client):
        uuid = _unique_uuid()
        data = np.random.bytes(1024)
        client.write(uuid, data)
        result = client.read(uuid)
        assert (result == np.frombuffer(data, dtype=np.uint8)).all()

    def test_write_read_torch_cpu(self, client):
        uuid = _unique_uuid()
        data = torch.arange(256, dtype=torch.uint8)
        client.write(uuid, data)
        result = client.read(uuid, device="cpu")
        assert torch.equal(result, data)


    @pytest.mark.skipif(True, reason="No GPU available")
    def test_write_read_torch_gpu(self, client, server_address):
        """GPU direct transfer requires a client with pinned memory."""
        gpu_client = PagedShmClient(server_address, pin=True)
        try:
            uuid = _unique_uuid()
            data = torch.arange(4096, dtype=torch.uint8, device="cuda")
            gpu_client.write(uuid, data)
            result = gpu_client.read(uuid, device="cuda")
            assert torch.equal(result.cpu(), data.cpu())
        finally:
            gpu_client.close()

    def test_read_nonexistent_item(self, client):
        with pytest.raises(RuntimeError, match="Server error"):
            client.read("nonexistent")


# ---------------------------------------------------------------------------
# Write atomicity – blocks are cleaned up on failure
# ---------------------------------------------------------------------------

class TestWriteAtomicity:
    """
    If a write fails after server‑side allocation, the client must
    automatically call ``delete`` to prevent block leaks.
    """

    def test_write_data_transfer_error_cleans_up(self, client, monkeypatch):
        uuid = _unique_uuid()
        data = b"test"

        # Simulate a failure during the local data transfer
        def failing_write(*args, **kwargs):
            raise RuntimeError("Simulated transfer error")
        monkeypatch.setattr(client._storage, "write", failing_write)

        with pytest.raises(RuntimeError, match="Simulated transfer error"):
            client.write(uuid, data)

        # The item should have been cleaned up – reading must fail
        with pytest.raises(RuntimeError, match="Server error"):
            client.read(uuid)


# ---------------------------------------------------------------------------
# Pin / Unpin
# ---------------------------------------------------------------------------

class TestPinUnpin:
    """Pin and unpin operations are passed through to the server."""

    def test_pin_unpin_cycle(self, client):
        uuid = _unique_uuid()
        client.write(uuid, b"data")
        # Both calls should succeed without errors
        client.pin(uuid)
        client.unpin(uuid)

    def test_pin_nonexistent(self, client):
        with pytest.raises(RuntimeError, match="Server error"):
            client.pin("nonexistent")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDelete:
    """Delete removes an item completely."""

    def test_delete_existing(self, client):
        uuid = _unique_uuid()
        client.write(uuid, b"data")
        client.delete(uuid)
        # Subsequent read must fail
        with pytest.raises(RuntimeError, match="Server error"):
            client.read(uuid)

    def test_delete_nonexistent(self, client):
        with pytest.raises(RuntimeError, match="Server error"):
            client.delete("nonexistent")


# ---------------------------------------------------------------------------
# Info & state queries
# ---------------------------------------------------------------------------

class TestInfoQueries:
    """Verify that metadata and state queries return expected structures."""

    def test_get_storage_info(self, client):
        info = client.get_storage_info()
        assert "name" in info
        assert "size" in info
        assert "block_size" in info
        assert info["size"] == 1024 * 1024
        assert info["block_size"] == 4096

    def test_get_manager_state(self, client):
        state = client.get_manager_state()
        assert "total_blocks" in state
        assert "free_blocks" in state

    def test_get_shm_name(self, client):
        name = client.get_shm_name()
        assert name.startswith("psm_")


# ---------------------------------------------------------------------------
# Edge cases & protocol robustness
# ---------------------------------------------------------------------------

class TestProtocolRobustness:
    """Check that the client handles malformed or unexpected server responses."""

    def test_unknown_command_raises(self, client):
        with pytest.raises(RuntimeError, match="Unknown command"):
            client._request(b"nonexistent_cmd")

    def test_malformed_uuid_payload(self, client):
        """
        The server expects raw UUID strings for single‑argument commands.
        Previously the client incorrectly JSON‑encoded them; this test
        confirms the fix is in place.
        """
        uuid = _unique_uuid()
        client.write(uuid, b"x")
        # open_read must accept the raw UUID and return block information
        item = client.open_read(uuid)
        assert "blocks" in item
        assert "size" in item
        client.close_read(uuid)

    def test_concurrent_readers(self, client):
        """Multiple concurrent read references are allowed."""
        uuid = _unique_uuid()
        client.write(uuid, b"shared")
        client.open_read(uuid)
        client.open_read(uuid)
        client.close_read(uuid)
        client.close_read(uuid)

    def test_write_and_delete_many(self, client):
        """Stress test: write, read, and delete several items, then check free blocks."""
        uuids = [_unique_uuid() for _ in range(10)]
        for u in uuids:
            client.write(u, np.zeros(1024, dtype=np.uint8))
        for u in uuids:
            result = client.read(u)
            assert result.nbytes == 1024
        for u in uuids:
            client.delete(u)

        # All blocks should be reclaimed
        state = client.get_manager_state()
        assert state["free_blocks"] == state["total_blocks"]