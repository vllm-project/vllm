# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from unittest.mock import patch

from vllm.renderers.paged_shm.storage import PagedShmStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_storage(size=1024, block_size=256, pin=False):
    return PagedShmStorage(size=size, block_size=block_size, pin=pin)


def _cleanup(storage):
    storage.close()
    try:
        storage.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Basic initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_create_new_shm(self):
        """Creating without a name allocates fresh shared memory."""
        store = _create_storage()
        assert store._created is True
        assert store.name is not None
        assert store.size == 1024
        assert store.n_block == 4
        assert not store.is_pinned
        _cleanup(store)

    def test_attach_to_existing_shm(self):
        """Attaching by name re-uses the same segment."""
        store1 = _create_storage()
        name = store1.name
        store2 = PagedShmStorage(size=1024, block_size=256, name=name, pin=False)
        assert store2._created is False
        assert store2.name == name
        store2.close()  # second instance must not unlink
        _cleanup(store1)

    def test_attach_nonexistent_name_raises(self):
        with pytest.raises(FileNotFoundError):
            PagedShmStorage(size=256, block_size=256, name="no_such_shm")


# ---------------------------------------------------------------------------
# Size / block calculations
# ---------------------------------------------------------------------------

class TestSizing:
    def test_exact_multiple(self):
        store = _create_storage(size=1024, block_size=256)
        assert store.n_block == 4
        assert store.size == 1024
        _cleanup(store)

    def test_not_multiple_silent_truncation(self):
        """size is silently truncated to block boundary."""
        store = _create_storage(size=1000, block_size=256)
        assert store.size == 768
        assert store.n_block == 3
        _cleanup(store)


# ---------------------------------------------------------------------------
# Write lifecycle (CPU data)
# ---------------------------------------------------------------------------

class TestWrite:
    def test_write_bytes(self):
        store = _create_storage(block_size=64)
        data = b"hello" * 10  # 50 bytes
        store.write(data, blocks=[0])
        # Verify the first 50 bytes of block 0
        np.testing.assert_array_equal(
            store._shm_np[0][:50],
            np.frombuffer(data, dtype=np.uint8)
        )
        _cleanup(store)

    def test_write_numpy_array(self):
        store = _create_storage(size=512, block_size=256)
        data = np.arange(300, dtype=np.uint8)
        store.write(data, blocks=[0, 1])
        assert np.array_equal(store._shm_np[0], data[0:256])
        assert np.array_equal(store._shm_np[1][:44], data[256:300])
        _cleanup(store)

    def test_write_cpu_tensor(self):
        store = _create_storage(block_size=256)
        data = torch.randint(0, 256, (200,), dtype=torch.uint8)
        store.write(data, blocks=[3])
        assert torch.equal(store._shm_tensor[3][:200], data)
        _cleanup(store)

    def test_write_too_large_raises(self):
        store = _create_storage(size=256, block_size=256)
        with pytest.raises(ValueError, match="Data too large"):
            store.write(np.zeros(257, dtype=np.uint8), blocks=[0])
        _cleanup(store)


# ---------------------------------------------------------------------------
# Read lifecycle
# ---------------------------------------------------------------------------

class TestRead:
    def test_read_to_numpy(self):
        store = _create_storage(block_size=64)
        expected = np.arange(100, dtype=np.uint8)
        store._shm_np[0][:64] = expected[0:64]
        store._shm_np[1][:36] = expected[64:100]

        result = store.read_to_numpy(100, blocks=[0, 1])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)
        _cleanup(store)

    def test_read_to_tensor_cpu(self):
        store = _create_storage(block_size=256)
        data = torch.ones(256, dtype=torch.uint8)
        store._shm_tensor[0].copy_(data)

        result = store.read_to_tensor(256, blocks=[0], device="cpu")
        assert result.device.type == "cpu"
        assert torch.equal(result, data)
        _cleanup(store)

    def test_read_exceeds_capacity_raises(self):
        store = _create_storage(size=256, block_size=256)
        with pytest.raises(ValueError, match="too large"):
            store.read_to_numpy(500, blocks=[0])
        _cleanup(store)


# ---------------------------------------------------------------------------
# Iterator helpers
# ---------------------------------------------------------------------------

class TestIterators:
    def test_iterator_numpy(self):
        store = _create_storage(size=512, block_size=256)
        data = np.arange(512, dtype=np.uint8)
        store._shm_np[0] = data[0:256]
        store._shm_np[1] = data[256:512]

        chunks = list(store.get_iterator_numpy(512, blocks=[0, 1])())
        assert len(chunks) == 2

        np.testing.assert_array_equal(chunks[0][0], data[0:256])
        assert chunks[0][1] == 256
        np.testing.assert_array_equal(chunks[1][0], data[256:512])
        _cleanup(store)

    def test_iterator_with_partial_last_block(self):
        store = _create_storage(block_size=256)
        data = np.arange(300, dtype=np.uint8)
        store._shm_np[0] = data[0:256]
        store._shm_np[1][:44] = data[256:300]

        chunks = list(store.get_iterator_numpy(300, blocks=[0, 1])())
        assert len(chunks) == 2
        assert chunks[1][1] == 44
        np.testing.assert_array_equal(chunks[1][0][:44], data[256:300])
        _cleanup(store)

    def test_iterator_tensor(self):
        store = _create_storage(block_size=128)
        t = torch.arange(256, dtype=torch.uint8)
        store._shm_tensor[1] = t[0:128]
        store._shm_tensor[2] = t[128:256]

        chunks = list(store.get_iterator_tensor(256, blocks=[1, 2])())
        assert torch.equal(chunks[0][0], t[0:128])
        assert torch.equal(chunks[1][0], t[128:256])
        _cleanup(store)


# ---------------------------------------------------------------------------
# GPU direct transfers (pinned memory)
# ---------------------------------------------------------------------------

class TestDeviceTransfers:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_write_from_device(self):
        store = _create_storage(size=1024, block_size=256, pin=True)
        gpu_data = torch.arange(512, dtype=torch.uint8, device="cuda")
        store.write_from_device(gpu_data, blocks=[0, 1])

        # Verify CPU side
        assert torch.equal(store._shm_tensor[0], gpu_data[0:256].cpu())
        assert torch.equal(store._shm_tensor[1][:256], gpu_data[256:512].cpu())
        _cleanup(store)

    def test_write_from_device_not_pinned_raises(self):
        store = _create_storage(pin=False)
        dummy = torch.zeros(1, dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError, match="not pinned"):
            store.write_from_device(dummy, blocks=[0])
        _cleanup(store)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_read_to_device(self):
        store = _create_storage(size=512, block_size=256, pin=True)

        cpu_data = torch.arange(512, dtype=torch.uint8)
        store._shm_tensor[0].copy_(cpu_data[0:256])
        store._shm_tensor[1][:256] = cpu_data[256:512]

        result = store.read_to_device(512, blocks=[0, 1], device="cuda")
        assert result.device.type == "cuda"
        assert torch.equal(result.cpu(), cpu_data)
        _cleanup(store)

    def test_read_to_device_not_pinned_raises(self):
        store = _create_storage(pin=False)
        with pytest.raises(RuntimeError, match="not pinned"):
            store.read_to_device(256, blocks=[0], device="cuda")
        _cleanup(store)

    def test_write_from_device_calls_swap_ops(self):
        """Verify that the custom op is invoked with correct address lists."""
        store = _create_storage(pin=False)
        store.is_pinned = True  # fake the pin
        data = torch.zeros(512, dtype=torch.uint8, device="cuda")
        with patch("vllm._custom_ops.swap_blocks_batch") as mock_op:
            store.write_from_device(data, blocks=[0, 2])
            mock_op.assert_called_once()
            # The sizes argument should reflect 256 + 256
            sizes_tensor = mock_op.call_args[0][2]
            assert torch.equal(sizes_tensor, torch.tensor([256, 256], dtype=torch.int64))
        _cleanup(store)


# ---------------------------------------------------------------------------
# Cleanup & resource management
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_explicit_unlink_removes_name(self):
        store = _create_storage()
        name = store.name
        _cleanup(store)
        with pytest.raises(FileNotFoundError):
            PagedShmStorage(size=1024, block_size=256, name=name)

    def test_attached_instance_does_not_unlink(self):
        store1 = _create_storage()
        name = store1.name
        store2 = PagedShmStorage(size=1024, block_size=256, name=name, pin=False)
        store2.close()  # only close, not unlink
        # name must still be valid
        store3 = PagedShmStorage(size=1024, block_size=256, name=name, pin=False)
        store3.close()
        _cleanup(store1)

    def test_del_triggers_unlink_for_creator(self):
        store = _create_storage()
        name = store.name
        del store
        # After deletion the segment should be unreachable by name
        with pytest.raises(FileNotFoundError):
            PagedShmStorage(size=1024, block_size=256, name=name)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_write_zero_bytes(self):
        store = _create_storage(block_size=64)
        store.write(b"", blocks=[0])
        # Should succeed with no effect
        _cleanup(store)

    def test_read_zero_bytes(self):
        store = _create_storage(block_size=64)
        result = store.read_to_numpy(0, blocks=[0])
        assert isinstance(result, np.ndarray)
        assert result.size == 0
        _cleanup(store)

    def test_invalid_block_index_propagates(self):
        store = _create_storage(block_size=256)
        # Accessing an out-of-range block via internal array
        with pytest.raises(IndexError):
            _ = store._shm_np[100]
        _cleanup(store)