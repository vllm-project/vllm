import asyncio
import ctypes
import json
import os
import numpy as np
import pytest
import torch
from collections import deque
from dataclasses import dataclass
from unittest import mock
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.utils import tensor_memory_pool
from vllm.distributed.ec_transfer.utils.tensor_memory_pool import (
    TensorMemoryPool,
    InsufficientMemoryError)
from vllm.distributed.ec_transfer.ec_lookup_buffer.mooncake_store import (
    ECMooncakeStore,
    MooncakeStoreConfig,
    ECMooncakeTensorPoolMetadata)

DEFAULT_BUFFER_SIZE=1024

# Fake implementation of MooncakeDistributedStore for testing
class FakeMooncakeDistributedStore:
    def __init__(self):
        self.data = {}  # key -> bytes or tensors
        self.registered_buffers: set[tuple[int, int]] = set()
        self.remove_calls = []  # Track remove_by_regex calls

    def setup(self, local_hostname, metadata_server, global_segment_size, local_buffer_size, protocol, device_name, master_server_address):
        pass  # No-op for fake

    def close(self):
        pass  # No-op

    def batch_is_exist(self, keys):
        return [k in self.data for k in keys]

    def get_batch(self, keys):
        return [self.data.get(k) for k in keys]

    # List of bytes read for each operation (positive = success, negative = error)
    def batch_get_into(self, keys, addrs, sizes):
        results = []
        for key, addr, size in zip(keys, addrs, sizes):
            addr = addr if type(addr) is mock.Mock else addr
            if key in self.data and any(
                addr >= baddr and addr + size <= baddr + bsize
                for baddr, bsize in self.registered_buffers):
                # Simulate copy: put data into buffer
                buffer = (ctypes.c_char * len(self.data[key])).from_buffer(bytearray(self.data[key]))
                ctypes.memmove(addr, ctypes.addressof(buffer), size)
                results.append(size)
            else:
                results.append(-1)
        return results

    def put_batch(self, keys, values, replica_config):
        for key, value in zip(keys, values):
            self.data[key] = value

    def batch_put_from(self, keys, addrs, sizes, replica_config):
        for key, addr, size in zip(keys, addrs, sizes):
            if any(
                addr >= baddr and addr + size <= baddr + bsize
                for baddr, bsize in self.registered_buffers):
                data: bytes = ctypes.string_at(addr, size)
                self.data[key] = data[:size]

    def register_buffer(self, addr, size):
        print(type(addr), addr, size)
        self.registered_buffers.add((addr, size))

    def unregister_buffer(self, addr, size):
        print(type(addr), addr, size)
        self.registered_buffers.remove((addr, size))
        print("remove completed")

    def remove_by_regex(self, pattern):
        import regex as re
        regex = re.compile(pattern)
        count = 0
        for key in list(self.data.keys()):
            if regex.match(key):
                del self.data[key]
                count += 1
        self.remove_calls.append(pattern)
        return count

# Fake ReplicateConfig
@dataclass
class FakeReplicateConfig:
    replica_num: int = 1

@pytest.fixture
def mock_inner_mooncake_store(monkeypatch):
    fake_store = FakeMooncakeDistributedStore()
    monkeypatch.setattr(
        'mooncake.store.MooncakeDistributedStore', lambda: fake_store
    )
    monkeypatch.setattr(
        'mooncake.store.ReplicateConfig', FakeReplicateConfig
    )
    return fake_store

@pytest.fixture
def temp_config_file(tmp_path):
    config_path = tmp_path / "mooncake_config.json"
    config_data = {
        "local_hostname": "test_host",
        "metadata_server": "test_meta",
        "global_segment_size": DEFAULT_BUFFER_SIZE,
        "local_buffer_size": DEFAULT_BUFFER_SIZE,
        "protocol": "tcp",
        "device_name": "test_device",
        "master_server_address": "test_master",
        "storage_root_dir": "",
        "transfer_timeout": 5,
        "replica_num": 2,
        "fast_transfer": False,
        "fast_transfer_buffer_size": DEFAULT_BUFFER_SIZE,
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    return str(config_path)

@pytest.fixture
def vllm_config(temp_config_file):
    config = mock.Mock(spec=VllmConfig)
    config.ec_transfer_config = mock.Mock()
    config.ec_transfer_config.ec_connector_extra_config = {
        "ec_mooncake_config_file_path": temp_config_file
    }
    return config

@pytest.fixture
def ec_mooncake_store(vllm_config, mock_inner_mooncake_store):
    store = ECMooncakeStore(vllm_config)
    yield store
    try:
        store.close()
    except RuntimeError as e:
        if 'Event loop is closed' in str(e):
            # exception for test_close()
            return
        else:
            raise

def test_init(vllm_config, mock_inner_mooncake_store):
    # Mock methods
    mock_inner_mooncake_store.setup = mock.MagicMock(name="setup")

    store = ECMooncakeStore(vllm_config)
    assert store.config.local_hostname == "test_host"
    assert store.config.replica_num == 2
    assert not store.config.fast_transfer
    mock_inner_mooncake_store.setup.assert_called_once()
    store.close()

def test_init_with_fast_transfer(monkeypatch, vllm_config, mock_inner_mooncake_store):
    # Mock methods
    mock_inner_mooncake_store.register_buffer = mock.MagicMock(name="register_buffer")
    mock_inner_mooncake_store.unregister_buffer = mock.MagicMock(name="unregister_buffer")
    tensorpool = mock.Mock(spec=TensorMemoryPool)
    monkeypatch.setattr(
        'vllm.distributed.ec_transfer.utils.tensor_memory_pool.TensorMemoryPool',
        lambda max_block_size: tensorpool
    )

    # Modify config to enable fast_transfer
    with open(vllm_config.ec_transfer_config.ec_connector_extra_config["ec_mooncake_config_file_path"], 'r+') as f:
        data = json.load(f)
        data["fast_transfer"] = True
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    store = ECMooncakeStore(vllm_config)
    assert store.config.fast_transfer
    mock_inner_mooncake_store.register_buffer.assert_called_with(
        mock.ANY, DEFAULT_BUFFER_SIZE
    )
    store.close()
    mock_inner_mooncake_store.unregister_buffer.assert_called_with(
        mock.ANY, DEFAULT_BUFFER_SIZE
    )
    # Make sure it registers & unregisters the same buffer
    mock_inner_mooncake_store.register_buffer.call_args == mock_inner_mooncake_store.unregister_buffer.call_args

def test_batch_exists(ec_mooncake_store, mock_inner_mooncake_store):
    mock_inner_mooncake_store.data = {"key1": b'data1', "key2": b'data2'}
    exists = ec_mooncake_store.batch_exists(["key1", "key3", "key2"])
    assert exists == [True, False, True]
    exists = ec_mooncake_store.batch_exists([])
    assert exists == []

def test_batch_get_non_fast(ec_mooncake_store, mock_inner_mooncake_store):
    # Prepare serialized data
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    meta = {
        "shape": list(tensor.shape),
        "original_dtype": str(tensor.dtype),
        "serialized_dtype": "float32"
    }

    meta_bytes = json.dumps(meta).encode("utf-8")
    len_bytes = len(meta_bytes).to_bytes(4, "big")
    data_bytes = tensor.cpu().numpy().tobytes()
    serialized = len_bytes + meta_bytes + data_bytes

    mock_inner_mooncake_store.data = {"key1": serialized, "key2": None}

    results = ec_mooncake_store.batch_get(["key1", "key2"])
    assert torch.equal(results[0].cpu(), tensor.cpu())
    assert results[1] is None

def test_batch_put_non_fast(ec_mooncake_store, mock_inner_mooncake_store):
    tensors = [
        torch.randn((2, 2), dtype=torch.bfloat16, device='cuda'),
        torch.randn((1, 4), dtype=torch.float32),
        torch.tensor([[1, 2]], dtype=torch.int32, device='cuda')]
    keys = ["key1", "key2", "key3"]

    ec_mooncake_store.batch_put(keys, tensors)
    ec_mooncake_store.wait_for_put()

    assert "key1" in mock_inner_mooncake_store.data
    assert "key2" in mock_inner_mooncake_store.data
    assert "key3" in mock_inner_mooncake_store.data

    # Verify deserialization
    stored1 = mock_inner_mooncake_store.data["key1"]
    len_meta = int.from_bytes(stored1[:4], "big")
    meta = json.loads(stored1[4:4 + len_meta].decode("utf-8"))
    assert meta["original_dtype"] == "torch.bfloat16"

    results = ec_mooncake_store.batch_get(["key1", "key2", "key3"])

    assert torch.equal(results[0].cpu(), tensors[0].cpu())
    assert torch.equal(results[1].cpu(), tensors[1].cpu())
    assert torch.equal(results[2].cpu(), tensors[2].cpu())

def test_batch_get_zero_copy(monkeypatch, vllm_config, mock_inner_mooncake_store):
    # Enable fast_transfer
    with open(vllm_config.ec_transfer_config.ec_connector_extra_config["ec_mooncake_config_file_path"], 'r+') as f:
        data = json.load(f)
        data["fast_transfer"] = True
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    store = ECMooncakeStore(vllm_config)

    # Prepare metadata
    meta = {"shape": [2, 2], "dtype": "torch.float32"}
    meta_bytes = json.dumps(meta).encode("utf-8")
    value1 = torch.randn((2, 2))
    value1_bytes = value1.numpy().tobytes()
    mock_inner_mooncake_store.data = {
        "key1_metadata": meta_bytes,
        "key1": value1_bytes,
    }

    results = store.batch_get(["key1", "key2"])
    assert torch.equal(value1.cpu(), results[0].cpu())
    assert results[1] is None

    store.close()

def test_batch_put_zero_copy(monkeypatch, vllm_config, mock_inner_mooncake_store):
    # Enable fast_transfer
    with open(vllm_config.ec_transfer_config.ec_connector_extra_config["ec_mooncake_config_file_path"], 'r+') as f:
        data = json.load(f)
        data["fast_transfer"] = True
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    store = ECMooncakeStore(vllm_config)

    tensors = [
        torch.tensor([[1, 2]], dtype=torch.int32, device='cuda'),
        torch.tensor([[3.0, 4.0]], dtype=torch.float32, device='cuda')]
    keys = ["key1", "key2"]

    store.batch_put(keys, tensors)
    store.wait_for_put()

    assert mock_inner_mooncake_store.data.get("key1") == tensors[0].cpu().numpy().tobytes()
    assert mock_inner_mooncake_store.data.get("key2") == tensors[1].cpu().numpy().tobytes()
    assert store.metadata_key("key1") in mock_inner_mooncake_store.data
    assert store.metadata_key("key2") in mock_inner_mooncake_store.data

    store.close()

def test_pool_eviction(monkeypatch, vllm_config, mock_inner_mooncake_store):
    # Enable fast_transfer
    with open(vllm_config.ec_transfer_config.ec_connector_extra_config["ec_mooncake_config_file_path"], 'r+') as f:
        data = json.load(f)
        data["fast_transfer"] = True
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    store = ECMooncakeStore(vllm_config)
    orig_tensor_pool = store.tensor_pool
    store.tensor_pool = mock.MagicMock(wraps=store.tensor_pool)

    evict_tensor = torch.randn((4, 4), dtype=torch.float32, device='cuda')
    store.batch_put(["evict_key"], [evict_tensor])
    store.wait_for_put()

    # Trigger allocation with eviction, 16 * 16 * 4 = 1024
    new_tensor = torch.randn((16, 16), dtype=torch.float32, device='cuda')
    store.batch_put(["new_key"], [new_tensor])
    store.wait_for_put()

    store.tensor_pool.free.assert_called_once()
    assert mock_inner_mooncake_store.data.get("new_key") == new_tensor.cpu().numpy().tobytes()
    assert "evict_key" not in mock_inner_mooncake_store.data

    store.tensor_pool = orig_tensor_pool
    store.close()

def test_close(ec_mooncake_store, mock_inner_mooncake_store):
    mock_inner_mooncake_store.close = mock.MagicMock(name="close")
    ec_mooncake_store.close()
    mock_inner_mooncake_store.close.assert_called_once()
