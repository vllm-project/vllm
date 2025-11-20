import json
import os
import pytest
import torch
from collections import deque
from dataclasses import dataclass
from unittest import mock
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.utils.tensor_memory_pool import TensorMemoryPool
from vllm.distributed.ec_transfer.ec_lookup_buffer.mooncake_store import ECMooncakeStore, MooncakeStoreConfig, ECMooncakeTensorPoolMetadata

# Fake implementation of MooncakeDistributedStore for testing
class FakeMooncakeDistributedStore:
    def __init__(self):
        self.data = {}  # key -> bytes
        self.buffers = {}  # addr -> {'size': int, 'data': bytes} for simulating zero-copy
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
            if key in self.data and addr in self.buffers and self.buffers[addr]['size'] >= size:
                # Simulate copy: put data into buffer
                self.buffers[addr]['data'] = self.data[key][:size]
                results.append(size)
            else:
                results.append(-1)
        return results

    def put_batch(self, keys, values, replica_config):
        for key, value in zip(keys, values):
            self.data[key] = value

    def batch_put_from(self, keys, addrs, sizes, replica_config):
        for key, addr, size in zip(keys, addrs, sizes):
            if addr in self.buffers and self.buffers[addr]['data'] is not None:
                # Simulate copy from buffer
                self.data[key] = self.buffers[addr]['data'][:size]
            else:
                self.data[key] = b'\x00' * size  # Dummy data if no buffer data

    def register_buffer(self, addr, size):
        self.buffers[addr] = {'size': size, 'data': None}

    def unregister_buffer(self, addr, size):
        if addr in self.buffers:
            del self.buffers[addr]

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
def mock_mooncake(monkeypatch):
    fake_store = FakeMooncakeDistributedStore()
    monkeypatch.setattr('mooncake_connector.mooncake.store.MooncakeDistributedStore', lambda: fake_store)
    monkeypatch.setattr('mooncake_connector.mooncake.store.ReplicateConfig', FakeReplicateConfig)
    return fake_store

@pytest.fixture
def temp_config_file(tmp_path):
    config_path = tmp_path / "mooncake_config.json"
    config_data = {
        "local_hostname": "test_host",
        "metadata_server": "test_meta",
        "global_segment_size": 1024,
        "local_buffer_size": 1024,
        "protocol": "tcp",
        "device_name": "test_device",
        "master_server_address": "test_master",
        "storage_root_dir": "",
        "transfer_timeout": 5,
        "replica_num": 2,
        "fast_transfer": False,
        "fast_transfer_buffer_size": 1024,
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
def mooncake_store(vllm_config, mock_mooncake):
    store = ECMooncakeStore(vllm_config)
    yield store
    store.close()

def test_init(vllm_config, mock_mooncake):
    store = ECMooncakeStore(vllm_config)
    assert store.config.local_hostname == "test_host"
    assert store.config.replica_num == 2
    assert not store.config.fast_transfer
    mock_mooncake.setup.assert_called_once()
    store.close()

def test_init_with_fast_transfer(vllm_config, mock_mooncake, monkeypatch):
    # Modify config to enable fast_transfer
    with open(vllm_config.ec_transfer_config.ec_connector_extra_config["ec_mooncake_config_file_path"], 'r+') as f:
        data = json.load(f)
        data["fast_transfer"] = True
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    mock_pool = mock.Mock(spec=TensorMemoryPool)
    mock_pool.base_address = 1234
    monkeypatch.setattr('mooncake_connector.TensorMemoryPool', lambda max_block_size: mock_pool)

    store = ECMooncakeStore(vllm_config)
    assert store.config.fast_transfer
    mock_mooncake.register_buffer.assert_called_with(1234, 1024)
    store.close()
    mock_mooncake.unregister_buffer.assert_called_with(1234, 1024)

def test_batch_exists(mooncake_store, mock_mooncake):
    mock_mooncake.data = {"key1": b'data1', "key2": b'data2'}
    exists = mooncake_store.batch_exists(["key1", "key3", "key2"])
    assert exists == [True, False, True]

def test_batch_get_non_fast(mooncake_store, mock_mooncake):
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

    mock_mooncake.data = {"key1": serialized, "key2": None}

    results = mooncake_store.batch_get(["key1", "key2"])
    assert torch.equal(results[0].cpu(), tensor.cpu())
    assert results[1] is None

def test_batch_put_non_fast(mooncake_store, mock_mooncake):
    tensors = [torch.tensor([1, 2], dtype=torch.int32), torch.tensor([3.0, 4.0], dtype=torch.float32)]
    keys = ["key1", "key2"]

    mooncake_store.batch_put(keys, tensors)
    mooncake_store.wait_for_put()

    assert "key1" in mock_mooncake.data
    assert "key2" in mock_mooncake.data

    # Verify deserialization
    stored1 = mock_mooncake.data["key1"]
    len_meta = int.from_bytes(stored1[:4], "big")
    meta = json.loads(stored1[4:4 + len_meta].decode("utf-8"))
    assert meta["original_dtype"] == "torch.int32"

def test_batch_get_zero_copy(monkeypatch, vllm_config, mock_mooncake):
    # Enable fast_transfer
    with open(vllm_config.ec_transfer_config.ec_connector_extra_config["ec_mooncake_config_file_path"], 'r+') as f:
        data = json.load(f)
        data["fast_transfer"] = True
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    mock_pool = mock.Mock(spec=TensorMemoryPool)
    mock_pool.base_address = 1234
    mock_pool.allocate.side_effect = [1000, 2000]  # Sample addrs
    mock_pool.free.side_effect = lambda addr: None
    mock_pool.load_tensor.side_effect = lambda addr, dtype, shape, device: torch.zeros(shape, dtype=dtype)  # Dummy load

    monkeypatch.setattr('mooncake_connector.TensorMemoryPool', lambda max_block_size: mock_pool)

    store = ECMooncakeStore(vllm_config)

    # Prepare metadata
    meta = {"shape": [2, 2], "dtype": "torch.float32"}
    meta_bytes = json.dumps(meta).encode("utf-8")
    mock_mooncake.data = {
        "key1_metadata": meta_bytes,
        "key2_metadata": None,
    }
    mock_mooncake.batch_get_into.side_effect = lambda keys, addrs, sizes: [16, 0]  # 4 elements * 4 bytes

    results = store.batch_get(["key1", "key2"])
    assert results[0].shape == (2, 2)
    assert results[1] is None

    mock_pool.allocate.assert_called()
    mock_pool.free.assert_called()
    store.close()

def test_batch_put_zero_copy(monkeypatch, vllm_config, mock_mooncake):
    # Enable fast_transfer
    with open(vllm_config.ec_transfer_config.ec_connector_extra_config["ec_mooncake_config_file_path"], 'r+') as f:
        data = json.load(f)
        data["fast_transfer"] = True
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    mock_pool = mock.Mock(spec=TensorMemoryPool)
    mock_pool.store_tensor.side_effect = [1000, 2000]  # Sample addrs
    mock_pool.free.side_effect = lambda addr: None

    monkeypatch.setattr('mooncake_connector.TensorMemoryPool', lambda max_block_size: mock_pool)
    monkeypatch.setattr('mooncake_connector.deque', lambda: deque())  # Mock fifo queue

    store = ECMooncakeStore(vllm_config)

    tensors = [torch.tensor([[1, 2]], dtype=torch.int32), torch.tensor([[3.0, 4.0]], dtype=torch.float32)]
    keys = ["key1", "key2"]

    store.batch_put(keys, tensors)
    store.wait_for_put()

    mock_pool.store_tensor.assert_called()
    mock_mooncake.put_batch.assert_called()  # For metadata
    mock_mooncake.batch_put_from.assert_called()

    store.close()

def test_pool_eviction(monkeypatch, vllm_config, mock_mooncake):
    # Enable fast_transfer
    with open(vllm_config.ec_transfer_config.ec_connector_extra_config["ec_mooncake_config_file_path"], 'r+') as f:
        data = json.load(f)
        data["fast_transfer"] = True
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    mock_pool = mock.Mock(spec=TensorMemoryPool)
    mock_pool.allocate.side_effect = [InsufficientMemoryError, 1000]  # Force eviction on first try
    mock_pool.free.side_effect = lambda addr: None
    mock_pool.store_tensor.side_effect = lambda tensor: 1000

    monkeypatch.setattr('mooncake_connector.TensorMemoryPool', lambda max_block_size: mock_pool)

    store = ECMooncakeStore(vllm_config)
    store.fifo_pool_queue = deque([ECMooncakeTensorPoolMetadata("evict_key", 500)])

    # Trigger allocation with eviction
    addr = store._pool_allocate(100)
    assert addr == 1000
    mock_pool.free.assert_called_with(500)
    mock_mooncake.remove_by_regex.assert_called()
    store.close()

def test_close(mooncake_store, mock_mooncake):
    mooncake_store.close()
    mock_mooncake.close.assert_called_once()