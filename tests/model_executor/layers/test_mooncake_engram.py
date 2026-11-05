# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from types import ModuleType

import pytest
import torch

from vllm.model_executor.layers.mooncake_engram import (
    MooncakeEngramStorage,
    _head_shard,
    _load_mooncake_config,
    close_mooncake_engram_storages,
    collect_mooncake_engram_storages,
    pack_mooncake_rows,
)


def _config(tmp_path, **updates):
    options = {
        "local_dir": str(tmp_path / "rows"),
        "shard_id": 1,
        "num_shards": 3,
        "master_addr": "http://master:5000",
        "session_id": "test-session",
        "max_lookup_rows": 128,
        "num_callers": 4,
        "rdma_device": "mlx5_{local_rank}",
    }
    options.update(updates)
    path = tmp_path / "engram.json"
    path.write_text(json.dumps(options), encoding="utf-8")
    return path


def test_mooncake_config_and_head_shard_validation(tmp_path):
    path = _config(tmp_path)
    assert _load_mooncake_config(str(path))["num_shards"] == 3
    assert _head_shard((2, 3, 5, 7, 11, 13, 17, 19, 23, 29), 3, 2) == (
        6,
        10,
        41,
        129,
    )
    with pytest.raises(ValueError, match="More Mooncake node shards"):
        _head_shard((2, 3), 3, 0)
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        _load_mooncake_config(str(_config(tmp_path, timeout=0)))


def test_pack_mooncake_rows_preserves_plane_bytes():
    values = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    assert pack_mooncake_rows(values).data_ptr() == values.data_ptr()
    scales = torch.tensor([[5], [6]], dtype=torch.uint8)
    packed = pack_mooncake_rows(values, scales)
    assert torch.equal(
        packed,
        torch.tensor([[1, 2, 5], [3, 4, 6]], dtype=torch.uint8),
    )


def test_initialize_publishes_only_node_owned_heads(tmp_path, monkeypatch):
    calls = {}

    class FakeStore:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def populate_shard_and_sync(self, local, **kwargs):
            calls["local"] = local
            calls["remote"] = kwargs

        def close(self):
            calls["closed"] = True

    package = ModuleType("mooncake")
    module = ModuleType("mooncake.engram")
    module.EngramStore = FakeStore
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.engram", module)
    monkeypatch.setattr(
        "vllm.distributed.get_world_group",
        lambda: type("WorldGroup", (), {"local_rank": 2})(),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)

    storage = MooncakeEngramStorage(
        str(_config(tmp_path)),
        namespace="model.layers.1.engram",
        head_sizes=(2, 3, 5, 7, 11, 13),
        row_bytes=2,
    )
    rows = torch.arange(storage.local_rows * 2, dtype=torch.uint8).reshape(-1, 2)
    storage.initialize(rows)

    assert tuple(calls["local"]) == ("head2", "head3")
    assert torch.equal(calls["local"]["head2"], rows[:5])
    assert torch.equal(calls["local"]["head3"], rows[5:])
    assert calls["remote"]["table_owners"] == {
        "head0": 0,
        "head1": 0,
        "head2": 1,
        "head3": 1,
        "head4": 2,
        "head5": 2,
    }
    assert calls["remote"]["caller_id"] == 2
    assert calls["remote"]["rdma_device"] == "mlx5_2"


def test_initialize_preserves_explicit_caller_id(tmp_path, monkeypatch):
    calls = {}

    class FakeStore:
        def __init__(self, **kwargs):
            pass

        def populate_shard_and_sync(self, local, **kwargs):
            calls.update(kwargs)

        def close(self):
            pass

    package = ModuleType("mooncake")
    module = ModuleType("mooncake.engram")
    module.EngramStore = FakeStore
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.engram", module)
    monkeypatch.setattr(
        "vllm.distributed.get_world_group",
        lambda: type("WorldGroup", (), {"local_rank": 2})(),
    )

    storage = MooncakeEngramStorage(
        str(_config(tmp_path, caller_id=1)),
        namespace="model.layers.1.engram",
        head_sizes=(2, 3, 5, 7, 11, 13),
        row_bytes=2,
    )
    rows = torch.zeros((storage.local_rows, 2), dtype=torch.uint8)
    storage.initialize(rows)

    assert calls["caller_id"] == 1
    assert calls["rdma_device"] == "mlx5_2"


def test_initialize_rejects_default_caller_outside_num_callers(tmp_path, monkeypatch):
    class FakeStore:
        populate_shard_and_sync = object()

    package = ModuleType("mooncake")
    module = ModuleType("mooncake.engram")
    module.EngramStore = FakeStore
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.engram", module)
    monkeypatch.setattr(
        "vllm.distributed.get_world_group",
        lambda: type("WorldGroup", (), {"local_rank": 2})(),
    )

    storage = MooncakeEngramStorage(
        str(_config(tmp_path, num_callers=2)),
        namespace="model.layers.1.engram",
        head_sizes=(2, 3, 5, 7, 11, 13),
        row_bytes=2,
    )
    rows = torch.zeros((storage.local_rows, 2), dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"\[0,2\), got 2"):
        storage.initialize(rows)


def test_collect_and_close_storages_in_namespace_order(tmp_path, monkeypatch):
    closed = []

    class FakeStore:
        def __init__(self, *, namespace, **kwargs):
            self.namespace = namespace

        def populate_shard_and_sync(self, local, **kwargs):
            pass

        def close(self):
            closed.append(self.namespace)

    package = ModuleType("mooncake")
    module = ModuleType("mooncake.engram")
    module.EngramStore = FakeStore
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.engram", module)
    monkeypatch.setattr(
        "vllm.distributed.get_world_group",
        lambda: type("WorldGroup", (), {"local_rank": 0})(),
    )

    def make_storage(namespace):
        storage = MooncakeEngramStorage(
            str(_config(tmp_path)),
            namespace=namespace,
            head_sizes=(2, 3, 5),
            row_bytes=1,
        )
        rows = torch.zeros((storage.local_rows, 1), dtype=torch.uint8)
        storage.initialize(rows)
        return storage

    storage_b = make_storage("model.layers.2.engram")
    storage_a = make_storage("model.layers.1.engram")
    model = torch.nn.Module()
    model.first = torch.nn.Module()
    model.first.mooncake_storage = storage_b
    model.second = torch.nn.Module()
    model.second.mooncake_storage = storage_a
    model.alias = torch.nn.Module()
    model.alias.mooncake_storage = storage_b

    storages = collect_mooncake_engram_storages(model)
    assert storages == (storage_a, storage_b)

    close_mooncake_engram_storages(storages)
    assert closed == ["model.layers.1.engram", "model.layers.2.engram"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mooncake_lookup_selects_compute_head_slice_cuda():
    calls = {}

    class FakeStore:
        @staticmethod
        def lookup(names, ids, output, **kwargs):
            calls.update(names=names, ids=ids.clone(), kwargs=kwargs)
            output.fill_(7)

    storage = MooncakeEngramStorage.__new__(MooncakeEngramStorage)
    storage._store = FakeStore()
    storage.names = ("head0", "head1", "head2", "head3")
    storage.offsets = (0, 11, 24, 41)
    storage.row_bytes = 3
    ids = torch.arange(8, dtype=torch.int64, device="cuda").reshape(2, 4)
    output = torch.zeros((2, 2, 3), dtype=torch.uint8, device="cuda")

    storage.lookup(ids, output, head_start=1)

    assert calls["names"] == ("head1", "head2")
    torch.testing.assert_close(calls["ids"], ids[:, 1:3])
    assert calls["kwargs"]["offsets"] == (11, 24)
    assert torch.all(output == 7)
