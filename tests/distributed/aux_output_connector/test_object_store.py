# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import regex as re

from vllm.distributed import mooncake_store
from vllm.distributed.aux_output_connector.mooncake import MooncakeAuxObjectStore
from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectStoreError,
    LocalAuxObjectStore,
)


class FakeMooncake:
    def __init__(self, fail_keys=(), capacity=None):
        self.objects = {}
        self.fail_keys = set(fail_keys)
        self.capacity = capacity
        self.configs = []

    def put_batch(self, keys, values, config):
        self.configs.append(config)
        rc = 0
        for key, value in zip(keys, values):
            if key in self.objects:
                continue
            if key in self.fail_keys or (
                self.capacity is not None and len(self.objects) >= self.capacity
            ):
                rc = -200
            else:
                self.objects[key] = bytes(value)
        return rc

    def get(self, key):
        return self.objects.get(key, b"")

    def batch_is_exist(self, keys):
        return [1 if key in self.objects else 0 for key in keys]

    def batch_remove(self, keys, force):
        assert force
        return [0 if self.objects.pop(key, None) is not None else -704 for key in keys]

    def remove_by_regex(self, pattern, force):
        assert force
        keys = [key for key in self.objects if re.match(pattern, key)]
        for key in keys:
            del self.objects[key]
        return len(keys)

    def close(self):
        pass


def _mooncake(**kwargs):
    fake = FakeMooncake(**kwargs)
    return MooncakeAuxObjectStore(fake, SimpleNamespace(with_hard_pin=True)), fake


@pytest.fixture(params=["local", "mooncake"])
def store(request):
    if request.param == "local":
        return LocalAuxObjectStore()
    return _mooncake()[0]


class TestAuxObjectStore:
    def test_first_write_wins(self, store):
        store.put_batch(["a", "b"], [b"1", b"2"])
        store.put_batch(["b", "c"], [b"x", b"3"])
        assert [store.get(k) for k in "abcd"] == [b"1", b"2", b"3", None]
        assert store.exists(["a", "d", "c"]) == [True, False, True]

    def test_remove_by_key(self, store):
        store.put_batch(["a", "b", "c"], [b"1", b"2", b"3"])
        store.remove(["a", "c", "missing"])
        assert store.exists(["a", "b", "c"]) == [False, True, False]
        store.put_batch(["a"], [b"new"])
        assert store.get("a") == b"new"

    def test_remove_prefix(self, store):
        keys = ["job/s/a", "job/s/b", "job/t/a", "job/s.x/a", "other/job/s/a"]
        store.put_batch(keys, [b"v"] * len(keys))
        assert store.remove_prefix("job/s/") == 2
        assert store.exists(keys) == [False, False, True, True, True]


class TestMooncakeAuxObjectStore:
    def test_create_pins_objects(self, monkeypatch):
        calls = []
        module = SimpleNamespace(
            MooncakeDistributedStore=FakeMooncake,
            ReplicateConfig=lambda: SimpleNamespace(with_hard_pin=False),
        )
        monkeypatch.setitem(sys.modules, "mooncake", SimpleNamespace(store=module))
        monkeypatch.setitem(sys.modules, "mooncake.store", module)
        monkeypatch.setattr(
            mooncake_store,
            "setup_mooncake_store",
            lambda store, config, hostname: calls.append((config, hostname)),
        )
        config = object()
        store = MooncakeAuxObjectStore.create(config, "host:1")
        store.put_batch(["a"], [b"1"])
        assert calls == [(config, "host:1")]
        assert [c.with_hard_pin for c in store._store.configs] == [True]

    @pytest.mark.parametrize(
        "kwargs,keys,missing",
        [
            ({"fail_keys": ["b"]}, ["a", "b", "c"], "'b'"),
            ({"capacity": 2}, ["a", "b", "c"], "'c'"),
        ],
        ids=["put-failed", "pool-full"],
    )
    def test_failed_write_raises(self, kwargs, keys, missing):
        store, _ = _mooncake(**kwargs)
        with pytest.raises(AuxObjectStoreError, match=f"1 of 3 .*first {missing}"):
            store.put_batch(keys, [b"v"] * len(keys))
        assert store.exists(keys).count(True) == 2
