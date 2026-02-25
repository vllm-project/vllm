# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for StoreReusedOffloadingManager.

Uses importlib to load reuse_manager.py and reuse_tracker.py directly,
with local stubs replacing vllm base classes, so tests run on Python 3.8.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from collections.abc import Iterable
from typing import Any, Optional

import pytest

# ---------------------------------------------------------------------------
# Stub out the vllm.v1.kv_offload.abstract module so importlib can load
# reuse_manager.py and reuse_tracker.py without a full vllm install.
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    def _ga(attr: str) -> Any:
        return type(attr, (), {})()
    m.__getattr__ = _ga  # type: ignore[assignment]
    return m


for _n in [
    "vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_utils",
    "vllm.logger",
]:
    if _n not in sys.modules:
        sys.modules[_n] = _stub(_n)

# Patch logger
sys.modules["vllm.logger"].init_logger = (  # type: ignore[assignment]
    lambda *a, **kw: __import__("logging").getLogger("test")
)
# BlockHash = plain int for tests
sys.modules["vllm.v1.core.kv_cache_utils"].BlockHash = int  # type: ignore[assignment]


# ---- Local abstract base classes (replicate just enough of abstract.py) ----

class LoadStoreSpec:
    @staticmethod
    def medium() -> str:
        return "STUB"


class PrepareStoreOutput:
    def __init__(self, block_hashes_to_store, store_spec, block_hashes_evicted):
        self.block_hashes_to_store = block_hashes_to_store
        self.store_spec = store_spec
        self.block_hashes_evicted = block_hashes_evicted


class OffloadingManager:
    def lookup(self, block_hashes):
        return 0
    def prepare_load(self, block_hashes):
        return LoadStoreSpec()
    def touch(self, block_hashes):
        pass
    def complete_load(self, block_hashes):
        pass
    def prepare_store(self, block_hashes):
        return None
    def complete_store(self, block_hashes, success=True):
        pass
    def take_events(self):
        return ()


class OffloadingEvent:
    pass


# Register the stubs so the imports in reuse_manager.py resolve correctly.
_abstract_stub = types.ModuleType("vllm.v1.kv_offload.abstract")
_abstract_stub.LoadStoreSpec = LoadStoreSpec  # type: ignore[assignment]
_abstract_stub.PrepareStoreOutput = PrepareStoreOutput  # type: ignore[assignment]
_abstract_stub.OffloadingManager = OffloadingManager  # type: ignore[assignment]
_abstract_stub.OffloadingEvent = OffloadingEvent  # type: ignore[assignment]
sys.modules["vllm.v1.kv_offload"] = _stub("vllm.v1.kv_offload")
sys.modules["vllm.v1.kv_offload.abstract"] = _abstract_stub


def _load(rel: str, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(_ROOT / rel))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_rt = _load("vllm/v1/kv_offload/reuse_tracker.py", "vllm.v1.kv_offload.reuse_tracker")
sys.modules["vllm.v1.kv_offload.reuse_tracker"] = _rt

_rm = _load("vllm/v1/kv_offload/reuse_manager.py", "vllm.v1.kv_offload.reuse_manager")
StoreReusedOffloadingManager = _rm.StoreReusedOffloadingManager


# ---------------------------------------------------------------------------
# FakeManager — a simple in-memory OffloadingManager stub
# ---------------------------------------------------------------------------

class FakeManager(OffloadingManager):
    def __init__(self):
        self.stored: list = []
        self.loaded: list = []
        self.completed_stores: list = []

    def prepare_store(self, block_hashes: Iterable[Any]):
        hashes = list(block_hashes)
        self.stored.append(hashes)
        return PrepareStoreOutput(
            block_hashes_to_store=list(hashes),
            store_spec=LoadStoreSpec(),
            block_hashes_evicted=[],
        )

    def prepare_load(self, block_hashes: Iterable[Any]) -> LoadStoreSpec:
        hashes = list(block_hashes)
        self.loaded.append(hashes)
        return LoadStoreSpec()

    def complete_store(self, block_hashes: Iterable[Any], success: bool = True) -> None:
        self.completed_stores.append((list(block_hashes), success))


# ---------------------------------------------------------------------------
# Tests: threshold filtering
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_first_occurrence_filtered_out(self):
        mgr = StoreReusedOffloadingManager(FakeManager(), store_threshold=2)
        out = mgr.prepare_store([1, 2, 3])
        assert out is not None
        assert out.block_hashes_to_store == []

    def test_second_occurrence_passes_through(self):
        mgr = StoreReusedOffloadingManager(FakeManager(), store_threshold=2)
        mgr.prepare_store([10, 20])
        out = mgr.prepare_store([10, 20])
        assert out is not None
        assert sorted(out.block_hashes_to_store) == [10, 20]

    def test_threshold_one_passes_all(self):
        mgr = StoreReusedOffloadingManager(FakeManager(), store_threshold=1)
        out = mgr.prepare_store([7, 8, 9])
        assert out is not None
        assert sorted(out.block_hashes_to_store) == [7, 8, 9]

    def test_threshold_three(self):
        mgr = StoreReusedOffloadingManager(FakeManager(), store_threshold=3)
        mgr.prepare_store([42])
        mgr.prepare_store([42])
        out = mgr.prepare_store([42])
        assert out is not None
        assert out.block_hashes_to_store == [42]

    def test_mixed_first_and_repeated(self):
        mgr = StoreReusedOffloadingManager(FakeManager(), store_threshold=2)
        mgr.prepare_store([1, 2])
        out = mgr.prepare_store([1, 3])  # 1 is second sight, 3 is first
        assert out is not None
        assert out.block_hashes_to_store == [1]


# ---------------------------------------------------------------------------
# Tests: delegation to backing manager
# ---------------------------------------------------------------------------

class TestDelegation:
    def test_lookup_delegated(self):
        fake = FakeManager()
        mgr = StoreReusedOffloadingManager(fake, store_threshold=2)
        assert mgr.lookup([1, 2, 3]) == 0

    def test_prepare_load_delegated(self):
        fake = FakeManager()
        mgr = StoreReusedOffloadingManager(fake, store_threshold=2)
        mgr.prepare_load([5, 6])
        assert fake.loaded == [[5, 6]]

    def test_complete_store_delegated(self):
        fake = FakeManager()
        mgr = StoreReusedOffloadingManager(fake, store_threshold=2)
        mgr.complete_store([99], success=True)
        assert fake.completed_stores == [([99], True)]

    def test_backing_prepare_store_always_called(self):
        fake = FakeManager()
        mgr = StoreReusedOffloadingManager(fake, store_threshold=2)
        mgr.prepare_store([1, 2])
        assert len(fake.stored) == 1 and fake.stored[0] == [1, 2]

    def test_none_from_backing_propagated(self):
        class NoneManager(FakeManager):
            def prepare_store(self, block_hashes: Iterable[Any]) -> None:
                return None

        mgr = StoreReusedOffloadingManager(NoneManager(), store_threshold=2)
        assert mgr.prepare_store([1, 2]) is None


# ---------------------------------------------------------------------------
# Tests: LRU eviction boundary
# ---------------------------------------------------------------------------

class TestEviction:
    def test_evicted_hash_resets(self):
        mgr = StoreReusedOffloadingManager(
            FakeManager(), store_threshold=2, max_tracker_size=1
        )
        mgr.prepare_store([100])  # 100 seen once
        mgr.prepare_store([200])  # 200 inserted, 100 evicted
        out = mgr.prepare_store([100])  # 100 re-inserted, count=1 again
        assert out is not None
        assert out.block_hashes_to_store == []
