# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for BlockReuseTracker and FilteredOffloadingManager.

The production classes are loaded directly from
vllm/v1/kv_offload/reuse_manager.py via importlib so tests always run
against the real source without needing a full vllm install (no CUDA,
no torch required for the pure-Python logic).
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# TYPE_CHECKING-only stubs so mypy can reason about the dynamically-loaded
# classes without importing the full vllm tree.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from collections import OrderedDict as _OrderedDict

    class BlockReuseTracker:  # pragma: no cover
        """Mypy-only stub for reuse_manager.BlockReuseTracker."""

        max_size: int
        store_threshold: int
        counts: _OrderedDict[Any, int]

        def __init__(self, max_size: int = ..., store_threshold: int = ...) -> None: ...

        def record(self, block_hash: Any) -> None: ...

        def check(self, block_hash: Any) -> bool: ...

    class FilteredOffloadingManager:  # pragma: no cover
        """Mypy-only stub for reuse_manager.FilteredOffloadingManager."""

        _backing: Any
        _tracker: BlockReuseTracker

        def __init__(
            self,
            backing: Any,
            store_threshold: int = ...,
            max_tracker_size: int = ...,
        ) -> None: ...

        def lookup(self, block_hashes: Iterable[Any]) -> int | None: ...

        def prepare_store(self, block_hashes: Iterable[Any]) -> Any: ...

        def prepare_load(self, block_hashes: Iterable[Any]) -> Any: ...

        def touch(self, block_hashes: Iterable[Any]) -> None: ...

        def complete_load(self, block_hashes: Iterable[Any]) -> None: ...

        def complete_store(
            self, block_hashes: Iterable[Any], success: bool = ...
        ) -> None: ...

        def take_events(self) -> Iterable[Any]: ...


# ---------------------------------------------------------------------------
# Stub out the vllm import chain so reuse_manager.py loads without CUDA/torch.
# ---------------------------------------------------------------------------
_vllm = types.ModuleType("vllm")
_v1 = types.ModuleType("vllm.v1")
_core = types.ModuleType("vllm.v1.core")
_kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
_kv_cache_utils.BlockHash = int  # type: ignore[attr-defined]
_kv_offload = types.ModuleType("vllm.v1.kv_offload")
_abstract = types.ModuleType("vllm.v1.kv_offload.abstract")

from abc import ABC, abstractmethod  # noqa: E402
from dataclasses import dataclass  # noqa: E402


class _LoadStoreSpec(ABC):
    @staticmethod
    @abstractmethod
    def medium() -> str: ...


@dataclass
class _PrepareStoreOutput:
    block_hashes_to_store: list
    store_spec: Any
    block_hashes_evicted: list


class _OffloadingManager(ABC):
    @abstractmethod
    def lookup(self, block_hashes: Iterable) -> int | None: ...

    @abstractmethod
    def prepare_load(self, block_hashes: Iterable) -> Any: ...

    @abstractmethod
    def prepare_store(self, block_hashes: Iterable) -> Any: ...

    def touch(self, block_hashes: Iterable) -> None:
        return

    def complete_load(self, block_hashes: Iterable) -> None:
        return

    def complete_store(self, block_hashes: Iterable, success: bool = True) -> None:
        return

    def take_events(self) -> Iterable:
        return ()


class _OffloadingEvent:
    pass


_abstract.LoadStoreSpec = _LoadStoreSpec  # type: ignore[attr-defined]
_abstract.PrepareStoreOutput = _PrepareStoreOutput  # type: ignore[attr-defined]
_abstract.OffloadingManager = _OffloadingManager  # type: ignore[attr-defined]
_abstract.OffloadingEvent = _OffloadingEvent  # type: ignore[attr-defined]

sys.modules.setdefault("vllm", _vllm)
sys.modules.setdefault("vllm.v1", _v1)
sys.modules.setdefault("vllm.v1.core", _core)
sys.modules.setdefault("vllm.v1.core.kv_cache_utils", _kv_cache_utils)
sys.modules.setdefault("vllm.v1.kv_offload", _kv_offload)
sys.modules.setdefault("vllm.v1.kv_offload.abstract", _abstract)


_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "vllm/v1/kv_offload/reuse_manager.py"
)
_spec = importlib.util.spec_from_file_location("reuse_manager", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

if not TYPE_CHECKING:
    BlockReuseTracker = _mod.BlockReuseTracker  # type: ignore[assignment,misc]
    FilteredOffloadingManager = (  # type: ignore[assignment,misc]
        _mod.FilteredOffloadingManager
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tracker(**kwargs: Any) -> BlockReuseTracker:
    defaults: dict[str, Any] = dict(max_size=100, store_threshold=2)
    defaults.update(kwargs)
    return BlockReuseTracker(**defaults)


def make_mock_backing(
    prepare_store_return: Any = None,
) -> Any:
    """Return a minimal mock OffloadingManager."""
    backing = MagicMock()
    if prepare_store_return is None:
        prepare_store_return = _PrepareStoreOutput(
            block_hashes_to_store=[],
            store_spec=None,
            block_hashes_evicted=[],
        )
    backing.prepare_store.return_value = prepare_store_return
    backing.lookup.return_value = 0
    backing.take_events.return_value = []
    return backing


# ---------------------------------------------------------------------------
# Tests: BlockReuseTracker.record / check
# ---------------------------------------------------------------------------


class TestBlockReuseTrackerRecord:
    def test_unseen_hash_check_returns_false(self) -> None:
        t = make_tracker(store_threshold=2)
        assert t.check(1) is False

    def test_single_record_below_threshold(self) -> None:
        t = make_tracker(store_threshold=2)
        t.record(1)
        assert t.check(1) is False

    def test_meets_threshold_after_enough_records(self) -> None:
        t = make_tracker(store_threshold=2)
        t.record(1)
        t.record(1)
        assert t.check(1) is True

    def test_threshold_of_one(self) -> None:
        t = make_tracker(store_threshold=1)
        t.record(42)
        assert t.check(42) is True

    def test_exceeds_threshold(self) -> None:
        t = make_tracker(store_threshold=2)
        for _ in range(5):
            t.record(7)
        assert t.check(7) is True

    def test_record_does_not_affect_other_hashes(self) -> None:
        t = make_tracker(store_threshold=2)
        t.record(1)
        t.record(1)
        assert t.check(2) is False

    def test_check_does_not_mutate_count(self) -> None:
        t = make_tracker(store_threshold=3)
        t.record(5)
        t.check(5)
        t.record(5)
        t.check(5)
        # Still only 2 records, threshold is 3
        assert t.check(5) is False
        t.record(5)
        assert t.check(5) is True


class TestBlockReuseTrackerLRUEviction:
    def test_lru_eviction_when_full(self) -> None:
        t = make_tracker(max_size=3, store_threshold=2)
        t.record(1)
        t.record(2)
        t.record(3)
        # Access hash 1 to make it MRU
        t.record(1)
        # Insert hash 4 — should evict hash 2 (LRU)
        t.record(4)
        assert t.check(2) is False  # evicted
        assert t.check(1) is True  # still present (count=2, threshold=2 → True)

    def test_max_size_one(self) -> None:
        t = make_tracker(max_size=1, store_threshold=2)
        t.record(1)
        t.record(1)
        assert t.check(1) is True
        t.record(2)  # evicts 1
        assert t.check(1) is False
        assert t.check(2) is False

    def test_max_size_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            BlockReuseTracker(max_size=0)


# ---------------------------------------------------------------------------
# Tests: FilteredOffloadingManager
# ---------------------------------------------------------------------------


class TestFilteredOffloadingManager:
    def _make_manager(
        self, threshold: int = 2, max_size: int = 100
    ) -> tuple[FilteredOffloadingManager, Any]:
        backing = make_mock_backing()
        mgr = FilteredOffloadingManager(
            backing=backing,
            store_threshold=threshold,
            max_tracker_size=max_size,
        )
        return mgr, backing

    def test_lookup_records_hashes_and_delegates(self) -> None:
        mgr, backing = self._make_manager(threshold=2)
        backing.lookup.return_value = 3
        result = mgr.lookup([10, 20, 30])
        assert result == 3
        backing.lookup.assert_called_once_with([10, 20, 30])
        # Hashes recorded once — below threshold
        assert mgr._tracker.check(10) is False

    def test_lookup_increments_tracker(self) -> None:
        mgr, backing = self._make_manager(threshold=2)
        mgr.lookup([99])
        mgr.lookup([99])
        assert mgr._tracker.check(99) is True

    def test_prepare_store_filters_below_threshold(self) -> None:
        mgr, backing = self._make_manager(threshold=2)
        # Record hash 1 only once — below threshold
        mgr._tracker.record(1)
        mgr.prepare_store([1])
        # Backing should be called with empty list (1 filtered out)
        backing.prepare_store.assert_called_once_with([])

    def test_prepare_store_passes_threshold_hashes(self) -> None:
        mgr, backing = self._make_manager(threshold=2)
        mgr._tracker.record(1)
        mgr._tracker.record(1)
        mgr._tracker.record(2)  # only once — filtered
        mgr.prepare_store([1, 2])
        backing.prepare_store.assert_called_once_with([1])

    def test_prepare_store_all_below_threshold(self) -> None:
        mgr, backing = self._make_manager(threshold=3)
        mgr.prepare_store([5, 6, 7])
        # All filtered — backing called with []
        backing.prepare_store.assert_called_once_with([])

    def test_prepare_store_all_above_threshold(self) -> None:
        mgr, backing = self._make_manager(threshold=1)
        mgr._tracker.record(1)
        mgr._tracker.record(2)
        mgr.prepare_store([1, 2])
        backing.prepare_store.assert_called_once_with([1, 2])

    def test_prepare_load_delegates(self) -> None:
        mgr, backing = self._make_manager()
        mgr.prepare_load([1, 2])
        backing.prepare_load.assert_called_once_with([1, 2])

    def test_touch_delegates(self) -> None:
        mgr, backing = self._make_manager()
        mgr.touch([3])
        backing.touch.assert_called_once_with([3])

    def test_complete_load_delegates(self) -> None:
        mgr, backing = self._make_manager()
        mgr.complete_load([3])
        backing.complete_load.assert_called_once_with([3])

    def test_complete_store_delegates(self) -> None:
        mgr, backing = self._make_manager()
        mgr.complete_store([3], success=True)
        backing.complete_store.assert_called_once_with([3], True)

    def test_take_events_delegates(self) -> None:
        mgr, backing = self._make_manager()
        backing.take_events.return_value = ["event1"]
        result = mgr.take_events()
        assert result == ["event1"]

    def test_check_called_before_backing_prepare_store(self) -> None:
        """Ensures filter happens before backing.prepare_store, not after."""
        mgr, backing = self._make_manager(threshold=2)
        # Record hash 10 twice (above threshold), hash 20 once (below)
        mgr._tracker.record(10)
        mgr._tracker.record(10)
        mgr._tracker.record(20)
        mgr.prepare_store([10, 20])
        # Only hash 10 should reach the backing manager
        args = backing.prepare_store.call_args[0][0]
        assert 10 in args
        assert 20 not in args
