# SPDX-License-Identifier: Apache-2.0
"""Tests for ``batched_contains()`` and ``batched_nixl_desc_exists()``.

All tests use lightweight mocks so they run without NIXL or CUDA hardware.
The mock approach binds the *real* unbound method onto a ``Mock(spec=…)``
object, so the actual implementation is exercised while external
dependencies (NIXL agent, memory allocators) are replaced by mocks.
"""

# Standard
from typing import List
from unittest.mock import Mock

# Third Party
import pytest
import torch

pytest.importorskip("nixl")

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.nixl_storage_backend import (
    NixlDynamicStorageAgent,
    NixlDynamicStorageBackend,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_key(chunk_hash: int) -> CacheEngineKey:
    """Create a single CacheEngineKey for testing."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.bfloat16,
    )


def _make_keys(n: int) -> List[CacheEngineKey]:
    """Create *n* distinct CacheEngineKeys (chunk_hash = 0 … n-1)."""
    return [_make_key(i) for i in range(n)]


def _mock_backend(**overrides) -> Mock:
    """Return a ``Mock(spec=NixlDynamicStorageBackend)`` with sensible defaults.

    Every test that exercises ``batched_contains`` or ``contains`` needs
    a backend with at least ``_exists_in_put_tasks_or_cache``,
    ``_format_object_key``, ``_cache_add``, and ``agent`` wired up.
    """
    backend = Mock(spec=NixlDynamicStorageBackend)
    backend.agent = Mock()
    backend._cache_add = Mock()
    backend.presence_cache_only = False
    # Default: _format_object_key returns a predictable string
    backend._format_object_key = Mock(
        side_effect=lambda key: f"formatted_{key.chunk_hash}"
    )
    backend.agent.batched_nixl_desc_exists = Mock(return_value=0)
    # batched_contains() reads self.path to pass it to the agent; the value
    # is unused here (the agent is mocked), but Mock(spec=...) raises on the
    # unset instance attribute without it.
    backend.path = None
    for k, v in overrides.items():
        setattr(backend, k, v)
    return backend


# ---------------------------------------------------------------------------
# NixlDynamicStorageAgent.batched_nixl_desc_exists
# ---------------------------------------------------------------------------


class TestBatchedNixlDescExists:
    """Unit tests for ``NixlDynamicStorageAgent.batched_nixl_desc_exists``."""

    @staticmethod
    def _make_agent() -> Mock:
        agent = Mock(spec=NixlDynamicStorageAgent)
        agent.nixl_agent = Mock()
        agent.backend = "OBJ"
        agent.mem_type = "OBJ"
        return agent

    @staticmethod
    def _call(agent: Mock, reg_list):
        return NixlDynamicStorageAgent.batched_nixl_desc_exists(agent, reg_list)

    def test_empty_list(self) -> None:
        agent = self._make_agent()
        assert self._call(agent, []) == 0

    def test_all_exist(self) -> None:
        agent = self._make_agent()
        reg_list = [(0, 0, 0, "k1"), (0, 0, 0, "k2"), (0, 0, 0, "k3")]
        agent.nixl_agent.query_memory.return_value = ["d1", "d2", "d3"]
        assert self._call(agent, reg_list) == 3
        agent.nixl_agent.query_memory.assert_called_once_with(
            reg_list, "OBJ", mem_type="OBJ"
        )

    def test_consecutive_then_miss(self) -> None:
        agent = self._make_agent()
        agent.nixl_agent.query_memory.return_value = ["d1", "d2", None, "d4"]
        assert self._call(agent, [(0, 0, 0, f"k{i}") for i in range(4)]) == 2

    def test_first_missing(self) -> None:
        agent = self._make_agent()
        agent.nixl_agent.query_memory.return_value = [None, "d2"]
        assert self._call(agent, [(0, 0, 0, "k1"), (0, 0, 0, "k2")]) == 0

    def test_exception_returns_zero(self) -> None:
        agent = self._make_agent()
        agent.nixl_agent.query_memory.side_effect = RuntimeError("boom")
        assert self._call(agent, [(0, 0, 0, "k1")]) == 0

    # -- FILE backend: resolves via os.path.exists, not query_memory --------

    @staticmethod
    def _make_file_agent() -> Mock:
        agent = Mock(spec=NixlDynamicStorageAgent)
        agent.nixl_agent = Mock()
        agent.backend = "POSIX"
        agent.mem_type = "FILE"
        return agent

    def test_file_all_exist(self) -> None:
        agent = self._make_file_agent()
        agent.nixl_desc_exists.return_value = True
        reg_list = [(0, 0, 0, "k1"), (0, 0, 0, "k2"), (0, 0, 0, "k3")]
        assert (
            NixlDynamicStorageAgent.batched_nixl_desc_exists(agent, reg_list, "/dir")
            == 3
        )
        # FILE must not use query_memory (it only answers for object stores)
        agent.nixl_agent.query_memory.assert_not_called()
        agent.nixl_desc_exists.assert_any_call("k1", "/dir")

    def test_file_consecutive_then_miss(self) -> None:
        agent = self._make_file_agent()
        agent.nixl_desc_exists.side_effect = [True, True, False, True]
        reg_list = [(0, 0, 0, f"k{i}") for i in range(4)]
        assert (
            NixlDynamicStorageAgent.batched_nixl_desc_exists(agent, reg_list, "/dir")
            == 2
        )

    def test_file_first_missing(self) -> None:
        agent = self._make_file_agent()
        agent.nixl_desc_exists.side_effect = [False, True]
        reg_list = [(0, 0, 0, "k1"), (0, 0, 0, "k2")]
        assert (
            NixlDynamicStorageAgent.batched_nixl_desc_exists(agent, reg_list, "/dir")
            == 0
        )

    def test_file_requires_path(self) -> None:
        agent = self._make_file_agent()
        with pytest.raises(ValueError, match="path must be provided"):
            NixlDynamicStorageAgent.batched_nixl_desc_exists(agent, [(0, 0, 0, "k1")])


# ---------------------------------------------------------------------------
# NixlDynamicStorageBackend.contains (refactored)
# ---------------------------------------------------------------------------


class TestContains:
    """Verify the refactored ``contains()`` delegates correctly."""

    @staticmethod
    def _call(backend: Mock, key: CacheEngineKey, pin: bool = False) -> bool:
        return NixlDynamicStorageBackend.contains(backend, key, pin)

    def test_cache_hit(self) -> None:
        """When the helper finds the key in the presence cache, return True."""
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.return_value = (True, True)

        assert self._call(backend, _make_key(1)) is True
        backend.key_exists.assert_not_called()

    def test_put_task_hit(self) -> None:
        """When the key is in put tasks, return False immediately."""
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.return_value = (True, False)

        assert self._call(backend, _make_key(1)) is False
        backend.key_exists.assert_not_called()

    def test_fallback_to_key_exists_hit(self) -> None:
        """When local lookup misses, fall back to key_exists and cache the result."""
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.return_value = (False, False)
        backend.key_exists.return_value = True

        key = _make_key(42)
        assert self._call(backend, key) is True
        backend.key_exists.assert_called_once_with(key)
        backend._cache_add.assert_called_once_with(key.chunk_hash)

    def test_fallback_to_key_exists_miss(self) -> None:
        """When key_exists also returns False, return False and don't cache."""
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.return_value = (False, False)
        backend.key_exists.return_value = False

        assert self._call(backend, _make_key(42)) is False
        backend._cache_add.assert_not_called()

    def test_presence_cache_only_skips_remote(self) -> None:
        """With presence_cache_only set, a presence-cache miss returns False
        without issuing the queryMem call (key_exists not called)."""
        backend = _mock_backend(presence_cache_only=True)
        backend._exists_in_put_tasks_or_cache.return_value = (False, False)

        assert self._call(backend, _make_key(7)) is False
        backend.key_exists.assert_not_called()
        backend._cache_add.assert_not_called()


# ---------------------------------------------------------------------------
# NixlDynamicStorageBackend.batched_contains
# ---------------------------------------------------------------------------


class TestBatchedContains:
    """Verify ``batched_contains()`` prefix-match semantics."""

    @staticmethod
    def _call(
        backend: Mock,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return NixlDynamicStorageBackend.batched_contains(backend, keys, pin)

    # -- trivial / edge cases -----------------------------------------------

    def test_empty_keys(self) -> None:
        backend = _mock_backend()
        assert self._call(backend, []) == 0

    def test_all_local_cache_hits(self) -> None:
        """Every key resolved from the local presence cache."""
        keys = _make_keys(3)
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.side_effect = [
            (True, True),
            (True, True),
            (True, True),
        ]
        assert self._call(backend, keys) == 3
        backend.agent.batched_nixl_desc_exists.assert_not_called()

    # -- early-stop on put-task ---------------------------------------------

    def test_first_key_in_put_tasks(self) -> None:
        keys = _make_keys(3)
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.side_effect = [
            (True, False),
        ]
        assert self._call(backend, keys) == 0

    def test_second_key_in_put_tasks(self) -> None:
        keys = _make_keys(3)
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.side_effect = [
            (True, True),
            (True, False),
        ]
        assert self._call(backend, keys) == 1

    # -- remote fallback ----------------------------------------------------

    def test_all_keys_remote(self) -> None:
        """No local hits; all keys go to batched_nixl_desc_exists."""
        keys = _make_keys(3)
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.side_effect = [
            (False, False),
        ]
        backend.agent.batched_nixl_desc_exists.return_value = 2

        assert self._call(backend, keys) == 2
        # Verify all 3 keys sent to agent
        call_args = backend.agent.batched_nixl_desc_exists.call_args[0][0]
        assert len(call_args) == 3

    def test_mixed_local_and_remote(self) -> None:
        """First key from cache, remaining from remote."""
        keys = _make_keys(3)
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.side_effect = [
            (True, True),
            (False, False),
        ]
        backend.agent.batched_nixl_desc_exists.return_value = 1

        assert self._call(backend, keys) == 2  # 1 local + 1 remote
        # Only the 2 remaining keys should go to agent
        call_args = backend.agent.batched_nixl_desc_exists.call_args[0][0]
        assert len(call_args) == 2

    def test_presence_cache_only_skips_remote(self) -> None:
        """With presence_cache_only set, batched_contains returns the count of
        leading local hits and never issues the remote batched query."""
        keys = _make_keys(3)
        backend = _mock_backend(presence_cache_only=True)
        backend._exists_in_put_tasks_or_cache.side_effect = [
            (True, True),
            (False, False),
        ]

        assert self._call(backend, keys) == 1
        backend.agent.batched_nixl_desc_exists.assert_not_called()
        backend._cache_add.assert_not_called()

    def test_remote_hits_are_cached(self) -> None:
        """Remote hits should be added to the presence cache."""
        keys = _make_keys(4)
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.side_effect = [
            (False, False),
        ]
        backend.agent.batched_nixl_desc_exists.return_value = 3

        self._call(backend, keys)
        # _cache_add should be called for each remote hit
        assert backend._cache_add.call_count == 3
        cached_hashes = [c.args[0] for c in backend._cache_add.call_args_list]
        assert cached_hashes == [0, 1, 2]

    # -- key formatting (catches the original chunk_hash vs formatted-key bug)

    def test_reg_list_uses_formatted_object_keys(self) -> None:
        """The reg_list sent to the agent must contain formatted object-key
        strings (from ``_format_object_key``), not raw chunk-hash integers.

        This test would have caught the original bug where ``key.chunk_hash``
        was passed instead of ``self._format_object_key(key)``.
        """
        keys = _make_keys(3)
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.side_effect = [
            (False, False),
        ]
        backend.agent.batched_nixl_desc_exists.return_value = 0

        self._call(backend, keys)

        call_args = backend.agent.batched_nixl_desc_exists.call_args[0][0]
        # Each meta_info element must be a string, not an int
        meta_infos = [t[3] for t in call_args]
        assert meta_infos == ["formatted_0", "formatted_1", "formatted_2"]
        assert all(isinstance(m, str) for m in meta_infos)

    def test_format_object_key_called_for_each_remaining_key(self) -> None:
        """_format_object_key must be called once per key that goes remote."""
        keys = _make_keys(4)
        backend = _mock_backend()
        # First 2 keys resolve locally, remaining 2 go remote
        backend._exists_in_put_tasks_or_cache.side_effect = [
            (True, True),
            (True, True),
            (False, False),
        ]
        backend.agent.batched_nixl_desc_exists.return_value = 0

        self._call(backend, keys)

        formatted_hashes = [
            c.args[0].chunk_hash for c in backend._format_object_key.call_args_list
        ]
        assert formatted_hashes == [2, 3]

    def test_single_key_present(self) -> None:
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.return_value = (True, True)
        assert self._call(backend, _make_keys(1)) == 1

    def test_single_key_missing_remote(self) -> None:
        backend = _mock_backend()
        backend._exists_in_put_tasks_or_cache.return_value = (False, False)
        backend.agent.batched_nixl_desc_exists.return_value = 0
        assert self._call(backend, _make_keys(1)) == 0
