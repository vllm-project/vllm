# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Responses API store LRU eviction.

These tests verify the eviction and deletion logic without importing the
full vLLM stack (no torch/CUDA dependency), by reimplementing the core
methods under test and binding them to lightweight mock objects.
"""

import asyncio
import logging
from collections import OrderedDict, deque
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Replicate the two methods under test so we can run without torch/vllm deps.
# These must stay in sync with OpenAIServingResponses in serving.py.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _evict_oldest_store_entries(self) -> None:
    while len(self.response_store) > self.store_max_size:
        evicted_id, _ = self.response_store.popitem(last=False)
        self.msg_store.pop(evicted_id, None)
        self.event_store.pop(evicted_id, None)

        # Cancel background task if it exists
        if task := self.background_tasks.pop(evicted_id, None):
            task.cancel()

        logger.debug("Evicted response %s from store (LRU)", evicted_id)


async def _delete_response(self, response_id: str):
    async with self.response_store_lock:
        response = self.response_store.pop(response_id, None)
        if response is None:
            return self._make_not_found_error(response_id)
        self.msg_store.pop(response_id, None)
        self.event_store.pop(response_id, None)

    if task := self.background_tasks.pop(response_id, None):
        task.cancel()

    response.status = "cancelled"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(response_id: str) -> MagicMock:
    resp = MagicMock()
    resp.id = response_id
    resp.status = "completed"
    return resp


def _make_serving(max_size: int = 3):
    serving = MagicMock()
    serving.response_store = OrderedDict()
    serving.msg_store = OrderedDict()
    serving.event_store = OrderedDict()
    serving.store_max_size = max_size
    serving.response_store_lock = asyncio.Lock()
    serving.background_tasks = {}
    serving._evict_oldest_store_entries = _evict_oldest_store_entries.__get__(
        serving, type(serving)
    )
    serving.delete_response = _delete_response.__get__(serving, type(serving))
    serving._make_not_found_error = MagicMock(
        return_value=MagicMock(error=MagicMock(code=404))
    )
    return serving


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStoreEviction:
    """Test LRU eviction logic for response_store, msg_store, event_store."""

    def test_eviction_removes_oldest(self):
        """When store exceeds max_size, oldest entries are evicted."""
        serving = _make_serving(max_size=2)

        for i in range(3):
            rid = f"resp_{i}"
            serving.response_store[rid] = _make_mock_response(rid)
            serving.msg_store[rid] = [{"role": "user", "content": f"msg_{i}"}]

        serving._evict_oldest_store_entries()

        assert len(serving.response_store) == 2
        assert "resp_0" not in serving.response_store
        assert "resp_1" in serving.response_store
        assert "resp_2" in serving.response_store
        assert "resp_0" not in serving.msg_store

    def test_eviction_cleans_all_stores(self):
        """Eviction removes entries from all three stores."""
        serving = _make_serving(max_size=1)

        rid = "resp_old"
        serving.response_store[rid] = _make_mock_response(rid)
        serving.msg_store[rid] = [{"role": "user", "content": "old"}]
        serving.event_store[rid] = (deque(), asyncio.Event())

        rid_new = "resp_new"
        serving.response_store[rid_new] = _make_mock_response(rid_new)

        serving._evict_oldest_store_entries()

        assert "resp_old" not in serving.response_store
        assert "resp_old" not in serving.msg_store
        assert "resp_old" not in serving.event_store
        assert "resp_new" in serving.response_store

    def test_no_eviction_under_limit(self):
        """No eviction when store is under max_size."""
        serving = _make_serving(max_size=5)

        for i in range(3):
            rid = f"resp_{i}"
            serving.response_store[rid] = _make_mock_response(rid)

        serving._evict_oldest_store_entries()

        assert len(serving.response_store) == 3

    def test_eviction_bulk(self):
        """Eviction handles large overshoot correctly."""
        serving = _make_serving(max_size=2)

        for i in range(100):
            rid = f"resp_{i}"
            serving.response_store[rid] = _make_mock_response(rid)

        serving._evict_oldest_store_entries()

        assert len(serving.response_store) == 2
        assert "resp_98" in serving.response_store
        assert "resp_99" in serving.response_store

    def test_eviction_cancels_background_tasks(self):
        """Eviction cancels background tasks for evicted responses."""
        serving = _make_serving(max_size=1)

        # Add response with background task
        rid_old = "resp_old"
        serving.response_store[rid_old] = _make_mock_response(rid_old)
        mock_task = MagicMock()
        serving.background_tasks[rid_old] = mock_task

        # Add new response to trigger eviction
        rid_new = "resp_new"
        serving.response_store[rid_new] = _make_mock_response(rid_new)
        serving._evict_oldest_store_entries()

        # Verify old task was cancelled
        mock_task.cancel.assert_called_once()
        assert rid_old not in serving.background_tasks

    @pytest.mark.asyncio
    async def test_delete_response(self):
        """delete_response removes from all stores."""
        serving = _make_serving()

        rid = "resp_to_delete"
        serving.response_store[rid] = _make_mock_response(rid)
        serving.msg_store[rid] = [{"role": "user", "content": "hi"}]
        serving.event_store[rid] = (deque(), asyncio.Event())

        result = await serving.delete_response(rid)

        assert result.status == "cancelled"
        assert rid not in serving.response_store
        assert rid not in serving.msg_store
        assert rid not in serving.event_store

    @pytest.mark.asyncio
    async def test_delete_nonexistent_response(self):
        """delete_response returns error for unknown ID."""
        serving = _make_serving()

        await serving.delete_response("nonexistent")

        serving._make_not_found_error.assert_called_once_with("nonexistent")

    def test_lru_move_to_end(self):
        """Accessing an entry moves it to end, protecting it from eviction."""
        serving = _make_serving(max_size=2)

        serving.response_store["resp_a"] = _make_mock_response("resp_a")
        serving.response_store["resp_b"] = _make_mock_response("resp_b")

        # Access resp_a, moving it to end (simulates retrieve)
        serving.response_store.move_to_end("resp_a")

        # Add a new entry, triggering eviction
        serving.response_store["resp_c"] = _make_mock_response("resp_c")
        serving._evict_oldest_store_entries()

        # resp_b should be evicted (oldest), resp_a should survive
        assert "resp_b" not in serving.response_store
        assert "resp_a" in serving.response_store
        assert "resp_c" in serving.response_store
