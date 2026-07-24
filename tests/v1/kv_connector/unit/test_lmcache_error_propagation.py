# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LMCacheMPWorkerAdapter error propagation."""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import zmq

# Modules to mock so that the in-tree fallback adapter can be imported
# without the external lmcache package installed. The mock is installed
# and torn down inside the _lmcache_mock fixture — no side effects at
# module level.
_LMCACHE_MODULES = [
    "lmcache",
    "lmcache.config",
    "lmcache.logging",
    "lmcache.observability",
    "lmcache.utils",
    "lmcache.v1",
    "lmcache.v1.cache_engine",
    "lmcache.v1.compute",
    "lmcache.v1.compute.blend",
    "lmcache.v1.config",
    "lmcache.v1.gpu_connector",
    "lmcache.v1.internal_api_server",
    "lmcache.v1.internal_api_server.api_server",
    "lmcache.v1.lookup_client",
    "lmcache.v1.lookup_client.lmcache_async_lookup_client",
    "lmcache.v1.offload_server",
    "lmcache.v1.offload_server.zmq_server",
    "lmcache.v1.plugin",
    "lmcache.v1.plugin.runtime_plugin_launcher",
    "lmcache.v1.plugin.plugin_launcher",
    "lmcache.v1.multiprocess",
    "lmcache.v1.multiprocess.custom_types",
    "lmcache.v1.multiprocess.mq",
    "lmcache.v1.multiprocess.protocol",
]

# Purely for type-checking — the real import happens inside _lmcache_mock.
if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.multi_process_adapter import (  # noqa: E501
        LMCacheMPWorkerAdapter,
        LoadStoreOp,  # noqa: F401
        ParallelStrategy,  # noqa: F401
    )

# Populated by _lmcache_mock fixture — used by TestSubmitRetrievePath tests
# that construct LoadStoreOp instances at runtime.
_load_store_op = None


@pytest.fixture(scope="module")
def _lmcache_mock():
    """Install lmcache mocks, import adapter classes, and tear down cleanly.

    Replaces every lmcache.* entry in sys.modules with a MagicMock so
    the in-tree fallback adapter can be imported without the external
    lmcache package. The mock + import are scoped inside this fixture;
    other test modules are never exposed to the mocked state.

    Teardown restores the original sys.modules entries AND removes the
    cached adapter module so that later tests re-import fresh.
    """
    # -- setup: save original state -----------------------------------
    global _load_store_op
    _sentinel = object()

    saved_lmcache = {m: sys.modules.get(m) for m in _LMCACHE_MODULES}
    _adapter_module_keys = [
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration",
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.multi_process_adapter",
    ]
    # Map: child module key → (parent module key, attribute name on parent)
    _li_key = _adapter_module_keys[0]  # ...v1.lmcache_integration
    _mp_key = _adapter_module_keys[1]  # ...multi_process_adapter
    _parent_attr_map = {
        _li_key: (
            "vllm.distributed.kv_transfer.kv_connector.v1",
            "lmcache_integration",
        ),
        _mp_key: (_li_key, "multi_process_adapter"),
    }

    # Save pre-existing adapter modules
    saved_adapter_modules = {k: sys.modules.get(k) for k in _adapter_module_keys}

    # Save parent package attribute state.  Three cases:
    #   (pkg, attr, value, False)  — parent existed, restore value after
    #   (pkg, attr, _sentinel, False) — parent existed, attr absent, delete after
    #   (parent_key, attr, None, True) — parent absent, delete new attr after
    saved_parent_attrs = []
    for child_key in _adapter_module_keys:
        parent_key, attr_name = _parent_attr_map[child_key]
        parent_pkg = sys.modules.get(parent_key)
        if parent_pkg is not None:
            old = getattr(parent_pkg, attr_name, _sentinel)
            saved_parent_attrs.append((parent_pkg, attr_name, old, False))
        else:
            saved_parent_attrs.append((parent_key, attr_name, None, True))

    # Install lmcache mocks
    for mod in _LMCACHE_MODULES:
        sys.modules[mod] = MagicMock()
    sys.modules["lmcache.utils"]._lmcache_nvtx_annotate = lambda f: f
    sys.modules["lmcache.utils"].init_logger = MagicMock(return_value=MagicMock())
    sys.modules["lmcache.logging"].init_logger = MagicMock(return_value=MagicMock())

    # Evict cached adapter modules from sys.modules so the import below
    # creates fresh copies inside the mocked lmcache environment.
    for key in _adapter_module_keys:
        sys.modules.pop(key, None)

    # Import adapter classes inside the mocked environment.
    # try/finally guarantees cleanup even if the import fails, so that
    # lmcache MagicMock entries do not leak into other test modules.
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.multi_process_adapter import (  # noqa: E501
            LMCacheMPWorkerAdapter,
            LoadStoreOp,
            ParallelStrategy,
        )

        _load_store_op = LoadStoreOp
        yield LMCacheMPWorkerAdapter, LoadStoreOp, ParallelStrategy
    finally:
        # -- teardown: restore to pre-fixture state -------------------
        _load_store_op = None

        # 1. Restore lmcache sys.modules
        for mod, original in saved_lmcache.items():
            if original is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = original

        # 2. Restore adapter sys.modules
        for key, original in saved_adapter_modules.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original

        # 3. Restore parent package attributes
        for pkg_or_key, attr_name, saved_value, was_absent in saved_parent_attrs:
            if was_absent:
                # Parent didn't exist before fixture — our import may have
                # created it and set the attribute.  Delete it if present.
                parent = sys.modules.get(pkg_or_key)
                if parent is not None:
                    with contextlib.suppress(AttributeError):
                        delattr(parent, attr_name)
            elif saved_value is _sentinel:
                with contextlib.suppress(AttributeError):
                    delattr(pkg_or_key, attr_name)
            else:
                setattr(pkg_or_key, attr_name, saved_value)


@pytest.fixture
def adapter(_lmcache_mock):
    """Create a fresh LMCacheMPWorkerAdapter with mocked dependencies.

    Patches the lmcache server interaction so get_lmcache_chunk_size
    returns a valid value without requiring a real LMCache process.
    """
    LMCacheMPWorkerAdapter, _, ParallelStrategy = _lmcache_mock

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
            ".multi_process_adapter.get_lmcache_chunk_size",
            return_value=256,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
            ".multi_process_adapter.MessageQueueClient",
        ),
    ):
        strategy = ParallelStrategy(
            use_mla=False,
            kv_world_size=1,
            kv_worker_id=0,
            actual_world_size=1,
            actual_worker_id=0,
            tp_size=1,
            pp_size=1,
        )
        return LMCacheMPWorkerAdapter(
            server_url="inproc://test",
            context=zmq.Context(),
            model_name="test-model",
            vllm_block_size=16,
            parallel_strategy=strategy,
        )


class TestGetBlockIdsWithLoadErrors:
    """Tests for get_block_ids_with_load_errors drain-and-reset behavior."""

    def test_returns_empty_set_when_no_errors(self, adapter: LMCacheMPWorkerAdapter):
        result = adapter.get_block_ids_with_load_errors()
        assert result == set()

    def test_returns_accumulated_errors_and_resets(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        adapter._load_error_block_ids = {1, 2, 3}
        result = adapter.get_block_ids_with_load_errors()
        assert result == {1, 2, 3}
        assert adapter._load_error_block_ids == set()

    def test_consecutive_calls_return_independent_sets(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        adapter._load_error_block_ids = {10, 20}
        first = adapter.get_block_ids_with_load_errors()
        assert first == {10, 20}
        # No new errors accumulated
        second = adapter.get_block_ids_with_load_errors()
        assert second == set()
        # New errors accumulated after reset
        adapter._load_error_block_ids.add(30)
        third = adapter.get_block_ids_with_load_errors()
        assert third == {30}


class TestStoreFutureErrorHandling:
    """Tests for store future exception handling in get_finished."""

    def test_store_exception_does_not_crash_and_cleans_up_future(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        """Store exception should log and clean up, not propagate."""
        mock_future = MagicMock()
        mock_future.query.return_value = True
        mock_future.result.side_effect = RuntimeError("store failure")
        adapter.store_futures = {"req-1": (mock_future, [])}

        _, finished_retrieves = adapter.get_finished(set())

        # The store future should be cleaned up
        assert "req-1" not in adapter.store_futures
        # The request is tracked in finished_stores for eventual reporting
        assert "req-1" in adapter.finished_stores
        assert finished_retrieves == set()

    def test_store_false_result_logs_and_cleans_up(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        """Store returning False should log error and clean up."""
        mock_future = MagicMock()
        mock_future.query.return_value = True
        mock_future.result.return_value = False
        adapter.store_futures = {"req-1": (mock_future, [])}

        _, finished_retrieves = adapter.get_finished(set())

        assert "req-1" not in adapter.store_futures
        assert "req-1" in adapter.finished_stores
        assert finished_retrieves == set()


class TestRetrieveFutureErrorHandling:
    """Tests for retrieve future error handling in get_finished."""

    def test_retrieve_exception_reports_all_tracked_blocks_as_errors(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        mock_future = MagicMock()
        mock_future.query.return_value = True
        mock_future.result.side_effect = RuntimeError("retrieve failure")
        adapter.retrieve_futures = {"req-2": (mock_future, [])}
        adapter._retrieve_block_ids = {"req-2": [5, 6, 7]}

        _, finished_retrieves = adapter.get_finished(set())

        assert finished_retrieves is not None
        assert "req-2" in finished_retrieves
        assert adapter.get_block_ids_with_load_errors() == {5, 6, 7}
        assert adapter.get_block_ids_with_load_errors() == set()

    def test_retrieve_exception_cleans_up_future(self, adapter: LMCacheMPWorkerAdapter):
        mock_future = MagicMock()
        mock_future.query.return_value = True
        mock_future.result.side_effect = RuntimeError("retrieve failure")
        adapter.retrieve_futures = {"req-2": (mock_future, [])}
        adapter._retrieve_block_ids = {"req-2": [5, 6, 7]}

        adapter.get_finished(set())

        assert "req-2" not in adapter.retrieve_futures
        assert "req-2" not in adapter._retrieve_block_ids

    def test_retrieve_partial_failure_reports_tracked_blocks(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        mock_future = MagicMock()
        mock_future.query.return_value = True
        mock_future.result.return_value = [True, False, True]  # 2nd chunk failed
        adapter.retrieve_futures = {"req-3": (mock_future, [])}
        adapter._retrieve_block_ids = {"req-3": [10, 20, 30, 40]}

        _, finished_retrieves = adapter.get_finished(set())

        assert finished_retrieves is not None
        assert "req-3" in finished_retrieves
        # All tracked blocks are conservatively reported, regardless of
        # which chunk failed (RetrieveResult is per-chunk, block_ids per-block)
        assert adapter.get_block_ids_with_load_errors() == {
            10,
            20,
            30,
            40,
        }
        assert adapter.get_block_ids_with_load_errors() == set()

    def test_retrieve_all_success_does_not_report_errors(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        mock_future = MagicMock()
        mock_future.query.return_value = True
        mock_future.result.return_value = [True, True, True]
        adapter.retrieve_futures = {"req-4": (mock_future, [])}
        adapter._retrieve_block_ids = {"req-4": [1, 2, 3]}

        _, finished_retrieves = adapter.get_finished(set())

        assert finished_retrieves is not None
        assert "req-4" in finished_retrieves
        assert adapter.get_block_ids_with_load_errors() == set()

    def test_retrieve_all_success_cleans_up_block_id_tracking(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        mock_future = MagicMock()
        mock_future.query.return_value = True
        mock_future.result.return_value = [True]
        adapter.retrieve_futures = {"req-5": (mock_future, [])}
        adapter._retrieve_block_ids = {"req-5": [1]}

        adapter.get_finished(set())

        assert "req-5" not in adapter._retrieve_block_ids


class TestRetrieveBlockIdTracking:
    """Tests for _retrieve_block_ids lifecycle."""

    def test_shutdown_cleans_up_tracking_dicts(self, adapter: LMCacheMPWorkerAdapter):
        adapter._retrieve_block_ids = {"req-a": [1, 2]}
        adapter._load_error_block_ids = {3, 4}

        adapter.shutdown()

        assert adapter._retrieve_block_ids == {}
        assert adapter._load_error_block_ids == set()

    def test_get_finished_cleans_up_stale_block_ids(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        """Block IDs from a previous error should not leak into a subsequent call."""
        mock_future = MagicMock()
        mock_future.query.return_value = True
        mock_future.result.side_effect = RuntimeError("fail")
        adapter.retrieve_futures = {"req-x": (mock_future, [])}
        adapter._retrieve_block_ids = {"req-x": [100, 200]}

        # First call: errors accumulate
        adapter.get_finished(set())
        assert adapter._load_error_block_ids == {100, 200}

        # Drain errors (simulates what kv_connector does each forward pass)
        drained = adapter.get_block_ids_with_load_errors()
        assert drained == {100, 200}

        # Second call with no new requests: no errors
        assert adapter.get_block_ids_with_load_errors() == set()


class TestMultipleRequests:
    """Tests for handling multiple concurrent requests."""

    def test_only_failed_requests_contribute_to_error_set(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        # Successful retrieve
        good_future = MagicMock()
        good_future.query.return_value = True
        good_future.result.return_value = [True, True]
        # Failed retrieve
        bad_future = MagicMock()
        bad_future.query.return_value = True
        bad_future.result.side_effect = RuntimeError("fail")

        adapter.retrieve_futures = {
            "req-ok": (good_future, []),
            "req-bad": (bad_future, []),
        }
        adapter._retrieve_block_ids = {
            "req-ok": [1, 2],
            "req-bad": [3, 4],
        }

        adapter.get_finished(set())

        error_ids = adapter.get_block_ids_with_load_errors()
        # The successful request should not contribute to errors
        assert 1 not in error_ids
        assert 2 not in error_ids
        # The failed request should
        assert 3 in error_ids
        assert 4 in error_ids
        # Drain-reset
        assert adapter.get_block_ids_with_load_errors() == set()

    def test_batched_retrieve_merges_other_requests_into_finished(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        """Other request IDs in a batch should be included in finished set."""
        mock_future = MagicMock()
        mock_future.query.return_value = True
        mock_future.result.return_value = [True]
        adapter.retrieve_futures = {
            "req-lead": (mock_future, ["req-follower-1", "req-follower-2"])
        }
        adapter._retrieve_block_ids = {"req-lead": [1, 2, 3]}

        _, finished_retrieves = adapter.get_finished(set())

        assert finished_retrieves is not None
        assert "req-lead" in finished_retrieves
        assert "req-follower-1" in finished_retrieves
        assert "req-follower-2" in finished_retrieves


class TestSubmitRetrievePath:
    """Full-path tests going through submit_retrieve_request APIs.

    These verify that the block-id tracking wired into submit_retrieve_request
    and batched_submit_retrieve_requests survives through to error reporting.
    """

    @staticmethod
    def _make_mock_future(result=None, side_effect=None):
        """Build a mock future chain: send_lmcache_request → .to_cuda_future()."""
        cuda_future = MagicMock()
        cuda_future.query.return_value = True
        if side_effect is not None:
            cuda_future.result.side_effect = side_effect
        elif result is not None:
            cuda_future.result.return_value = result
        else:
            cuda_future.result.return_value = [True]
        mock_future = MagicMock()
        mock_future.to_cuda_future.return_value = cuda_future
        return mock_future, cuda_future

    def test_submit_retrieve_request_tracks_block_ids(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        """Calling submit_retrieve_request must populate _retrieve_block_ids."""
        mock_future, _ = self._make_mock_future()
        mock_event = MagicMock()
        op = _load_store_op(block_ids=[7, 8, 9], token_ids=[1, 2, 3])

        with patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
            ".multi_process_adapter.send_lmcache_request",
            return_value=mock_future,
        ):
            adapter.submit_retrieve_request("req-submit-1", op, mock_event)

        assert "req-submit-1" in adapter._retrieve_block_ids
        assert adapter._retrieve_block_ids["req-submit-1"] == [7, 8, 9]
        assert "req-submit-1" in adapter.retrieve_futures

    def test_submit_then_exception_reports_blocks(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        """Submit a retrieve, make it fail, verify blocks reported as errors."""
        mock_future, _ = self._make_mock_future(
            side_effect=RuntimeError("retrieve failed")
        )
        mock_event = MagicMock()
        op = _load_store_op(block_ids=[50, 60], token_ids=[1, 2])

        with patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
            ".multi_process_adapter.send_lmcache_request",
            return_value=mock_future,
        ):
            adapter.submit_retrieve_request("req-fail-1", op, mock_event)

        _, finished_retrieves = adapter.get_finished(set())

        assert finished_retrieves is not None
        assert "req-fail-1" in finished_retrieves
        assert adapter.get_block_ids_with_load_errors() == {50, 60}
        assert adapter.get_block_ids_with_load_errors() == set()

    def test_submit_then_partial_failure_reports_all_blocks(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        """Submit → partial retrieve failure → conservative block report."""
        mock_future, _ = self._make_mock_future(result=[True, False])
        mock_event = MagicMock()
        # 6 blocks → token mode: 1 key → result len 1, not 6
        # We use token mode so len(result) != len(block_ids) is guaranteed
        op = _load_store_op(
            block_ids=[11, 12, 13, 14, 15, 16], token_ids=list(range(32))
        )

        with patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
            ".multi_process_adapter.send_lmcache_request",
            return_value=mock_future,
        ):
            adapter.submit_retrieve_request("req-partial", op, mock_event)

        _, finished_retrieves = adapter.get_finished(set())

        assert finished_retrieves is not None
        assert "req-partial" in finished_retrieves
        assert adapter.get_block_ids_with_load_errors() == {
            11,
            12,
            13,
            14,
            15,
            16,
        }
        assert adapter.get_block_ids_with_load_errors() == set()

    def test_batched_submit_tracks_combined_block_ids(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        """batched_submit should track all block IDs under primary request_id."""
        mock_future, _ = self._make_mock_future()
        mock_event = MagicMock()
        ops = [
            _load_store_op(block_ids=[1, 2], token_ids=[10, 20]),
            _load_store_op(block_ids=[3, 4], token_ids=[30, 40]),
        ]
        request_ids = ["req-head", "req-tail-1"]

        with patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
            ".multi_process_adapter.send_lmcache_request",
            return_value=mock_future,
        ):
            adapter.batched_submit_retrieve_requests(request_ids, ops, mock_event)

        # All block IDs from all ops tracked under head request
        assert adapter._retrieve_block_ids["req-head"] == [1, 2, 3, 4]
        _, other_reqs = adapter.retrieve_futures["req-head"]
        assert "req-tail-1" in other_reqs

    def test_batched_submit_then_failure_reports_all_combined_blocks(
        self, adapter: LMCacheMPWorkerAdapter
    ):
        """Batched submit → future fails → all combined blocks in error set."""
        mock_future, _ = self._make_mock_future(
            side_effect=RuntimeError("batched retrieve failed")
        )
        mock_event = MagicMock()
        ops = [
            _load_store_op(block_ids=[10, 20], token_ids=[1, 2]),
            _load_store_op(block_ids=[30, 40], token_ids=[3, 4]),
            _load_store_op(block_ids=[50, 60], token_ids=[5, 6]),
        ]
        request_ids = ["req-head", "req-2", "req-3"]

        with patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
            ".multi_process_adapter.send_lmcache_request",
            return_value=mock_future,
        ):
            adapter.batched_submit_retrieve_requests(request_ids, ops, mock_event)

        _, finished_retrieves = adapter.get_finished(set())

        assert finished_retrieves is not None
        assert "req-head" in finished_retrieves
        assert "req-2" in finished_retrieves
        assert "req-3" in finished_retrieves
        assert adapter.get_block_ids_with_load_errors() == {
            10,
            20,
            30,
            40,
            50,
            60,
        }
        assert adapter.get_block_ids_with_load_errors() == set()
