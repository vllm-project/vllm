# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# flake8: noqa: E402
# ruff: noqa: E402

import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock


class MockLmcacheLoader(Loader):
    def create_module(self, spec):
        mod = MagicMock()
        mod.__path__ = []  # Treat as a package
        # Special bypass for the decorator
        if spec.name == "lmcache.utils":
            mod._lmcache_nvtx_annotate = lambda f: f
        return mod

    def exec_module(self, module):
        pass


class MockLmcacheFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("lmcache"):
            return ModuleSpec(fullname, MockLmcacheLoader())
        return None


# Insert our custom mock finder into Python's import system
sys.meta_path.insert(0, MockLmcacheFinder())


# Now the target adapter can be safely imported without external dependencies.
# fmt: off
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.multi_process_adapter import (  # noqa: E501
    LMCacheMPWorkerAdapter,
)

# fmt: on


def test_get_finished_filters_aborted_requests():
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter.store_futures = {}
    adapter.retrieve_futures = {}
    adapter.finished_stores = set()
    adapter.previously_finished = set()

    dead_req_id = "req_dead_1"
    mock_future = MagicMock()
    mock_future.query.return_value = True
    mock_future.result.return_value = [True]

    adapter.retrieve_futures[dead_req_id] = (mock_future, [])
    finished_reqs = {dead_req_id}

    ret_stores, finished_retrieves = adapter.get_finished(finished_reqs)

    assert dead_req_id not in finished_retrieves, (
        "Failed: Aborted request leaked to finished_retrieves"
    )
    assert dead_req_id not in adapter.retrieve_futures, (
        "Failed: Aborted request remains in retrieve_futures"
    )


def test_get_finished_filters_aborted_batched_requests():
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter.store_futures = {}
    adapter.retrieve_futures = {}
    adapter.finished_stores = set()
    adapter.previously_finished = set()

    live_req_id = "req_live_1"
    aborted_req_id = "req_dead_batch_1"

    mock_future = MagicMock()
    mock_future.query.return_value = True
    mock_future.result.return_value = [True]

    # Simulating batched_submit_retrieve_requests:
    # primary ID is live_req_id, other_reqs contains aborted_req_id
    adapter.retrieve_futures[live_req_id] = (mock_future, [aborted_req_id])

    # Engine says aborted_req_id is finished/aborted
    finished_reqs = {aborted_req_id}

    ret_stores, finished_retrieves = adapter.get_finished(finished_reqs)

    # The live request should be returned as finished retrieving
    assert live_req_id in finished_retrieves, (
        "Failed: Live primary request missing from finished_retrieves"
    )
    # The aborted secondary request should NOT be returned
    assert aborted_req_id not in finished_retrieves, (
        "Failed: Aborted batched request leaked to finished_retrieves"
    )
    # The underlying future should be cleaned up
    assert live_req_id not in adapter.retrieve_futures, (
        "Failed: Completed batched retrieve future was not removed"
    )
