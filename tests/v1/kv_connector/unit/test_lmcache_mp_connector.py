# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
    LMCacheMPConnectorUpstream,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)


def test_prefetch_locks_released_on_abort():
    req = MagicMock()
    req.request_id = "test-request-1"
    req.all_token_ids = list(range(32))
    req.block_hashes = []
    req.cache_salt = ""
    tracker = LMCacheMPRequestTracker(req)
    tracker.num_lmcache_hit_blocks = 2
    assert tracker.state == LMCacheMPRequestState.PREFETCHING

    connector = object.__new__(LMCacheMPConnectorUpstream)
    connector.vllm_block_size = 16
    connector.scheduler_adapter = MagicMock()
    connector.request_trackers = {"test-request-1": tracker}

    connector._cleanup_request_tracker("test-request-1")

    connector.scheduler_adapter.cleanup_lookup_result.assert_called_once_with(
        "test-request-1"
    )
    connector.scheduler_adapter.free_lookup_locks.assert_called_once_with(
        token_ids=list(range(32)), start=0, end=32, request_id="test-request-1"
    )


def test_no_double_release_after_alloc():
    req = MagicMock()
    req.request_id = "test-request-1"
    req.all_token_ids = list(range(32))
    req.block_hashes = []
    req.cache_salt = ""
    tracker = LMCacheMPRequestTracker(req)
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
    tracker.num_lmcache_hit_blocks = 2

    connector = object.__new__(LMCacheMPConnectorUpstream)
    connector.vllm_block_size = 16
    connector.scheduler_adapter = MagicMock()
    connector.request_trackers = {"test-request-1": tracker}

    connector._cleanup_request_tracker("test-request-1")

    connector.scheduler_adapter.cleanup_lookup_result.assert_not_called()
    connector.scheduler_adapter.free_lookup_locks.assert_not_called()
