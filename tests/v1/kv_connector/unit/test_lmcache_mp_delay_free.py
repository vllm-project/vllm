# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block frees are delayed only for requests that scheduled a store."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("lmcache")

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPConnectorUpstream,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)

BLOCK_SIZE = 16
BLOCKS_IN_CHUNK = 2


class _Request:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.kv_transfer_params = None


class _SynthesizingWorkerAdapter:
    """Report every engine-finished id as a finished store.

    This is the worker behavior that crashes EngineCore when the scheduler
    has already freed the request: a finished_sending report with no store.
    """

    def get_finished(self, finished_req_ids: set[str]):
        return set(finished_req_ids), set()


def _tracker(
    request_id: str, num_tokens: int, num_blocks: int
) -> LMCacheMPRequestTracker:
    tracker = LMCacheMPRequestTracker.__new__(LMCacheMPRequestTracker)
    tracker.request_id = request_id
    tracker.cache_salt = ""
    tracker.all_token_ids = list(range(num_tokens))
    tracker.allocated_block_ids = list(range(num_blocks))
    tracker.num_scheduled_tokens = 0
    tracker.num_stored_blocks = 0
    tracker.num_vllm_hit_blocks = 0
    tracker.num_lmcache_hit_blocks = 0
    tracker.store_scheduled = False
    tracker.state = LMCacheMPRequestState.READY
    return tracker


def _scheduler(tracker: LMCacheMPRequestTracker | None = None):
    connector = LMCacheMPConnectorUpstream.__new__(LMCacheMPConnectorUpstream)
    connector._delay_free_req_ids = set()
    connector.request_trackers = {}
    connector.vllm_block_size = BLOCK_SIZE
    connector.scheduler_adapter = MagicMock()
    connector.scheduler_adapter.num_blocks_per_chunk.return_value = BLOCKS_IN_CHUNK
    if tracker is not None:
        connector.request_trackers[tracker.request_id] = tracker
    return connector


def _empty_output():
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        num_scheduled_tokens={},
    )


def _worker():
    worker = LMCacheMPConnectorUpstream.__new__(LMCacheMPConnectorUpstream)
    worker._delay_free_req_ids = set()
    worker._connector_metadata = None
    worker.worker_adapter = _SynthesizingWorkerAdapter()
    return worker


def test_load_only_request_does_not_delay_free():
    tracker = _tracker("load-only", num_tokens=BLOCK_SIZE, num_blocks=1)
    scheduler = _scheduler(tracker)
    output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="load-only")],
        num_scheduled_tokens={"load-only": BLOCK_SIZE},
    )

    scheduler._process_new_requests(output, LMCacheMPConnectorMetadata())

    assert tracker.store_scheduled is False
    delay_free, params = scheduler.request_finished(_Request("load-only"), [0])

    assert delay_free is False
    assert params is None
    assert scheduler.request_trackers == {}
    scheduler.scheduler_adapter.end_session.assert_called_once_with("load-only")
    metadata = scheduler.build_connector_meta(_empty_output())
    assert metadata.delay_free_req_ids == set()


def test_scheduled_store_delays_free_until_worker_reports_it():
    tracker = _tracker("stored", num_tokens=4 * BLOCK_SIZE, num_blocks=4)
    scheduler = _scheduler(tracker)
    output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="stored")],
        num_scheduled_tokens={"stored": 4 * BLOCK_SIZE},
    )
    store_metadata = LMCacheMPConnectorMetadata()

    scheduler._process_new_requests(output, store_metadata)

    assert tracker.store_scheduled is True
    assert len(store_metadata) == 1
    delay_free, _ = scheduler.request_finished(_Request("stored"), [0, 1, 2, 3])
    assert delay_free is True

    metadata = scheduler.build_connector_meta(_empty_output())
    assert metadata.delay_free_req_ids == {"stored"}
    # The ids are handed off once.
    assert scheduler.build_connector_meta(_empty_output()).delay_free_req_ids == set()

    worker = _worker()
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv(None)
    sending, receiving = worker.get_finished({"stored", "load-only"})
    assert sending == {"stored"}
    assert receiving == set()
    sending, receiving = worker.get_finished({"stored", "load-only"})
    assert sending == set()
    assert receiving == set()


def test_cached_request_store_is_tracked():
    tracker = _tracker("cached", num_tokens=4 * BLOCK_SIZE, num_blocks=0)
    scheduler = _scheduler(tracker)
    output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["cached"],
            new_block_ids=[(list(range(4)),)],
        ),
        num_scheduled_tokens={"cached": 4 * BLOCK_SIZE},
    )

    scheduler._process_cached_requests(output, LMCacheMPConnectorMetadata())

    assert tracker.store_scheduled is True
    assert scheduler.request_finished(_Request("cached"), [0, 1, 2, 3])[0] is True


def test_worker_drops_finished_sending_when_no_store_was_scheduled():
    worker = _worker()
    metadata = LMCacheMPConnectorMetadata()
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv(None)

    sending, receiving = worker.get_finished({"aborted"})

    assert sending == set()
    assert receiving == set()
