# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LMCacheMPConnector.maybe_prefetch_request.

The full connector requires an LMCache backend and a running scheduler
adapter. These tests instead bind the methods under test to a MagicMock
with `spec=LMCacheMPConnector`, so we exercise the real prefetch logic
against a stubbed scheduler adapter without spinning up LMCache.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("lmcache")

from tests.v1.core.utils import create_requests  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from vllm.v1.request import RequestStatus  # noqa: E402


def _bind(connector: MagicMock, *method_names: str) -> None:
    """Bind real `LMCacheMPConnector` methods onto a MagicMock instance."""
    for name in method_names:
        method = getattr(LMCacheMPConnector, name)
        setattr(connector, name, method.__get__(connector, LMCacheMPConnector))


def _make_connector() -> MagicMock:
    connector = MagicMock(spec=LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = MagicMock()
    _bind(
        connector,
        "maybe_prefetch_request",
        "_get_or_create_request_tracker",
        "_maybe_submit_lookup_request",
    )
    return connector


def test_maybe_prefetch_request_submits_lookup_for_fresh_waiting_request():
    """A plain WAITING request with no computed tokens should trigger a
    lookup submission and have its tracker created."""
    connector = _make_connector()
    request = create_requests(num_requests=1, num_tokens=10)[0]
    assert request.status == RequestStatus.WAITING
    assert request.num_computed_tokens == 0

    submitted = connector.maybe_prefetch_request(request)

    assert submitted is True
    assert request.request_id in connector.request_trackers
    submit = connector.scheduler_adapter.maybe_submit_lookup_request
    submit.assert_called_once()
    call = submit.call_args
    assert call.args[0] == request.request_id
    assert call.kwargs["token_ids"] == list(request.all_token_ids)
    # cache_salt comes from the tracker; defaults to "" when request has none.
    assert call.kwargs["cache_salt"] == (request.cache_salt or "")


@pytest.mark.parametrize(
    "skip_status",
    [
        RequestStatus.PREEMPTED,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        RequestStatus.RUNNING,
    ],
)
def test_maybe_prefetch_request_skips_non_plain_waiting(skip_status):
    """The hook must bail (and not submit a lookup) for any request that
    is not in the plain WAITING state."""
    connector = _make_connector()
    request = create_requests(num_requests=1, num_tokens=10)[0]
    request.status = skip_status

    submitted = connector.maybe_prefetch_request(request)

    assert submitted is False
    connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()
    assert request.request_id not in connector.request_trackers


def test_maybe_prefetch_request_skips_when_partially_computed():
    """Already-partially-computed requests should not have prefetch hints
    issued, even if their status is still WAITING."""
    connector = _make_connector()
    request = create_requests(num_requests=1, num_tokens=10)[0]
    request.num_computed_tokens = 4

    submitted = connector.maybe_prefetch_request(request)

    assert submitted is False
    connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()


def test_maybe_prefetch_request_is_idempotent_for_repeat_calls():
    """Calling the hook twice for the same fresh waiting request reuses the
    existing tracker and just re-submits the lookup -- the underlying
    `scheduler_adapter.maybe_submit_lookup_request` is itself the
    deduplication boundary on the LMCache side."""
    connector = _make_connector()
    request = create_requests(num_requests=1, num_tokens=10)[0]

    assert connector.maybe_prefetch_request(request) is True
    tracker = connector.request_trackers[request.request_id]

    assert connector.maybe_prefetch_request(request) is True
    # Same tracker instance is retained on the second call.
    assert connector.request_trackers[request.request_id] is tracker
    assert connector.scheduler_adapter.maybe_submit_lookup_request.call_count == 2
