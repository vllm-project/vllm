# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the on_new_request() scheduler hook (added in #41383).
https://github.com/vllm-project/vllm/issues/41784

Quick recap of the problem:
  - Scheduler processes the running queue first each iteration
  - Waiting loop only fires when there's leftover token budget
  - Under load the running queue fills the budget and waiting loop is skipped
  - Connector never learns about waiting requests until much later
  - Disk/remote KV fetch starts too late -> GPU stall waiting for data

on_new_request() fires in add_request() so the connector hears about
new requests immediately, not whenever the scheduler gets around to
polling the waiting queue.
"""

import contextlib
from unittest.mock import MagicMock, call

import pytest

from .utils import create_request, create_scheduler, create_vllm_config

pytestmark = pytest.mark.cpu_test


def _scheduler_with_mock_connector():
    """
    Returns a scheduler that has a MagicMock connector injected.

    We build it without kv_transfer_config so the scheduler doesn't try
    to instantiate a real connector, then slot the mock in afterwards.
    That keeps these tests runnable without NIXL/LMCache installed.
    """
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config = None
    scheduler = create_scheduler(vllm_config)
    assert scheduler.connector is None

    mock_conn = MagicMock()
    scheduler.connector = mock_conn
    return scheduler, mock_conn


def test_on_new_request_called_when_request_arrives():
    """
    Core regression test for #41784.

    connector.on_new_request() must be called the moment add_request()
    is invoked - not deferred until the scheduler polls the waiting queue.
    """
    scheduler, mock_conn = _scheduler_with_mock_connector()

    req = create_request(num_tokens=16)
    scheduler.add_request(req)

    mock_conn.on_new_request.assert_called_once_with(req)


def test_on_new_request_called_for_each_distinct_request():
    """
    Three separate requests -> three on_new_request calls, one each, in order.
    """
    scheduler, mock_conn = _scheduler_with_mock_connector()

    reqs = [create_request(num_tokens=16) for _ in range(3)]
    for r in reqs:
        scheduler.add_request(r)

    assert mock_conn.on_new_request.call_count == 3
    mock_conn.on_new_request.assert_has_calls(
        [call(r) for r in reqs], any_order=False
    )


def test_no_duplicate_notify_on_streaming_update():
    """
    A streaming chunk update re-uses the same request_id. on_new_request
    should not fire again - the connector already knows the request
    and a second call could trigger a redundant disk fetch.
    """
    scheduler, mock_conn = _scheduler_with_mock_connector()

    req = create_request(num_tokens=16)
    scheduler.add_request(req)
    count_after_first = mock_conn.on_new_request.call_count

    # Simulate duplicate add (streaming update path hits the `if existing` branch)
    with contextlib.suppress(Exception):  # scheduler may raise on true dup
        scheduler.add_request(req)

    assert mock_conn.on_new_request.call_count == count_after_first


def test_no_connector_does_not_crash():
    """
    When no KV connector is configured (pure local serving), add_request
    must work fine. The call is guarded by `if self.connector is not None`.
    """
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config = None
    scheduler = create_scheduler(vllm_config)

    assert scheduler.connector is None

    req = create_request(num_tokens=16)
    scheduler.add_request(req)  # must not raise


def test_on_new_request_not_called_on_internal_reenqueue():
    """
    When the scheduler preempts a request and re-enqueues it internally
    via _enqueue_waiting_request(), on_new_request must NOT fire again.

    That path bypasses add_request() entirely - the connector already
    knows the request. A double-call there would be wrong.
    """
    scheduler, mock_conn = _scheduler_with_mock_connector()

    req = create_request(num_tokens=16)
    scheduler.add_request(req)
    assert mock_conn.on_new_request.call_count == 1

    # Internal re-enqueue (what preemption does - goes straight to
    # _enqueue_waiting_request, not through add_request)
    scheduler._enqueue_waiting_request(req)

    assert mock_conn.on_new_request.call_count == 1
