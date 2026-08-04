# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``WAIT_PREFETCH_STATUS`` request handler path.

The prefetch-controller tests cover the condition-variable wait in isolation;
these cover the ``LookupModule`` handler that ``WAIT_PREFETCH_STATUS`` dispatches
to, i.e. the ``wait_prefetch_status -> query_prefetch_status`` path: count
computation, event emission, and exactly-once job consumption. The storage
manager is mocked, so no GPU or native bitmap is needed.
"""

# Standard
from unittest import mock
import threading

# First Party
from lmcache.v1.multiprocess.modules.lookup import LookupModule, _PrefetchJob


def _make_ctx(wait_result=True, found=None):
    storage_manager = mock.Mock()
    storage_manager.wait_prefetch_status.return_value = wait_result
    storage_manager.query_prefetch_status.return_value = found
    ctx = mock.Mock()
    ctx.storage_manager = storage_manager
    ctx.event_bus = mock.Mock()
    ctx.chunk_size = 256
    return ctx


def _make_module(ctx):
    # Bypass __init__ (which wires up otel metrics needing a full context); the
    # handler methods only touch _ctx, _prefetch_jobs, and _prefetch_job_lock.
    module = object.__new__(LookupModule)
    module._ctx = ctx
    module._prefetch_jobs = {}
    module._prefetch_job_lock = threading.Lock()
    return module


def test_wait_prefetch_status_returns_count_and_consumes_job():
    found = mock.Mock()
    found.count_leading_ones.return_value = 8
    ctx = _make_ctx(wait_result=True, found=found)
    module = _make_module(ctx)
    module._prefetch_jobs["req"] = _PrefetchJob(
        handle=mock.sentinel.handle,
        world_size=2,
        request_id="req",
        requested_tokens=512,
    )

    # wait succeeds -> query returns the bitmap -> count_leading_ones() // world_size.
    assert module.wait_prefetch_status("req", timeout=1.0) == 4
    ctx.storage_manager.wait_prefetch_status.assert_called_once_with(
        mock.sentinel.handle, 1.0
    )
    ctx.event_bus.publish.assert_called_once()
    # Exactly-once: the job is removed after a non-None result.
    assert "req" not in module._prefetch_jobs


def test_wait_prefetch_status_timeout_returns_none_and_keeps_job():
    ctx = _make_ctx(wait_result=False)
    module = _make_module(ctx)
    job = _PrefetchJob(
        handle=mock.sentinel.handle,
        world_size=1,
        request_id="req",
        requested_tokens=0,
    )
    module._prefetch_jobs["req"] = job

    assert module.wait_prefetch_status("req", timeout=0.5) is None
    ctx.storage_manager.query_prefetch_status.assert_not_called()
    # Job is kept so a later wait/query can still resolve it.
    assert module._prefetch_jobs["req"] is job


def test_wait_prefetch_status_unknown_request_returns_zero():
    module = _make_module(_make_ctx())
    assert module.wait_prefetch_status("missing", timeout=1.0) == 0
