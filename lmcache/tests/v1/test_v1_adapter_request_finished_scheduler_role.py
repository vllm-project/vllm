# SPDX-License-Identifier: Apache-2.0
"""Regression test for LMCache#3337.

``request_finished`` is a Scheduler-side connector API. In vLLM v1's
split-process architecture:

- The Scheduler typically does **not** initialize ``self.lmcache_engine``;
  only the Worker role builds it by default (the Scheduler also builds
  it when ``enable_scheduler_bypass_lookup`` is set).
- The Scheduler **does** own ``self.lookup_client`` -- it is created
  only on the Scheduler role (see
  ``vllm_service_factory.maybe_create_lookup_client``).

Previously the ``FINISHED_ABORTED`` cleanup path asserted both
``self.lmcache_engine is not None`` and ``self.lookup_client is not None``.
Under concurrent abort cleanups the first assert fired as an unhandled
``AssertionError`` and killed the EngineCore process for every connected
user.

The fix replaces both asserts with ``if-warn-skip`` so the engine stays
alive and only the affected request's cleanup is dropped. It also moves
the async lookup cancel out of the engine-None branch -- since the
Scheduler owns ``lookup_client`` independently of the engine, the lookup
cancel must still run for Scheduler-side aborts to avoid leaking
in-flight async lookups.

This test locks in:

1. Scheduler default (engine=None, lookup_client=None, async_loading=False)
   + abort -> no crash, warning emitted, no engine attribute is read.
2. Scheduler default (engine=None) + ``async_loading=True`` +
   lookup_client present + abort -> ``lookup_client.cancel_lookup``
   still runs (resource-leak regression guard).
3. Engine initialized + abort -> ``storage_manager.cancel_request`` runs.
4. Engine initialized + ``async_loading=True`` + lookup_client present
   + abort -> both cancels run.
5. ``async_loading=True`` + lookup_client missing + abort -> warns and
   skips the lookup cancel instead of crashing.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
import logging

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
# Third Party (after importorskip)
from vllm.v1.request import RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

_ADAPTER_LOGGER_NAME = "lmcache.integration.vllm.vllm_v1_adapter"


@contextmanager
def _capture_adapter_warnings() -> Iterator[list[logging.LogRecord]]:
    """Capture WARNING records emitted by the adapter logger.

    lmcache's ``init_logger`` sets ``propagate = False`` on the adapter
    logger, so its records never reach pytest's ``caplog`` (which attaches a
    handler to the root logger). Attach a handler directly to the named
    logger instead -- the convention used across the v1 adapter tests -- and
    force WARNING level so a higher ``LMCACHE_LOG_LEVEL`` cannot filter the
    records out before the handler sees them.
    """
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    logger = logging.getLogger(_ADAPTER_LOGGER_NAME)
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


class _FakeStorageManager:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    def cancel_request(self, req_id: str) -> None:
        self.cancelled.append(req_id)


class _FakeEngine:
    def __init__(self) -> None:
        self.storage_manager = _FakeStorageManager()


class _FakeLookupClient:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    def cancel_lookup(self, req_id: str) -> None:
        self.cancelled.append(req_id)


def _make_aborted_request(req_id: str) -> SimpleNamespace:
    """Build the minimum request shape ``request_finished`` reads."""
    return SimpleNamespace(
        request_id=req_id,
        status=RequestStatus.FINISHED_ABORTED,
    )


def _make_connector(
    *,
    engine: object,
    lookup_client: object,
    async_loading: bool,
) -> LMCacheConnectorV1Impl:
    """Bypass ``__init__`` and pin only the attributes ``request_finished``
    reads, so the test exercises the abort cleanup path in isolation.

    ``lmcache_engine`` and ``lookup_client`` are read-only properties on
    ``LMCacheConnectorV1Impl`` that delegate to ``self._manager``, so the
    fixture installs a fake manager rather than assigning to those
    properties directly.
    """
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=engine,
        lookup_client=lookup_client,
    )
    connector.async_loading = async_loading
    connector.use_layerwise = False
    connector.config = SimpleNamespace(
        get_extra_config_value=lambda key, default: default
    )
    connector._request_trackers = {}
    return connector


def test_request_finished_scheduler_default_engine_none_skips_cleanup() -> None:
    """Scheduler default config: ``lmcache_engine`` is None, abort must
    not crash the EngineCore.

    Regression for https://github.com/LMCache/LMCache/issues/3337.
    """
    connector = _make_connector(engine=None, lookup_client=None, async_loading=False)
    request = _make_aborted_request("req-scheduler-default-abort")

    with _capture_adapter_warnings() as records:
        delay_free, return_params = connector.request_finished(request, [0, 1])

    assert delay_free is False
    assert return_params is None

    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert any(
        "req-scheduler-default-abort" in r.getMessage()
        and "lmcache_engine" in r.getMessage()
        for r in warnings
    ), (
        "Expected a warning naming the request id and lmcache_engine; "
        f"got {[r.getMessage() for r in warnings]}"
    )


def test_request_finished_scheduler_default_async_loading_runs_lookup_cancel() -> None:
    """Scheduler default (engine=None) + ``async_loading=True`` +
    lookup_client present: ``lookup_client.cancel_lookup`` must still
    run. Resource-leak regression guard -- the Scheduler owns
    ``lookup_client`` even when it does not build an engine, so the
    lookup cancel cannot be nested inside the engine-not-None branch.
    """
    lookup_client = _FakeLookupClient()
    connector = _make_connector(
        engine=None, lookup_client=lookup_client, async_loading=True
    )
    request = _make_aborted_request("req-scheduler-async-abort")

    connector.request_finished(request, [0, 1])

    assert lookup_client.cancelled == ["req-scheduler-async-abort"], (
        "Scheduler-side abort must still cancel async lookups even when "
        "lmcache_engine is None; otherwise async lookups leak."
    )


def test_request_finished_engine_initialized_runs_storage_manager_cancel() -> None:
    """Engine-initialized path (Worker, or Scheduler with bypass lookup):
    ``cancel_request`` must still fire on abort. Guards against the fix
    accidentally short-circuiting the happy path.
    """
    engine = _FakeEngine()
    connector = _make_connector(engine=engine, lookup_client=None, async_loading=False)
    request = _make_aborted_request("req-engine-abort")

    delay_free, return_params = connector.request_finished(request, [0, 1])

    assert delay_free is False
    assert return_params is None
    assert engine.storage_manager.cancelled == ["req-engine-abort"]


def test_request_finished_engine_with_async_loading_runs_both_cancels() -> None:
    """Engine-initialized + ``async_loading=True`` + ``lookup_client``
    present: both ``storage_manager.cancel_request`` and
    ``lookup_client.cancel_lookup`` must fire on abort.
    """
    engine = _FakeEngine()
    lookup_client = _FakeLookupClient()
    connector = _make_connector(
        engine=engine, lookup_client=lookup_client, async_loading=True
    )
    request = _make_aborted_request("req-async-abort")

    connector.request_finished(request, [0, 1])

    assert engine.storage_manager.cancelled == ["req-async-abort"]
    assert lookup_client.cancelled == ["req-async-abort"]


def test_request_finished_async_loading_without_lookup_client_skips_cancel() -> None:
    """``async_loading=True`` + ``lookup_client`` missing: the second
    assert in the original block would have crashed here too. The fix
    must warn and skip, matching the fail-soft contract.
    """
    engine = _FakeEngine()
    connector = _make_connector(engine=engine, lookup_client=None, async_loading=True)
    request = _make_aborted_request("req-no-lookup-abort")

    with _capture_adapter_warnings() as records:
        connector.request_finished(request, [0, 1])

    # storage_manager.cancel_request still fired (only the lookup path skipped)
    assert engine.storage_manager.cancelled == ["req-no-lookup-abort"]

    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert any(
        "req-no-lookup-abort" in r.getMessage() and "lookup_client" in r.getMessage()
        for r in warnings
    ), (
        "Expected a warning naming the request id and lookup_client; "
        f"got {[r.getMessage() for r in warnings]}"
    )
