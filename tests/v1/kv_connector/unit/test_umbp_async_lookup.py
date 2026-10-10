# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.umbp_test_utils import (
    _kv_cache_config,
    _SchedulerHandle,
    _vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    LoadSpec,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)


def test_scheduler_async_lookup_defers_then_returns_hit(monkeypatch):
    gate = threading.Event()
    key_calls = 0
    context_builds = 0
    original_keys = BlockIdentityCodec.keys_for_topology
    original_context = UMBPStoreConnectorScheduler._build_lookup_context

    def build_context(*args, **kwargs):
        nonlocal context_builds
        context_builds += 1
        return original_context(*args, **kwargs)

    def counted_keys(*args, **kwargs):
        nonlocal key_calls
        key_calls += 1
        return original_keys(*args, **kwargs)

    monkeypatch.setattr(BlockIdentityCodec, "keys_for_topology", counted_keys)
    monkeypatch.setattr(
        UMBPStoreConnectorScheduler, "_build_lookup_context", build_context
    )

    class _GatedHandle(_SchedulerHandle):
        def lookup(self, keys):
            gate.wait(timeout=5)
            return [True] * len(keys)

    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {
                "mode": "embedded",
                "lookup_async": True,
                "load_async": False,
            }
        ),
        _kv_cache_config(),
        _GatedHandle([]),
        BlockIdentityCodec(UMBPNamespace("async")),
    )
    request = SimpleNamespace(
        request_id="async",
        num_tokens=32,
        block_hashes=[b"a", b"b"],
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    for _ in range(3):
        assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    assert key_calls == 1
    gate.set()
    result = (None, False)
    for _ in range(100):
        result = scheduler.get_num_new_matched_tokens(request, 0)
        if result != (None, False):
            break
        time.sleep(0.01)
    scheduler.close()

    # The final prompt token must still execute, leaving one reusable block.
    assert result == (16, False)
    assert key_calls == 1
    assert context_builds == 1


@pytest.mark.parametrize("change", ["local", "length", "hash", "request"])
def test_async_lookup_discards_results_for_changed_request(monkeypatch, change):
    """A completed old query must not be interpreted using a new prefix."""
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"lookup_async": True}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("changed-query")),
    )
    queries = []

    def submit(fn, keys):
        future: Future[list[bool]] = Future()
        queries.append((keys, future))
        return future

    monkeypatch.setattr(scheduler._lookup_executor, "submit", submit)
    request = SimpleNamespace(
        request_id="r", num_tokens=49, block_hashes=[b"a", b"b", b"c"]
    )
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
        queries[0][1].set_result([True] * len(queries[0][0]))
        local_tokens = 0
        if change == "local":
            local_tokens = 16
        elif change == "length":
            request.num_tokens = 65
            request.block_hashes.append(b"d")
        elif change == "hash":
            request.block_hashes[-1] = b"different"
        else:
            request = SimpleNamespace(**vars(request))
        assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (
            None,
            False,
        )
        assert len(queries) == 2
        queries[1][1].set_result([False] * len(queries[1][0]))
        assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (0, False)
    finally:
        scheduler.close()


def test_scheduler_reset_clears_pending_lookup_state(monkeypatch):
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"mode": "embedded", "lookup_async": True}),
        _kv_cache_config(),
        _SchedulerHandle([]),
        BlockIdentityCodec(UMBPNamespace("reset")),
    )
    future: Future[list[bool]] = Future()
    monkeypatch.setattr(scheduler._lookup_executor, "submit", lambda *a: future)
    request = SimpleNamespace(request_id="req", num_tokens=32, block_hashes=[b"a"])
    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    scheduler._load_specs["req"] = LoadSpec(0, 16)

    assert scheduler.reset_store()
    assert future.cancelled()
    assert scheduler._pending_lookups == {}
    assert scheduler._load_specs == {}
    scheduler.close()


@pytest.mark.parametrize("failure", [TimeoutError("lookup timeout"), [True, False]])
def test_async_lookup_failure_recomputes_and_allows_retry(failure):
    class _Handle(_SchedulerHandle):
        def lookup(self, keys):
            if isinstance(self.hits, Exception):
                raise self.hits
            return self.hits

    handle = _Handle(failure)
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config({"lookup_async": True, "load_async": False}),
        _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("lookup-retry")),
    )
    request = SimpleNamespace(
        request_id="retry", num_tokens=32, block_hashes=[b"a", b"b"]
    )
    try:
        for expected in ((0, False), (16, False)):
            deadline = time.monotonic() + 5
            result = scheduler.get_num_new_matched_tokens(request, 0)
            while result == (None, False) and time.monotonic() < deadline:
                time.sleep(0.01)
                result = scheduler.get_num_new_matched_tokens(request, 0)
            assert result == expected
            handle.hits = [True]
    finally:
        scheduler.close()
