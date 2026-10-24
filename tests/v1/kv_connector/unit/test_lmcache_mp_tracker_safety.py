# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that request_finished handles absent trackers gracefully.

A request aborted before scheduling (abort_immediately path) never goes
through get_num_new_matched_tokens, so no tracker is created. The
request_finished hook must tolerate this instead of asserting.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_test

# The connector module has a hard dependency on the lmcache package (even its
# except-ImportError fallback re-imports lmcache submodules). When lmcache
# is not installed we stub the namespace so the module can be imported.
_need_stubs = "lmcache" not in sys.modules
if _need_stubs:
    _stub_names = [
        "lmcache",
        "lmcache.integration",
        "lmcache.integration.vllm",
        "lmcache.integration.vllm.utils",
        "lmcache.integration.vllm.vllm_multi_process_adapter",
        "lmcache.utils",
        "lmcache.v1",
        "lmcache.v1.multiprocess",
        "lmcache.v1.multiprocess.custom_types",
    ]
    for _name in _stub_names:
        _mod = ModuleType(_name)
        # Make parent packages recognise children.
        _parts = _name.rsplit(".", 1)
        if len(_parts) == 2 and _parts[0] in sys.modules:
            setattr(sys.modules[_parts[0]], _parts[1], _mod)
        sys.modules[_name] = _mod

    # Provide the symbols that the module imports at top level.
    sys.modules["lmcache.integration.vllm.utils"].mla_enabled = (  # type: ignore[attr-defined]
        lambda *_a, **_kw: False
    )
    sys.modules["lmcache.utils"].init_logger = MagicMock()  # type: ignore[attr-defined]

    # Sentinel classes so `isinstance` / `spec=` work minimally.
    _ct = sys.modules["lmcache.v1.multiprocess.custom_types"]
    _ct.BlockAllocationRecord = type("BlockAllocationRecord", (), {})  # type: ignore[attr-defined]
    _ct.RequestAllocationRecord = _ct.BlockAllocationRecord  # type: ignore[attr-defined]

    _adapter = sys.modules["lmcache.integration.vllm.vllm_multi_process_adapter"]
    _adapter.LMCacheMPSchedulerAdapter = type("LMCacheMPSchedulerAdapter", (), {})  # type: ignore[attr-defined]
    _adapter.LMCacheMPWorkerAdapter = type("LMCacheMPWorkerAdapter", (), {})  # type: ignore[attr-defined]
    _adapter.LoadStoreOp = type("LoadStoreOp", (), {})  # type: ignore[attr-defined]
    _adapter.ParallelStrategy = type("ParallelStrategy", (), {})  # type: ignore[attr-defined]

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnectorUpstream,
    LMCacheMPRequestTracker,
)


def _make_connector(block_size: int = 16) -> LMCacheMPConnectorUpstream:
    """Build a minimal connector with only the fields request_finished needs."""
    conn = object.__new__(LMCacheMPConnectorUpstream)
    conn.request_trackers = {}
    conn.vllm_block_size = block_size
    conn.scheduler_adapter = MagicMock()
    return conn


def _make_request(
    request_id: str = "req-1",
    kv_transfer_params: dict | None = None,
) -> MagicMock:
    req = MagicMock()
    req.request_id = request_id
    req.kv_transfer_params = kv_transfer_params
    return req


def test_request_finished_no_tracker_no_crash():
    """Aborted-before-scheduled request with the trigger key must not crash."""
    conn = _make_connector()
    req = _make_request(
        kv_transfer_params={
            "do_remote_prefill": True,
            "num_lmcache_extra_cached_tokens": 0,
        },
    )

    delay_free, return_params = conn.request_finished(req, [])

    assert delay_free is True
    assert isinstance(return_params, dict)
    assert "num_lmcache_extra_cached_tokens" not in return_params


def test_request_finished_with_tracker_populates_key():
    """When a tracker exists, the extra-cached-tokens key is populated."""
    conn = _make_connector(block_size=16)
    req = _make_request(
        kv_transfer_params={
            "do_remote_prefill": True,
            "num_lmcache_extra_cached_tokens": 0,
        },
    )

    tracker = MagicMock(spec=LMCacheMPRequestTracker)
    tracker.num_lmcache_hit_blocks = 5
    tracker.num_vllm_hit_blocks = 2
    conn.request_trackers[req.request_id] = tracker

    delay_free, return_params = conn.request_finished(req, [])

    assert delay_free is True
    assert return_params["num_lmcache_extra_cached_tokens"] == 3 * 16


def test_request_finished_no_trigger_key():
    """Without the trigger key, the tracker lookup is never attempted."""
    conn = _make_connector()
    req = _make_request(
        kv_transfer_params={"do_remote_prefill": True},
    )

    delay_free, return_params = conn.request_finished(req, [])

    assert delay_free is True
    assert isinstance(return_params, dict)
    assert "num_lmcache_extra_cached_tokens" not in return_params


def test_request_finished_no_params():
    """A request with no kv_transfer_params returns None for return_params."""
    conn = _make_connector()
    req = _make_request(kv_transfer_params=None)

    delay_free, return_params = conn.request_finished(req, [])

    assert delay_free is True
    assert return_params is None
