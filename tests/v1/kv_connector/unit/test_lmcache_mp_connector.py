# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import logging
import sys
import types
from typing import Any, cast
from unittest.mock import MagicMock

_STUBBED_MODULES: list[str] = []


def _set_stub(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module
        _STUBBED_MODULES.append(name)


def _install_optional_dependency_stubs() -> None:
    modules = {
        "lmcache": types.ModuleType("lmcache"),
        "lmcache.integration": types.ModuleType("lmcache.integration"),
        "lmcache.integration.vllm": types.ModuleType("lmcache.integration.vllm"),
        "lmcache.v1": types.ModuleType("lmcache.v1"),
        "lmcache.v1.multiprocess": types.ModuleType("lmcache.v1.multiprocess"),
    }
    for name, module in modules.items():
        _set_stub(name, module)

    vllm_utils = cast(Any, types.ModuleType("lmcache.integration.vllm.utils"))
    vllm_utils.mla_enabled = lambda _model_config: False
    _set_stub("lmcache.integration.vllm.utils", vllm_utils)

    lmcache_utils = cast(Any, types.ModuleType("lmcache.utils"))
    lmcache_utils.init_logger = logging.getLogger
    lmcache_utils._lmcache_nvtx_annotate = lambda func: func
    _set_stub("lmcache.utils", lmcache_utils)

    custom_types = cast(Any, types.ModuleType("lmcache.v1.multiprocess.custom_types"))

    class BlockAllocationRecord:
        pass

    custom_types.BlockAllocationRecord = BlockAllocationRecord
    _set_stub("lmcache.v1.multiprocess.custom_types", custom_types)

    _set_stub("uvloop", types.ModuleType("uvloop"))
    request_module = cast(Any, types.ModuleType("vllm.v1.request"))

    class RequestStatus(enum.Enum):
        PREEMPTED = enum.auto()

    request_module.RequestStatus = RequestStatus
    _set_stub("vllm.v1.request", request_module)

    lmcache_integration = cast(
        Any,
        types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
        ),
    )

    class LoadStoreOp:
        def __init__(self, token_ids, block_ids, start, end, **kwargs):
            self.token_ids = token_ids
            self.block_ids = block_ids
            self.start = start
            self.end = end

        def __len__(self):
            return len(self.block_ids)

    class ParallelStrategy:
        def __init__(self, *args, **kwargs):
            pass

    lmcache_integration.LMCacheMPSchedulerAdapter = object
    lmcache_integration.LMCacheMPWorkerAdapter = object
    lmcache_integration.LoadStoreOp = LoadStoreOp
    lmcache_integration.ParallelStrategy = ParallelStrategy
    _set_stub(
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration",
        lmcache_integration,
    )


try:
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
        LMCacheMPConnectorUpstream,
        LMCacheMPRequestState,
        LMCacheMPRequestTracker,
    )
except ImportError:
    sys.modules.pop(
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector",
        None,
    )
    _install_optional_dependency_stubs()
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
        LMCacheMPConnectorUpstream,
        LMCacheMPRequestState,
        LMCacheMPRequestTracker,
    )

    for module_name in reversed(_STUBBED_MODULES):
        sys.modules.pop(module_name, None)


def _make_tracker(state: LMCacheMPRequestState) -> LMCacheMPRequestTracker:
    request = MagicMock()
    request.request_id = "req-1"
    request.cache_salt = ""
    request.all_token_ids = list(range(64))
    request.block_hashes = []

    tracker = LMCacheMPRequestTracker(request)
    tracker.state = state
    tracker.num_lmcache_hit_blocks = 3
    tracker.num_vllm_hit_blocks = 1
    return tracker


def _make_connector(tracker: LMCacheMPRequestTracker) -> LMCacheMPConnectorUpstream:
    connector = object.__new__(LMCacheMPConnectorUpstream)
    connector.vllm_block_size = 16
    connector.scheduler_adapter = MagicMock()
    connector.request_trackers = {tracker.request_id: tracker}
    return connector


def test_cleanup_prefetching_tracker_releases_lookup_locks():
    tracker = _make_tracker(LMCacheMPRequestState.PREFETCHING)
    connector = _make_connector(tracker)

    connector._cleanup_request_tracker("req-1")

    connector.scheduler_adapter.cleanup_lookup_result.assert_called_once_with("req-1")
    connector.scheduler_adapter.free_lookup_locks.assert_called_once_with(
        token_ids=list(range(64)),
        start=0,
        end=48,
        request_id="req-1",
    )
    assert "req-1" not in connector.request_trackers


def test_cleanup_non_prefetching_tracker_does_not_release_lookup_locks_again():
    tracker = _make_tracker(LMCacheMPRequestState.WAITING_FOR_LOAD)
    connector = _make_connector(tracker)

    connector._cleanup_request_tracker("req-1")

    connector.scheduler_adapter.cleanup_lookup_result.assert_not_called()
    connector.scheduler_adapter.free_lookup_locks.assert_not_called()
    assert "req-1" not in connector.request_trackers
