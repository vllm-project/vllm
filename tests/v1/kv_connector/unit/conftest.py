# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Conftest for lmcache_mp_connector tests.
Patches lmcache modules in sys.modules so the connector module can be imported
without the lmcache package installed.

This must run at module load time (before any test imports the connector).
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


def _mock_lmcache():
    """Patch all lmcache modules before they are imported."""

    # Stub for LMCacheMPSchedulerAdapter
    class MockSchedulerAdapter:
        def __init__(self, **kwargs):
            pass

        def num_blocks_per_chunk(self):
            return 1

        def maybe_submit_lookup_request(self, *args, **kwargs):
            pass

        def check_lookup_result(self, *args, **kwargs):
            return None

        def cleanup_lookup_result(self, *args, **kwargs):
            pass

        def free_lookup_locks(self, *args, **kwargs):
            pass

    # Stub for LMCacheMPWorkerAdapter
    class MockWorkerAdapter:
        def __init__(self, **kwargs):
            pass

    # Stub for ParallelStrategy
    class MockParallelStrategy:
        def __init__(self, *args, **kwargs):
            self.kv_world_size = 1
            self.kv_worker_id = 0
            self.tp_size = 1

    # Stub for LoadStoreOp
    class MockLoadStoreOp:
        pass

    # Create the module hierarchy with proper attribute assignments
    def _make_mod(name, **attrs):
        mod = sys.modules.get(name)
        if mod is None:
            mod = MagicMock()
            mod.__name__ = name
            sys.modules[name] = mod
        for k, v in attrs.items():
            setattr(mod, k, v)
        return mod

    # lmcache.integration.vllm.vllm_multi_process_adapter
    _make_mod(
        "lmcache.integration.vllm.vllm_multi_process_adapter",
        LMCacheMPSchedulerAdapter=MockSchedulerAdapter,
        LMCacheMPWorkerAdapter=MockWorkerAdapter,
        LoadStoreOp=MockLoadStoreOp,
        ParallelStrategy=MockParallelStrategy,
    )

    # lmcache.integration.vllm.utils
    _make_mod("lmcache.integration.vllm.utils", mla_enabled=lambda *a: False)

    # lmcache.utils
    _make_mod("lmcache.utils", init_logger=lambda n: MagicMock())

    # lmcache.v1.multiprocess.custom_types
    class MockBlockAllocationRecord:
        pass

    class MockRequestAllocationRecord:
        pass

    _make_mod(
        "lmcache.v1.multiprocess.custom_types",
        BlockAllocationRecord=MockBlockAllocationRecord,
        RequestAllocationRecord=MockRequestAllocationRecord,
    )


_mock_lmcache()

# Connector imports must come after _mock_lmcache() sets up sys.modules.
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnectorUpstream,
)


@pytest.fixture
def connector():
    """Create a connector with mocked dependencies."""
    mock_adapter = MagicMock()
    mock_adapter.num_blocks_per_chunk.return_value = 1

    vllm_config = MagicMock()
    vllm_config.kv_transfer_config.get_from_extra_config.side_effect = (
        lambda key, default=None: {
            "lmcache.mp.host": "tcp://localhost",
            "lmcache.mp.port": 5555,
            "lmcache.mp.mq_timeout": 300.0,
            "lmcache.mp.heartbeat_interval": 10.0,
        }.get(key, default)
    )
    vllm_config.cache_config.block_size = 16
    vllm_config.parallel_config.world_size = 1
    vllm_config.parallel_config.rank = 0
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.pipeline_parallel_size = 1

    kv_cache_config = MagicMock()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector."
        "create_scheduler_adapter",
        return_value=mock_adapter,
    ):
        connector = LMCacheMPConnectorUpstream(
            vllm_config=vllm_config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=kv_cache_config,
        )

    connector._mock_adapter = mock_adapter
    return connector


@pytest.fixture
def mock_blocks():
    """Create mock KVCacheBlocks with 10 blocks.
    NOTE: get_block_ids must return a tuple, not a list.
    """
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (list(range(10)),)
    return blocks


@pytest.fixture
def mock_request_factory():
    """Factory for creating mock Requests with a custom request_id."""

    def _create(request_id="test-request"):
        request = MagicMock()
        request.request_id = request_id
        request.all_token_ids = list(range(256))
        request.block_hashes = [b"hash%d" % i for i in range(16)]
        request.cache_salt = ""
        request.status = MagicMock()
        request.status.__eq__ = lambda self, other: False  # Not PREEMPTED
        return request

    return _create


@pytest.fixture
def setup_tracker_with_hits():
    """Factory for setting up a tracker with LMCache hit blocks."""

    def _setup(connector, request, num_lmcache_blocks=10):
        tracker = connector._get_or_create_request_tracker(request)
        tracker.num_lmcache_hit_blocks = num_lmcache_blocks
        tracker.num_vllm_hit_blocks = 0
        return tracker

    return _setup
