# SPDX-License-Identifier: Apache-2.0
"""Regression: HTTP-layer passthroughs dropped by the engine refactor."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.server import MPCacheServer


def test_storage_manager_returns_context_storage_manager() -> None:
    sm = MagicMock(name="storage_manager")
    ctx = MagicMock()
    ctx.storage_manager = sm

    engine = MPCacheServer(ctx, modules=[])
    assert engine.storage_manager is sm


def test_cache_contexts_unwraps_entries_from_gpu_transfer_module() -> None:
    gpu0, gpu1 = MagicMock(name="gpu_ctx_0"), MagicMock(name="gpu_ctx_1")
    gpu_transfer = MagicMock(spec=LMCacheDrivenTransferModule)
    gpu_transfer.context_entries_snapshot.return_value = {
        0: ContextEntry(cache_context=gpu0, model_name="m", world_size=1),
        7: ContextEntry(cache_context=gpu1, model_name="m", world_size=1),
    }

    engine = MPCacheServer(MagicMock(), modules=[MagicMock(), gpu_transfer])
    # Values must be unwrapped GPUCacheContexts.
    assert engine.cache_contexts == {0: gpu0, 7: gpu1}


def test_cache_contexts_returns_none_in_engine_driven_mode() -> None:
    engine = MPCacheServer(MagicMock(), modules=[MagicMock()])
    assert engine.cache_contexts is None


def test_clear_delegates_to_management_module() -> None:
    mgmt = MagicMock(spec=ManagementModule)
    engine = MPCacheServer(MagicMock(), modules=[MagicMock(), mgmt])
    engine.clear()
    mgmt.clear.assert_called_once_with()


def test_clear_raises_without_management_module() -> None:
    engine = MPCacheServer(MagicMock(), modules=[])
    with pytest.raises(RuntimeError, match="no ManagementModule"):
        engine.clear()
