# SPDX-License-Identifier: Apache-2.0
"""Regression test for issue #3104.

In TP=1 non-MP vLLM mode, scheduler and worker are initialized in the
same Python process and both construct ``InternalAPIServer``. Prior to
the fix, the module-level ``app`` was shared across all instances, so
the later-initialized scheduler (whose ``lmcache_engine`` is ``None``)
overwrote the worker's ``app.state.lmcache_adapter``, causing every
cache endpoint on the worker port to return 503.

This test verifies that each ``InternalAPIServer`` owns its own
FastAPI app and the adapters do not clobber each other.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.metadata import LMCacheMetadata


def _make_config(port_start: int) -> LMCacheEngineConfig:
    cfg = LMCacheEngineConfig.from_defaults(chunk_size=256, local_cpu=True)
    cfg.internal_api_server_enabled = True
    cfg.internal_api_server_port_start = port_start
    cfg.internal_api_server_socket_path_prefix = None
    return cfg


def _make_worker_manager(cfg: LMCacheEngineConfig) -> MagicMock:
    """Mock LMCacheManager representing a worker (engine present)."""
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=None,
        kv_shape=(1, 2, 256, 8, 128),
    )
    engine = MagicMock(spec=LMCacheEngine)
    engine.config = cfg
    engine.metadata = metadata
    engine.clear.return_value = 7

    manager = MagicMock()
    manager.lmcache_engine = engine
    manager.config = cfg
    return manager


def _make_scheduler_manager(cfg: LMCacheEngineConfig) -> MagicMock:
    """Mock LMCacheManager representing a scheduler (no engine)."""
    manager = MagicMock()
    manager.lmcache_engine = None
    manager.config = cfg
    return manager


def test_scheduler_does_not_overwrite_worker_adapter() -> None:
    """Scheduler initialized after worker must not break worker endpoints."""
    cfg = _make_config(port_start=17000)

    worker_manager = _make_worker_manager(cfg)
    worker_server = InternalAPIServer(worker_manager)

    # Scheduler comes up after worker in TP=1 non-MP mode.
    scheduler_manager = _make_scheduler_manager(cfg)
    scheduler_server = InternalAPIServer(scheduler_manager)

    # Each server must own a distinct FastAPI app.
    assert worker_server.app is not scheduler_server.app

    # Worker's adapter is still its own manager, not overwritten by
    # the scheduler's manager (which has lmcache_engine=None).
    assert worker_server.app.state.lmcache_adapter is worker_manager
    assert scheduler_server.app.state.lmcache_adapter is scheduler_manager

    # Worker cache endpoint works (returns 200 instead of 503).
    with TestClient(worker_server.app) as client:
        response = client.delete("/cache/clear")
    assert response.status_code == 200
    assert response.json()["num_removed"] == 7

    # Scheduler endpoint correctly reports engine unavailable (503),
    # because the scheduler genuinely has no engine.
    with TestClient(scheduler_server.app) as client:
        response = client.delete("/cache/clear")
    assert response.status_code == 503


def test_worker_after_scheduler_still_works() -> None:
    """Reverse order: worker initialized after scheduler also stays healthy."""
    cfg = _make_config(port_start=17100)

    scheduler_manager = _make_scheduler_manager(cfg)
    scheduler_server = InternalAPIServer(scheduler_manager)

    worker_manager = _make_worker_manager(cfg)
    worker_server = InternalAPIServer(worker_manager)

    assert worker_server.app is not scheduler_server.app
    assert worker_server.app.state.lmcache_adapter is worker_manager
    assert scheduler_server.app.state.lmcache_adapter is scheduler_manager

    with TestClient(worker_server.app) as client:
        response = client.delete("/cache/clear")
    assert response.status_code == 200
    assert response.json()["num_removed"] == 7
