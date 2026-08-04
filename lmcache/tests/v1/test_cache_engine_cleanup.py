# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for LMCacheEngine.cleanup_memory_objs.

Some storage backends (LocalCPU, LocalDisk, P2P, Maru) pin the MemoryObj
they return via async prefetch, while others (Remote, Nixl, plugin tiers)
do not. cleanup_memory_objs must skip unpin for the latter to avoid
driving pin_count below zero.
"""

# Standard
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventStatus
from lmcache.v1.pin_monitor import PinMonitor

# Local
from .utils import create_test_memory_obj


@pytest.fixture
def pin_monitor() -> Generator[None, None, None]:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, lmcache_instance_id="test_cleanup"
    )
    PinMonitor.GetOrCreate(config)
    yield
    PinMonitor.DestroyInstance()


def test_cleanup_memory_objs_handles_mixed_pin_state(
    caplog: pytest.LogCaptureFixture, pin_monitor: Any
) -> None:
    """
    Cleanup must unpin chunks that were pinned during prefetch and
    skip unpin for chunks that were not, leaving every chunk with a
    non-negative pin_count and no "Double unpin" warning.
    """
    pinned_obj = create_test_memory_obj()
    pinned_obj.pin()
    pinned_obj.ref_count_up()
    assert pinned_obj.metadata.pin_count == 1

    nonpinned_obj = create_test_memory_obj()
    assert nonpinned_obj.metadata.pin_count == 0

    future = MagicMock()
    future.result.return_value = [
        [(None, pinned_obj)],
        [(None, nonpinned_obj)],
    ]

    engine = SimpleNamespace(event_manager=MagicMock())
    engine.event_manager.get_event_status.return_value = EventStatus.DONE
    engine.event_manager.pop_event.return_value = future

    caplog.set_level(logging.WARNING, logger="lmcache")

    LMCacheEngine.cleanup_memory_objs(engine, "test_lookup")  # type: ignore[arg-type]

    assert pinned_obj.metadata.pin_count == 0
    assert nonpinned_obj.metadata.pin_count == 0
    assert "Double unpin" not in caplog.text
    assert "is negative" not in caplog.text

    pinned_obj.ref_count_down()
