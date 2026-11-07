# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_config import (
    _make_mamba_hybrid_kv_cache_config,
    _make_vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.state import (
    SchedulerOffloadConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers import (
    OnComputeStoreHandler,
    OnHandoffStoreHandler,
    StoreTriggerHandler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.factory import (  # noqa: E501
    StoreHandlerFactory,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import StoreTrigger


def _create_handlers(
    kv_cache_config: KVCacheConfig,
    extra_config: dict[str, Any] | None = None,
    vllm_config=None,
) -> dict[StoreTrigger, StoreTriggerHandler]:
    vllm_config = vllm_config or _make_vllm_config(extra_config=extra_config)
    spec = MockOffloadingSpec(build_offloading_config(vllm_config, kv_cache_config))
    config = SchedulerOffloadConfig.from_spec(spec, vllm_config, kv_cache_config)
    events_tracker = OffloadingEventsTracker(
        spec.kv_events_config,
        {
            group.group_idx: group.kv_event_group_spec
            for group in config.kv_group_configs
        },
    )
    return StoreHandlerFactory.create_handlers(
        config,
        spec.get_manager().store_triggers,
        events_tracker,
        vllm_config,
        kv_cache_config,
        spec.extra_config,
    )


def test_the_layout_handler_is_added_first():
    """ON_HANDOFF runs before the handlers of the manager, when the layout
    hands off state blocks."""
    handlers = _create_handlers(_make_mamba_hybrid_kv_cache_config())

    assert list(handlers) == [StoreTrigger.ON_HANDOFF, StoreTrigger.ON_COMPUTE]
    assert isinstance(handlers[StoreTrigger.ON_HANDOFF], OnHandoffStoreHandler)
    assert isinstance(handlers[StoreTrigger.ON_COMPUTE], OnComputeStoreHandler)


def test_a_trigger_of_the_layout_and_of_the_manager_raises():
    """test_unsupported_store_triggers_raise covers an empty set and a
    repeated manager trigger; only a layout/manager overlap is left."""
    with pytest.raises(ValueError, match="unique and not empty"):
        _create_handlers(
            _make_mamba_hybrid_kv_cache_config(), {"store_triggers": ["ON_HANDOFF"]}
        )


def test_check_support_is_called_once_per_class(monkeypatch):
    """A refusal is per handler class, before any handler is made."""
    calls: list[tuple[type, object, object, object, object]] = []

    def record(cls, vllm_config, kv_cache_config, extra_config, events_tracker):
        calls.append((cls, vllm_config, kv_cache_config, extra_config, events_tracker))

    monkeypatch.setattr(OnHandoffStoreHandler, "check_support", classmethod(record))
    monkeypatch.setattr(OnComputeStoreHandler, "check_support", classmethod(record))
    kv_cache_config = _make_mamba_hybrid_kv_cache_config()
    vllm_config = _make_vllm_config(extra_config={"key": "value"})

    _create_handlers(kv_cache_config, {"key": "value"}, vllm_config)

    assert [cls for cls, *_ in calls] == [OnHandoffStoreHandler, OnComputeStoreHandler]
    trackers = set()
    for _, seen_vllm_config, seen_kv_cache_config, seen_extra_config, tracker in calls:
        assert seen_vllm_config is vllm_config
        assert seen_kv_cache_config is kv_cache_config
        assert seen_extra_config == {"key": "value"}
        assert isinstance(tracker, OffloadingEventsTracker)
        trackers.add(id(tracker))
    # Every class gets the same tracker, which is the one the handler will get.
    assert len(trackers) == 1
