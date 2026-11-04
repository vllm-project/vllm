# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.state import (
    SchedulerOffloadConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.base import (  # noqa: E501
    StoreTriggerHandler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.on_compute import (  # noqa: E501
    OnComputeStoreHandler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.on_handoff import (  # noqa: E501
    OnHandoffStoreHandler,
)
from vllm.v1.kv_offload.base import StoreTrigger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class StoreHandlerFactory:
    """Registry of store handler classes, keyed by trigger.

    Mirrors OffloadingSpecFactory and CachePolicyFactory: built-in handlers
    are registered below, at import of this module, so a handler must be
    registered before the scheduler builds its handlers. The scheduler names
    no handler class.
    """

    _registry: dict[StoreTrigger, type[StoreTriggerHandler]] = {}

    @classmethod
    def register_handler(cls, handler_cls: type[StoreTriggerHandler]) -> None:
        """Register a handler class for its trigger. Raise if registered twice."""
        trigger = handler_cls.trigger
        if trigger in cls._registry:
            raise ValueError(f"Store trigger {trigger!r} is already registered.")
        cls._registry[trigger] = handler_cls

    @classmethod
    def create_handlers(
        cls,
        config: SchedulerOffloadConfig,
        manager_store_triggers: Sequence[StoreTrigger],
        events_tracker: OffloadingEventsTracker,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        extra_config: Mapping[str, Any],
    ) -> dict[StoreTrigger, StoreTriggerHandler]:
        """Make one handler per store trigger, in trigger order.

        The layout handlers come first, then one handler per trigger that the
        manager asks for.

        Args:
            config: the offloading config of this engine.
            manager_store_triggers: the triggers that the manager asks for.
            events_tracker: the events tracker that the handlers will get.
            vllm_config: the engine config.
            kv_cache_config: the KV cache layout of this engine.
            extra_config: kv_connector_extra_config of this connector.

        Returns:
            trigger -> handler instance.

        Raises:
            ValueError: on an empty or duplicate trigger set, or on a
                trigger with no registered handler.

        """
        triggers = cls._get_store_triggers(config, manager_store_triggers)
        handler_classes = [cls._get_handler_cls(trigger) for trigger in triggers]
        for handler_cls in handler_classes:
            handler_cls.check_support(
                vllm_config, kv_cache_config, extra_config, events_tracker
            )
        return {
            trigger: handler_cls(config, events_tracker)
            for trigger, handler_cls in zip(triggers, handler_classes, strict=True)
        }

    @classmethod
    def _get_store_triggers(
        cls,
        config: SchedulerOffloadConfig,
        manager_store_triggers: Sequence[StoreTrigger],
    ) -> tuple[StoreTrigger, ...]:
        # The handlers that the KV cache layout requires run first.
        layout_triggers: tuple[StoreTrigger, ...] = (
            (StoreTrigger.ON_HANDOFF,) if config.supports_handoff else ()
        )
        triggers = layout_triggers + tuple(manager_store_triggers)
        if not triggers or len(set(triggers)) != len(triggers):
            raise ValueError(
                f"The store triggers of the KV cache layout and of the "
                f"offloading manager must be unique and not empty: "
                f"{triggers}"
            )
        return triggers

    @classmethod
    def _get_handler_cls(cls, trigger: StoreTrigger) -> type[StoreTriggerHandler]:
        handler_cls = cls._registry.get(trigger)
        if handler_cls is None:
            raise ValueError(
                f"No store handler is registered for trigger {trigger!r}. "
                f"Registered: {list(cls._registry)}."
            )
        return handler_cls


# Register built-in handlers here.
StoreHandlerFactory.register_handler(OnComputeStoreHandler)
StoreHandlerFactory.register_handler(OnHandoffStoreHandler)
