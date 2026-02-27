# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

from vllm.config import VllmConfig
from vllm.distributed.disaggregated.request_manager import (
    DisaggregatedRequestManager)


class DisaggregatedRequestManagerFactory:
    _registry: dict[str, Callable[..., DisaggregatedRequestManager]] = {}

    @classmethod
    def register_request_manager(
            cls, class_name: str,
            ctor: Callable[..., DisaggregatedRequestManager]) -> None:
        """Register a request manager along with its constructor."""
        if class_name in cls._registry:
            raise ValueError(
                f"Request manager '{class_name}' is already registered.")

        cls._registry[class_name] = ctor

    @classmethod
    def create_request_managers(
        cls,
        config: "VllmConfig",
    ) -> list[DisaggregatedRequestManager]:

        managers = []
        for manager_ctor in cls._registry.values():
            manager = manager_ctor(config)
            managers.append((manager.priority, manager))
        return [manager for _, manager in sorted(managers, key=lambda x: x[0])]

    @classmethod
    def register(cls, name: Optional[str] = None):
        """Class decorator to register a `DisaggregatedRequestManager`.

        Usage:
            @DisaggregatedRequestManagerFactory.register("MyManager")
            class MyManager(DisaggregatedRequestManager):
                ...
        If `name` is None, the class' __name__ is used.
        """

        def _decorator(manager_cls: type[DisaggregatedRequestManager]):
            register_name = name or manager_cls.__name__

            def _ctor(config: VllmConfig) -> DisaggregatedRequestManager:
                return manager_cls(config)

            cls.register_request_manager(register_name, _ctor)
            return manager_cls

        return _decorator
