# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.tiering.base import SecondaryTierManager

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec


class SecondaryTierFactory:
    _registry: dict[str, Callable[[], type[SecondaryTierManager]]] = {}

    @classmethod
    def register_tier(cls, tier_type: str, module_path: str, class_name: str) -> None:
        if tier_type in cls._registry:
            raise ValueError(f"Tier '{tier_type}' is already registered.")

        def loader() -> type[SecondaryTierManager]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[tier_type] = loader

    @classmethod
    def create_secondary_tier(
        cls,
        tier_config: dict,
        primary_kv_view: memoryview,
        offloading_spec: "OffloadingSpec",
    ) -> SecondaryTierManager:
        tier_cls = cls.get_tier_class(tier_config)
        config = tier_config.copy()
        tier_type = config.pop("type")
        return tier_cls(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
            **config,
        )

    @classmethod
    def get_tier_class(cls, tier_config: dict) -> type[SecondaryTierManager]:
        tier_type = tier_config.get("type")
        if not tier_type:
            raise ValueError("Secondary tier configuration must include 'type'")
        if tier_type not in cls._registry:
            raise ValueError(
                f"Unknown secondary tier type: {tier_type!r}. "
                f"Supported types: {list(cls._registry)}"
            )
        return cls._registry[tier_type]()


SecondaryTierFactory.register_tier(
    "example",
    "vllm.v1.kv_offload.tiering.example.manager",
    "ExampleSecondaryTierManager",
)

SecondaryTierFactory.register_tier(
    "fs",
    "vllm.v1.kv_offload.tiering.fs.manager",
    "FileSystemTierManager",
)

SecondaryTierFactory.register_tier(
    "obj",
    "vllm.v1.kv_offload.tiering.obj.manager",
    "ObjectStoreSecondaryTierManager",
)
