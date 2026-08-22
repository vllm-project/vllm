# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import SecondaryTierManager

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class SecondaryTierFactory:
    """Registry for SecondaryTierManager implementations, resolved by type.

    Mirrors OffloadingSpecFactory (vllm/v1/kv_offload/factory.py): built-in
    tiers are pre-registered below. External tiers can either
    register_tier() a friendly short name up front, or skip registration
    entirely and pass a module_path at lookup time (out-of-tree, no vLLM
    fork/patch required) -- see get_tier_class.
    """

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
        config.pop("module_path", None)
        return tier_cls(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
            **config,
        )

    @classmethod
    def get_tier_class(cls, tier_config: dict) -> type[SecondaryTierManager]:
        """Get a secondary tier class by type.

        Args:
            tier_config: Tier configuration dict. Must contain 'type' (the
                tier name or out-of-tree class name). If 'type' is not a
                registered tier and 'module_path' is given, the class is
                imported from there -- an out-of-tree tier needs no
                register_tier() call, just this module path in the config.

        Returns:
            The secondary tier manager class.

        Raises:
            ValueError: If 'type' is missing, or is neither registered nor
                resolvable via module_path.
        """
        tier_type = tier_config.get("type")
        if not tier_type:
            raise ValueError("Secondary tier configuration must include 'type'")
        if tier_type in cls._registry:
            return cls._registry[tier_type]()
        module_path = tier_config.get("module_path")
        if module_path is None:
            raise ValueError(
                f"Unknown secondary tier type: {tier_type!r}. "
                f"Supported types: {list(cls._registry)}. "
                "For an out-of-tree tier, also set 'module_path'."
            )
        logger.warning_once(
            "Loading out-of-tree secondary tier. This API is "
            "experimental and subject to change in the future "
            "as we iterate the design."
        )
        logger.info(
            "Loading out-of-tree secondary tier '%s' from '%s'.",
            tier_type,
            module_path,
        )
        module = importlib.import_module(module_path)
        tier_cls = getattr(module, tier_type)
        assert issubclass(tier_cls, SecondaryTierManager)
        return tier_cls


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
    "p2p",
    "vllm.v1.kv_offload.tiering.p2p.manager",
    "P2PSecondaryTierManager",
)

SecondaryTierFactory.register_tier(
    "obj",
    "vllm.v1.kv_offload.tiering.obj.manager",
    "ObjectStoreSecondaryTierManager",
)
