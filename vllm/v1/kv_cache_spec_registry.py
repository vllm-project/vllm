# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Registry for KVCacheSpec types and their associated managers.

This module provides a pluggable architecture for registering custom KVCacheSpec
subclasses without modifying vLLM core code. Out-of-tree platforms can define
custom specs and managers by using the @register_kv_cache_spec decorator.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheSpec


@dataclass(frozen=True)
class KVCacheSpecMetadata:
    """Metadata for a registered KVCacheSpec."""

    kvcache_spec_cls: type["KVCacheSpec"]
    manager_class: type["SingleTypeKVCacheManager"]
    # The base spec class for grouping compatibility checks.
    # KVCacheSpecs with the same uniform_type_base_spec will be
    # grouped into one kvcache group
    uniform_type_base_spec: type["KVCacheSpec"]


_REGISTRY_KVCACHESPEC_LIST: dict[type["KVCacheSpec"], KVCacheSpecMetadata] = {}


class KVCacheSpecRegistry:
    """Global registry for KVCacheSpec types and their associated managers."""

    @classmethod
    def _ensure_registered(cls, vllm_config=None) -> None:
        """
        Run full KVCacheSpec registration if the registration is not done.
        """
        if _REGISTRY_KVCACHESPEC_LIST:
            return

        if vllm_config is None:
            from vllm.config import get_current_vllm_config_or_none

            vllm_config = get_current_vllm_config_or_none()

        # lazy import to avoid circular dependency
        from vllm.v1.core.single_type_kv_cache_manager import (
            register_all_kvcache_specs,
        )

        register_all_kvcache_specs(vllm_config)

    @classmethod
    def register(
        cls,
        kvcache_spec_cls: type["KVCacheSpec"],
        manager_class: type["SingleTypeKVCacheManager"] | None = None,
        uniform_type_base_spec: type["KVCacheSpec"] | None = None,
    ) -> None:
        """
        Register a KVCacheSpec class with its manager and base spec.

        Args:
            kvcache_spec_cls: The KVCacheSpec subclass to register
            manager_class: The SingleTypeKVCacheManager to use for this spec
            uniform_type_base_spec: The base spec class for grouping compatibility.
                instead of being grouped to different kvcache group, `kvcache_spec_cls`
                and `uniform_type_base_spec` will be trated as uniform type.
                If None, defaults to kvcache_spec_cls itself (for built-in base specs).
        """
        assert manager_class is not None, (
            "manager_class is required when override=False"
        )
        if uniform_type_base_spec is None:
            uniform_type_base_spec = kvcache_spec_cls
        if kvcache_spec_cls in _REGISTRY_KVCACHESPEC_LIST:
            registered_spec = _REGISTRY_KVCACHESPEC_LIST[kvcache_spec_cls]
            is_same_registration = (
                manager_class == registered_spec.manager_class
                and uniform_type_base_spec == registered_spec.uniform_type_base_spec
            )
            assert is_same_registration, (
                f"Conflicting registration for KVCacheSpec "
                f": {kvcache_spec_cls.__name__}"
            )

        _REGISTRY_KVCACHESPEC_LIST[kvcache_spec_cls] = KVCacheSpecMetadata(
            kvcache_spec_cls=kvcache_spec_cls,
            manager_class=manager_class,
            uniform_type_base_spec=uniform_type_base_spec,
        )

    @classmethod
    def override(
        cls,
        kvcache_spec_cls: type["KVCacheSpec"],
        target_kv_cache_spec_cls: type["KVCacheSpec"] | None = None,
        manager_class: type["SingleTypeKVCacheManager"] | None = None,
        uniform_type_base_spec: type["KVCacheSpec"] | None = None,
    ) -> None:
        """
        Override the target_kv_cache_spec_cls, manager and/or uniform_type_base_spec
        by kvcache_spec_cls.

        Args:
            kvcache_spec_cls: The custom KVCacheSpec class to override.
            target_kv_cache_spec_cls: The target KVCacheSpec class to be overridden
            manager_class: New manager class. If None, keeps existing value.
            uniform_type_base_spec: New grouping base class. If None, keeps existing.

        Raises:
            ValueError: If target_kv_cache_spec_cls is not already registered

        Example::

            # Replace FullAttentionSpec with a CustomFullAttentionSpec:
            KVCacheSpecRegistry.override(
                kvcache_spec_cls=CustomFullAttentionSpec,
                target_kv_cache_spec_cls=FullAttentionSpec,
                manager_class=CustomFullAttentionManager,
            )
        """
        assert target_kv_cache_spec_cls is not None, (
            "Please specify a target_kv_cache_spec_cls when override a KVCacheSpec"
        )
        if target_kv_cache_spec_cls not in _REGISTRY_KVCACHESPEC_LIST:
            raise ValueError(
                f"Cannot override unregistered spec {kvcache_spec_cls.__name__}. "
                f"Use register() to add new specs."
            )

        target_kv_cache_spec = _REGISTRY_KVCACHESPEC_LIST[target_kv_cache_spec_cls]
        new_manager = (
            manager_class
            if manager_class is not None
            else target_kv_cache_spec.manager_class
        )
        new_grouping_base = (
            uniform_type_base_spec
            if uniform_type_base_spec is not None
            else target_kv_cache_spec.uniform_type_base_spec
        )

        logger.info(
            "Overriding KVCacheSpec %s: manager %s -> %s, "
            "uniform_type_base_spec %s -> %s",
            kvcache_spec_cls.__name__,
            target_kv_cache_spec.manager_class.__name__,
            new_manager.__name__,
            target_kv_cache_spec.uniform_type_base_spec.__name__,
            new_grouping_base.__name__,
        )

        _REGISTRY_KVCACHESPEC_LIST[target_kv_cache_spec_cls] = KVCacheSpecMetadata(
            kvcache_spec_cls=kvcache_spec_cls,
            manager_class=new_manager,
            uniform_type_base_spec=new_grouping_base,
        )

    @classmethod
    def get_manager_class(
        cls, kvcache_spec: "KVCacheSpec"
    ) -> type["SingleTypeKVCacheManager"] | None:
        """
        Get the single type kvcache manager class for a given kvcache spec instance.

        Args:
            kvcache_spec: A KVCacheSpec instance

        Returns:
            The SingleTypeKVCacheManager class to use for this kvcache_spec
        """
        cls._ensure_registered()
        kvcache_spec_cls = type(kvcache_spec)

        # Walk up the MRO to find a registered base class
        for base in kvcache_spec_cls.__mro__:
            if base in _REGISTRY_KVCACHESPEC_LIST:
                return _REGISTRY_KVCACHESPEC_LIST[base].manager_class

        return None

    @classmethod
    def get_uniform_type_base_spec(
        cls, kvcache_spec: "KVCacheSpec"
    ) -> type["KVCacheSpec"] | None:
        """
        Get the base kvcache spec class for grouping compatibility checks.
        KVCacheSpecs with uniform_type_base_spec will be trated as one group.

        Args:
            kvcache_spec: A KVCacheSpec instance

        Returns:
            The base KVCacheSpec class for checking uniform type kvcache specs
        """
        cls._ensure_registered()
        kvcache_spec_cls = type(kvcache_spec)

        # Walk up the MRO to find a registered base spec
        for base in kvcache_spec_cls.__mro__:
            if base in _REGISTRY_KVCACHESPEC_LIST:
                return _REGISTRY_KVCACHESPEC_LIST[base].uniform_type_base_spec

        return None

    @classmethod
    def check_kv_cache_spec_registry(
        cls, kv_cache_spec: dict[str, "KVCacheSpec"]
    ) -> None:
        """
        Check if the KVCacheSpecs of each layer are registered as expected.
        """
        cls._ensure_registered()
        for layer_name, spec in kv_cache_spec.items():
            # use raise instead of assert to make it effective in production environment
            if cls.get_uniform_type_base_spec(spec) is None:
                raise ValueError(
                    f"Unsupported KV cache spec type for layer {layer_name}: "
                    f"{type(spec)}. Please register it using "
                    f"@register_kv_cache_spec decorator."
                )
            if cls.get_manager_class(spec) is None:
                raise ValueError(
                    f"No manager found for KV cache spec type for layer "
                    f"{layer_name}: {type(spec)}. Please register it using "
                    f"@register_kv_cache_spec decorator."
                )


def register_kv_cache_spec(
    manager_class: type["SingleTypeKVCacheManager"] | None = None,
    uniform_type_base_spec: type["KVCacheSpec"] | None = None,
    override: bool = False,
    target_kv_cache_spec_cls: type["KVCacheSpec"] | None = None,
):
    """
    Decorator to register a custom KVCacheSpec class.

    Args:
        manager_class: The SingleTypeKVCacheManager to use for this spec.
            Required when override=False, optional when override=True.
        uniform_type_base_spec: The base spec class for uniform type kv cache specs
            compatibility. If None, the spec is treated as a new base
            type (when override=False) or keeps existing value (when override=True).
        override: If True, calls override() instead of register(). Use this when
            you want to change the manager for an existing spec without creating
            a new subclass.
        target_kv_cache_spec_cls: The target KVCacheSpec class to be overridden.
            Required when override=True, and required to be None when override=False.

    Examples:
    - Register a new specs:
        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec
        )
        @dataclass(frozen=True, kw_only=True)
        class CustomFullAttentionSpec(FullAttentionSpec):
            pass

    - Override an existing spec:
        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            override=True,
            target_kv_cache_spec_cls=FullAttentionSpec
        )
        @dataclass(frozen=True, kw_only=True)
        class CustomFullAttentionSpec(FullAttentionSpec):
            pass
    """

    def decorator(kvcache_spec_cls: type["KVCacheSpec"]) -> type["KVCacheSpec"]:
        if override:
            # Use override() method for existing specs
            KVCacheSpecRegistry.override(
                kvcache_spec_cls=kvcache_spec_cls,
                manager_class=manager_class,
                uniform_type_base_spec=uniform_type_base_spec,
                target_kv_cache_spec_cls=target_kv_cache_spec_cls,
            )
        else:
            # Use register() method for new specs
            assert target_kv_cache_spec_cls is None, (
                f"Unexpected target_kv_cache_spec_cls: {target_kv_cache_spec_cls}"
                " when just registering a KVCacheSpec"
            )
            KVCacheSpecRegistry.register(
                kvcache_spec_cls=kvcache_spec_cls,
                manager_class=manager_class,
                uniform_type_base_spec=uniform_type_base_spec,
            )
        return kvcache_spec_cls

    return decorator
