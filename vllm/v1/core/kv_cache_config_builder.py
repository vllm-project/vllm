# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable KV cache config builder resolution."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheLayout,
        KVCacheSpec,
    )

logger = init_logger(__name__)


class KVCacheConfigBuilder:
    """Active KV cache config builder, resolved on first use.

    Resolution priority is owned by the platform hook
    (:meth:`vllm.platforms.interface.Platform.get_kv_cache_config_builder_cls`);
    this class only caches the resolved builder and delegates to it.
    """

    _active: "DefaultKVCacheConfigBuilder | None" = None

    @classmethod
    def _resolve(cls, vllm_config: "VllmConfig") -> "DefaultKVCacheConfigBuilder":
        if cls._active is None:
            from vllm.platforms import current_platform

            builder_cls = resolve_obj_by_qualname(
                current_platform.get_kv_cache_config_builder_cls(vllm_config)
            )
            cls._active = builder_cls()
        return cls._active

    @classmethod
    def reset(cls) -> None:
        cls._active = None

    @classmethod
    def get_kv_cache_configs(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
    ) -> list["KVCacheConfig"]:
        return cls._resolve(vllm_config).get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_memory
        )

    @classmethod
    def get_kv_cache_groups(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
    ) -> list["KVCacheGroupSpec"]:
        return cls._resolve(vllm_config).get_kv_cache_groups(vllm_config, kv_cache_spec)

    @classmethod
    def get_kv_cache_config_from_groups(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
        available_memory: int,
    ) -> "KVCacheConfig":
        return cls._resolve(vllm_config).get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )

    @classmethod
    def validate_kv_cache_layout(
        cls,
        layout: "KVCacheLayout",
        kv_cache_groups: list["KVCacheGroupSpec"],
        vllm_config: "VllmConfig | None" = None,
    ) -> None:
        if vllm_config is None:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        return cls._resolve(vllm_config).validate_kv_cache_layout(
            layout, kv_cache_groups
        )

    @classmethod
    def may_override_num_blocks(cls, vllm_config: "VllmConfig", num_blocks: int) -> int:
        return cls._resolve(vllm_config).may_override_num_blocks(
            vllm_config, num_blocks
        )

    @classmethod
    def _get_kv_cache_bytes_per_block(
        cls,
        kv_cache_groups: list["KVCacheGroupSpec"],
        vllm_config: "VllmConfig | None" = None,
    ) -> int:
        if vllm_config is None:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        return cls._resolve(vllm_config)._get_kv_cache_bytes_per_block(kv_cache_groups)
