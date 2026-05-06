# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable KV cache config builder.

Resolution order:
    platform override > model declaration > default

The default builder simply forwards to :mod:`vllm.v1.core.kv_cache_planning`.
Platforms or models can subclass :class:`KVCacheConfigBuilder` to customize
KV cache planning end-to-end.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec


class KVCacheConfigBuilder:
    """Strategy class for model- or platform-specific KV cache planning."""

    def get_kv_cache_configs(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
    ) -> list["KVCacheConfig"]:
        """Return the per-worker KV cache configs.

        The default implementation delegates to the standard planning logic
        in :mod:`vllm.v1.core.kv_cache_planning`. Subclasses may override
        this to provide platform- or model-specific layouts.
        """
        from vllm.v1.core.kv_cache_planning import (
            get_kv_cache_configs as _default_get_kv_cache_configs,
        )

        return _default_get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_memory
        )


def _load_builder(cls_path: str) -> KVCacheConfigBuilder:
    """Import and instantiate a builder from its fully-qualified class path."""
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls()


def resolve_builder(vllm_config: "VllmConfig") -> KVCacheConfigBuilder:
    """Resolve the KV cache config builder to use.

    Priority: platform override > model declaration > default.
    """
    from vllm.platforms import current_platform

    platform_cls_path = current_platform.get_kv_cache_config_builder_cls(vllm_config)
    if platform_cls_path is not None:
        return _load_builder(platform_cls_path)

    model_cls_path = vllm_config.model_config.kv_cache_config_builder_cls
    if model_cls_path is not None:
        return _load_builder(model_cls_path)

    return KVCacheConfigBuilder()


def get_kv_cache_configs(
    vllm_config: "VllmConfig",
    kv_cache_specs: list[dict[str, "KVCacheSpec"]],
    available_memory: list[int],
) -> list["KVCacheConfig"]:
    """Resolve the active builder and delegate KV cache planning to it."""
    builder = resolve_builder(vllm_config)
    return builder.get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)
