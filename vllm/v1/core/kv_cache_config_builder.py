# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interface and resolution for pluggable KV cache config builders."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
    )


class KVCacheConfigBuilder(ABC):
    """Interface for model- or platform-specific KV cache planning.

    Subclasses normally inherit from ``DefaultKVCacheConfigBuilder`` and
    override only the hooks they need. A platform may replace the top-level
    planning method when it cannot use Core's cross-worker planning flow.
    """

    @abstractmethod
    def get_kv_cache_configs(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
    ) -> list["KVCacheConfig"]:
        """Generate the full KV cache configurations for every worker.

        The main entry point: takes the per-worker KV cache specs and the
        memory available on each worker, runs the whole planning pipeline
        (merge specs, group layers, project to workers, auto-fit
        max_model_len, admission checks, per-worker layouts, min-blocks
        convergence), and returns one ready-to-allocate
        :class:`~vllm.v1.kv_cache_interface.KVCacheConfig` per worker.
        """
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
    ) -> list["KVCacheGroupSpec"]:
        """Split a worker's layers into logical KV cache groups."""
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_config_from_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
        num_blocks: int,
    ) -> "KVCacheConfig":
        """Materialize groups for exactly ``num_blocks`` global block IDs.

        Every non-host-resident tensor must name the same device backing size,
        which must scale linearly with ``num_blocks``. Host-resident tensors,
        such as HiSparse source storage, are excluded from GPU accounting.
        """
        raise NotImplementedError


def get_kv_cache_config_builder(
    vllm_config: "VllmConfig",
) -> KVCacheConfigBuilder:
    """Resolve the builder selected by the current platform.

    Resolution priority is owned by the platform hook
    (:meth:`vllm.platforms.interface.Platform.get_kv_cache_config_builder_cls`).
    Builders are stateless (every method takes ``vllm_config``), so a fresh
    instance is returned per call and engines with different configs in the
    same process always get the correct builder.
    """
    from vllm.platforms import current_platform

    builder_cls = resolve_obj_by_qualname(
        current_platform.get_kv_cache_config_builder_cls(vllm_config)
    )
    return builder_cls()


def get_profiling_kv_cache_config(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, "KVCacheSpec"],
    min_blocks: int,
) -> "KVCacheConfig":
    """Build the smallest KV cache config for CUDA graph profiling.

    The config is sized to ``min_blocks`` blocks (at least one block per
    captured sequence) so the profiling run can capture every graph without
    consuming real KV cache memory. It goes through the active builder's
    normal grouping and placement hooks, so custom builders do not need to
    re-implement the profiling choreography.

    Consumed by the cudagraph profiling workers (``cudagraph_utils.py`` and
    ``gpu_model_runner.py``).
    """
    builder = get_kv_cache_config_builder(vllm_config)
    groups = builder.get_kv_cache_groups(vllm_config, kv_cache_spec)
    return builder.get_kv_cache_config_from_groups(
        vllm_config, groups, max(min_blocks, 1)
    )
