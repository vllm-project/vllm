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
        KVCacheLayerConfig,
    )


class KVCacheConfigBuilder(ABC):
    """Interface for platform KV cache placement.

    .. note::
        This interface is experimental and subject to change. The exact hooks
        and the ``num_blocks`` contract may evolve before it stabilizes.

    Subclasses normally inherit from ``DefaultKVCacheConfigBuilder`` and
    override only the hooks they need. The cross-worker planning flow
    (:func:`vllm.v1.core.kv_cache_planning.get_kv_cache_configs`) is Core-owned;
    this builder supplies the placement decisions: how layers are split into
    groups and how a group list is materialized into tensors for a given block
    count. Grouping coordination lives in ``DefaultKVCacheConfigBuilder`` (a
    model implementing :class:`~vllm.model_executor.models.interfaces.
    SupportsKVCacheGrouping` owns its grouping there); a platform that replaces
    the builder fully owns its own grouping.
    """

    @abstractmethod
    def get_kv_cache_groups(
        self,
        vllm_config: "VllmConfig",
        layer_configs: dict[str, "KVCacheLayerConfig"],
    ) -> list["KVCacheGroupSpec"]:
        """Split the given layers into KV cache groups.

        Planning passes all workers' layers; profiling passes one worker's.
        Core rejects a result that does not put each layer in exactly one group.

        The default builder honors a model's
        :class:`~vllm.model_executor.models.interfaces.SupportsKVCacheGrouping`
        declaration here before falling back to the default grouping; a
        platform-specific builder decides its own grouping.
        """
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_config_from_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
        num_blocks: int,
    ) -> "KVCacheConfig":
        """Materialize the groups for exactly ``num_blocks`` blocks.

        Each device backing's size is ``bytes_per_block * num_blocks``. All
        device tensors share one backing allocation unless the builder
        overrides ``_get_pool_bytes_per_block``. Host-resident tensors, such as
        HiSparse source storage, are excluded from the device byte accounting
        and sized independently.
        """
        raise NotImplementedError

    def _get_pool_bytes_per_block(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
    ) -> int | None:
        """Return the device bytes one block ID consumes, or ``None``.

        Private and transitional; expected to go away with worker-driven
        allocation (RFC #42449). With ``None`` (the default), Core derives the
        value from a one-block placement with one device backing. Override it
        only for placements with several device allocations.
        """
        return None


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
