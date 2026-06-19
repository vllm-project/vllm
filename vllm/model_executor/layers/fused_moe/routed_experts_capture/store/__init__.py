# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-experts secondary-tier offload sidecar.

The scheduler-side offloaded-block buffer
(``RoutedExpertsManager.routed_experts_by_cpu_block``) follows the KV cache
through the offload tiers. The classes here let it cascade to / promote from a
secondary tier (disk/object/Mooncake/...) in lockstep with the KV blocks.

Routing is an MoE product, so it lives under ``fused_moe``; it only REUSES the
KV offload lifecycle (``kv_offload/``) as transport. The generic tier hook
(``BlockLifecycleObserver``) is defined in ``kv_offload/base.py``; everything
routing-specific is split across this package:

  - ``base``     : backend interface (``RoutedExpertsSecondaryStore``), the
                   builder context (``RoutedExpertsStoreContext``), and the
                   pluggable factory (``RoutedExpertsStoreFactory``).
  - ``fs``       : the built-in filesystem backend (``FileRoutedExpertsStore``)
                   plus its builder; self-registers under ``type="fs"``.
  - ``observer`` : ``RoutedExpertsBlockLifecycleObserver``, which bridges
                   KV-tier cascade / promotion events to a backend.

Importing ``fs`` here is what runs its ``register_store("fs", ...)`` side
effect, so the built-in tier is available as soon as the package is imported.
"""

from vllm.model_executor.layers.fused_moe.routed_experts_capture.store.base import (
    RoutedExpertsSecondaryStore,
    RoutedExpertsStoreContext,
    RoutedExpertsStoreFactory,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.store.fs import (
    FileRoutedExpertsStore,
    build_fs_routed_experts_store,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.store.observer import (
    RoutedExpertsBlockLifecycleObserver,
)

__all__ = [
    "FileRoutedExpertsStore",
    "RoutedExpertsBlockLifecycleObserver",
    "RoutedExpertsSecondaryStore",
    "RoutedExpertsStoreContext",
    "RoutedExpertsStoreFactory",
    "build_fs_routed_experts_store",
]
