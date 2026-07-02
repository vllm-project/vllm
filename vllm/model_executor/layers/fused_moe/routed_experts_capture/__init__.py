# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-experts capture + KV-offload sidecar.

Returns, per request, the LOGICAL expert IDs each token was routed to at every
MoE layer (the ``enable_return_routed_experts`` feature). Routing is captured
on the worker, accounted in a scheduler-side slot buffer, and — when KV
offload is active — follows the KV cache through every tier (GPU <-> CPU <->
disk/object/...). The routing data only REUSES the KV offload lifecycle as
transport; it does not depend on the MoE compute path.

Modules:
  - ``common``   : torch-free helpers shared by worker and scheduler
                   (anchor-group selection, HF expert-count resolution).
  - ``capturer`` : worker-side ``RoutedExpertsCapturer`` (GPU forward hook).
  - ``manager``  : scheduler-side ``RoutedExpertsManager`` (slot/offload
                   buffer) plus the block-map helpers.
  - ``store``    : pluggable secondary-tier backends — the
                   ``RoutedExpertsSecondaryStore`` interface, the
                   ``RoutedExpertsStoreFactory``, the built-in filesystem
                   backend, and the lifecycle observer that bridges KV-tier
                   cascade/promotion events to a backend.
"""

from vllm.model_executor.layers.fused_moe.routed_experts_capture.capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.common import (
    find_full_attention_gid,
    get_num_experts,
    get_num_experts_per_tok,
    require_full_attention_gid,
    routed_experts_output_rank,
    routing_slot_shape_dtype,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.manager import (
    FullAttnBlockMap,
    RoutedExpertsManager,
    compute_full_attn_block_map,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (
    RoutedExpertsWorkerWriter,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.store import (
    FileRoutedExpertsStore,
    RoutedExpertsBlockLifecycleObserver,
    RoutedExpertsSecondaryStore,
    RoutedExpertsStoreContext,
    RoutedExpertsStoreFactory,
    build_fs_routed_experts_store,
)

__all__ = [
    "FileRoutedExpertsStore",
    "FullAttnBlockMap",
    "RoutedExpertsBlockLifecycleObserver",
    "RoutedExpertsCapturer",
    "RoutedExpertsManager",
    "RoutedExpertsSecondaryStore",
    "RoutedExpertsStoreContext",
    "RoutedExpertsStoreFactory",
    "RoutedExpertsWorkerWriter",
    "build_fs_routed_experts_store",
    "compute_full_attn_block_map",
    "find_full_attention_gid",
    "get_num_experts",
    "get_num_experts_per_tok",
    "require_full_attention_gid",
    "routed_experts_output_rank",
    "routing_slot_shape_dtype",
]
