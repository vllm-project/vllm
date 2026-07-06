# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-experts capture and KV-connector sidecars.

The worker captures logical expert IDs for enable_return_routed_experts and
writes them into a scheduler-visible slot buffer. Supported KV connectors keep
the routing rows with their KV blocks when blocks move off GPU.
"""

from vllm.model_executor.layers.fused_moe.routed_experts_capture.async_output import (
    RoutedExpertsTensors,
    RoutedExpertsWriteTask,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.common import (
    get_num_experts,
    get_num_experts_per_token,
    get_routed_experts_output_rank,
    get_routing_slot_shape_and_dtype,
    require_full_attn_group_id,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.manager import (
    FullAttnBlockMap,
    RoutedExpertsManager,
    compute_full_attn_block_map,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (
    RoutedExpertsWorkerWriter,
)

__all__ = [
    "FullAttnBlockMap",
    "RoutedExpertsCapturer",
    "RoutedExpertsManager",
    "RoutedExpertsTensors",
    "RoutedExpertsWorkerWriter",
    "RoutedExpertsWriteTask",
    "compute_full_attn_block_map",
    "get_num_experts",
    "get_num_experts_per_token",
    "get_routed_experts_output_rank",
    "get_routing_slot_shape_and_dtype",
    "require_full_attn_group_id",
]
