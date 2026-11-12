# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request


class KVObligationAdmission:
    """Reserve every admitted request's remaining full-attention KV footprint."""

    def __init__(self, config: "VllmConfig", manager: "KVCacheManager") -> None:
        self.manager = manager
        groups = manager.kv_cache_config.kv_cache_groups
        single_type_managers = manager.coordinator.single_type_managers
        parallel = config.parallel_config
        if (
            len(groups) != 1
            or len(single_type_managers) != 1
            or (group := groups[0]).manager_cls is not FullAttentionManager
            or type(single_type_managers[0]) is not FullAttentionManager
            or group.host_resident
            or group.is_eagle_group
            or len(specs := group.distinct_specs) != 1
            or type(spec := specs[0]) is not AttentionSpec
            or spec.tokens_per_state != 1
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or config.cache_config.enable_extensible_kv_cache
            or config.model_config.runner_type != "generate"
            or config.model_config.is_encoder_decoder
            or config.model_config.is_multimodal_model
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or config.scheduler_config.cache_aware_admission_window
            or config.scheduler_config.policy != "fcfs"
        ):
            raise ValueError(
                "Deterministic KV reserve requires FCFS, a fixed KV pool with "
                "one full-attention group, and decoder-only generation without "
                "speculation, KV transfer, context parallelism, or cache-aware "
                "admission."
            )
        self.block_size = spec.block_size
        self.max_model_len = manager.max_model_len

    def peak_tokens(self, request: "Request") -> int:
        return min(
            max(request.num_prompt_tokens + request.max_tokens, request.num_tokens),
            self.max_model_len,
        )

    def owed_blocks(self, request: "Request") -> int:
        (blocks,) = self.manager.get_blocks(request.request_id).blocks
        return max(0, cdiv(self.peak_tokens(request), self.block_size) - len(blocks))

    def rejection_reason(self, request: "Request") -> str | None:
        if request.resumable:
            return "Deterministic KV reserve requires a fixed prompt and output budget."
        # Block zero is reserved as the null block.
        capacity = self.manager.block_pool.num_gpu_blocks - 1
        required = cdiv(self.peak_tokens(request), self.block_size)
        if required > capacity:
            return (
                f"Prompt and maximum output require {required} KV blocks, but the "
                f"engine has {capacity}. Reduce the prompt or maximum output tokens."
            )
        return None
