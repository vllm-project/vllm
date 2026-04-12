# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class RoutedExpertsCaptureHelper:
    def __init__(self) -> None:
        self._initialized = False
        self._attn_gid = 0
        self._slot_mapping: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def init(self, runner: Any) -> None:
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            runner.model_config.enable_return_routed_experts,
        )
        capturer = RoutedExpertsCapturer.create()
        self._attn_gid = self._get_attention_kv_cache_gid(runner)
        min_block_size = min(
            group.kv_cache_spec.block_size
            for group in runner.kv_cache_config.kv_cache_groups
        )
        num_groups = len(runner.kv_cache_config.kv_cache_groups)
        max_num_kv_tokens = (
            runner.kv_cache_config.num_blocks // num_groups
        ) * min_block_size
        dcp_size = runner.vllm_config.parallel_config.decode_context_parallel_size
        pcp_size = runner.vllm_config.parallel_config.prefill_context_parallel_size
        if pcp_size * dcp_size > 1:
            max_num_kv_tokens *= pcp_size * dcp_size

        capturer.init_buffer(
            max_num_batched_tokens=runner.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=max_num_kv_tokens,
            vllm_config=runner.vllm_config,
        )
        self.bind(runner, capturer)
        self._initialized = True

    def bind(self, runner: Any, capturer: RoutedExpertsCapturer) -> None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )

        for module in runner.compilation_config.static_forward_context.values():
            if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
                layer_id = module.layer_id

                def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                    _capturer.capture(_layer_id, topk_ids)

                module.router.set_capture_fn(_capture_fn)

    def before_execute(self) -> None:
        if not self._initialized:
            return
        capturer = RoutedExpertsCapturer.get_instance()
        if capturer is None:
            logger.error("RoutedExpertsCapturer not initialized.")
            return
        capturer.clear_buffer()

    def record_slot_mapping(
        self,
        slot_mappings: tuple[torch.Tensor, ...],
        num_tokens: int,
    ) -> None:
        if not self._initialized:
            return
        slot_mapping_attn = slot_mappings[self._attn_gid]
        self._slot_mapping = slot_mapping_attn[:num_tokens].cpu().numpy()

    def save(self) -> None:
        if not self._initialized or self._slot_mapping is None:
            return
        capturer = RoutedExpertsCapturer.get_instance()
        if capturer is None:
            logger.error("RoutedExpertsCapturer not initialized.")
            return
        capturer.save_captured_experts(indices=self._slot_mapping)

    @staticmethod
    def _get_attention_kv_cache_gid(runner: Any) -> int:
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, AttentionSpec):
                return gid
        return 0
