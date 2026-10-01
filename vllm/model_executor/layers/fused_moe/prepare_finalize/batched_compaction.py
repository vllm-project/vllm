# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEExperts

logger = init_logger(__name__)


class BatchedExpertCompaction:
    """Compact packed expert prefixes and restore the communication layout."""

    def __init__(self):
        self._compact_tokens: int | None = None
        self._restore_shapes: list[tuple[int, ...] | None] = [None, None]
        self._contiguous_scales: bool = False

    def configure(
        self,
        expert_capacity: int | None,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        supports_token_dropping: bool,
        supports_compaction: bool,
        contiguous_scales: bool,
    ) -> int | None:
        if (
            supports_token_dropping
            and expert_capacity is not None
            and supports_compaction
        ):
            self._contiguous_scales = contiguous_scales
            tokens_per_dispatcher = max(1, min(max_tokens_per_rank, expert_capacity))
            self._compact_tokens = tokens_per_dispatcher * num_dispatchers
            logger.info_once(
                "Compact expert layout: at most %d rows per expert; "
                "dispatch limit remains %d tokens per rank.",
                self._compact_tokens,
                max_tokens_per_rank,
            )
            return tokens_per_dispatcher

        self._compact_tokens = None
        return None

    @staticmethod
    def supports_experts(experts: FusedMoEExperts, use_fp8_dispatch: bool) -> bool:
        from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )
        from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
            BatchedTritonExperts,
        )
        from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
            BatchedHummingGroupedExperts,
        )

        if isinstance(experts, BatchedTritonExperts):
            return not experts.quant_config.is_quantized or (
                use_fp8_dispatch and experts.quant_config.use_fp8_w8a8
            )
        if not use_fp8_dispatch or not experts.quant_config.use_fp8_w8a8:
            return False
        supported_types = BatchedDeepGemmExperts | BatchedHummingGroupedExperts
        return isinstance(experts, supported_types)

    @staticmethod
    def uses_contiguous_scales(experts: FusedMoEExperts) -> bool:
        from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
            BatchedHummingGroupedExperts,
        )

        return isinstance(experts, BatchedHummingGroupedExperts)

    def compact(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        ubatch_id: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._restore_shapes[ubatch_id] = None
        if self._compact_tokens is not None:
            payload = x if isinstance(x, torch.Tensor) else x[0]
            payload_rows = payload.shape[1]
            if self._compact_tokens < payload_rows:
                self._restore_shapes[ubatch_id] = tuple(payload.shape)
                if isinstance(x, torch.Tensor):
                    x = payload[:, : self._compact_tokens].contiguous()
                else:
                    values, scales = x
                    scale_slice = scales[:, : self._compact_tokens]
                    if self._contiguous_scales:
                        compact_scales = scale_slice.contiguous()
                    else:
                        compact_scales = torch.empty_like(
                            scale_slice, memory_format=torch.preserve_format
                        )
                        compact_scales.copy_(scale_slice)
                    x = (
                        values[:, : self._compact_tokens].contiguous(),
                        compact_scales,
                    )
        return x

    def restore(
        self, output: torch.Tensor, ubatch_id: int
    ) -> tuple[torch.Tensor, Callable[[], None]]:
        shape = self._restore_shapes[ubatch_id]
        if shape is not None:
            restored = output.new_zeros(shape)
            restored[:, : output.shape[1]].copy_(output)
            output = restored

        def receiver():
            # Retain the restored allocation through the asynchronous recv hook.
            _ = output
            self._restore_shapes[ubatch_id] = None

        return output, receiver
