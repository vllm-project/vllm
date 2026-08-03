# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side execution-artifact capture."""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)


class ArtifactWorkerConnector:
    """Own model hooks and snapshots for enabled execution artifacts."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        max_num_batched_tokens: int,
    ) -> None:
        self._routed_experts_capturer: RoutedExpertsCapturer | None = None
        if vllm_config.artifact_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer(
                max_num_batched_tokens=max_num_batched_tokens,
                vllm_config=vllm_config,
            )
            bind_routed_experts_capturer(model, capturer)
            self._routed_experts_capturer = capturer

    def capture_routed_experts(self, num_tokens: int) -> torch.Tensor | None:
        """Return a stable routed-experts snapshot for the current step."""
        if self._routed_experts_capturer is None:
            return None
        return self._routed_experts_capturer.get_routing_data(num_tokens)
