# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU vision tower for Qwen2.5-VL using FlexMLRT."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.models.vision import get_npu_vision_backend

logger = init_logger(__name__)


class Qwen2_5_VisionTransformerNPU(nn.Module):
    """Drop-in NPU replacement for Qwen2_5_VisionTransformer."""

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del prefix  # NPU backend has no trainable submodule tree.

        self.out_hidden_size = vision_config.out_hidden_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2

        try:
            self._npu_backend = get_npu_vision_backend()
        except Exception as e:
            logger.error("[Qwen2.5VL] NPU backend init failed: %s", e)
            raise RuntimeError(
                f"NPU vision backend initialization failed: {e}. "
                "Unset VLLM_VISION_NPU_CACHE to use PyTorch backend."
            ) from e

        if self._npu_backend is None:
            raise RuntimeError(
                "VLLM_VISION_NPU_CACHE is set but NPU backend failed to initialize."
            )

        logger.info("[Qwen2.5VL] Using NPU vision backend")

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        if pixel_values.dtype == torch.bfloat16:
            pixel_values_np = pixel_values.cpu().float().numpy()
        else:
            pixel_values_np = pixel_values.cpu().numpy().astype(np.float32)
        grid_thw_np = np.array(grid_thw, dtype=np.int64)

        embeddings_np = self._npu_backend.forward(pixel_values_np, grid_thw_np)

        if envs.VLLM_NPU_TIMING:
            import time

            gpu_transfer_start = time.monotonic()
            embeddings = torch.from_numpy(embeddings_np).to(
                device="cuda", dtype=torch.bfloat16
            )
            gpu_transfer_ms = (time.monotonic() - gpu_transfer_start) * 1000
            logger.debug(
                "[NPU Timing] CPU→GPU transfer: %.2fms (%.2f MB)",
                gpu_transfer_ms,
                embeddings_np.nbytes / 1024**2,
            )
            logger.debug("[Vision→LLM] Vision embeddings shape: %s", embeddings.shape)
        else:
            embeddings = torch.from_numpy(embeddings_np).to(
                device="cuda", dtype=torch.bfloat16
            )

        actual_tokens = embeddings.shape[0]
        merge_size = self.spatial_merge_size
        expected_tokens_per_image = [
            (t * h * w) // (merge_size * merge_size) for t, h, w in grid_thw
        ]
        total_expected = sum(expected_tokens_per_image)

        if actual_tokens != total_expected:
            logger.warning(
                "[NPU] Token count mismatch: NPU output %s tokens, "
                "but vLLM expects %s based on grid_thw. "
                "Repeating tokens to match expected count.",
                actual_tokens,
                total_expected,
            )
            repeat_factor = total_expected / actual_tokens
            if repeat_factor == int(repeat_factor):
                embeddings = embeddings.repeat_interleave(int(repeat_factor), dim=0)
            else:
                embeddings = embeddings.unsqueeze(0).unsqueeze(0)
                embeddings = F.interpolate(
                    embeddings,
                    size=(total_expected, embeddings.shape[-1]),
                    mode="nearest",
                )
                embeddings = embeddings.squeeze(0).squeeze(0)

            logger.debug(
                "[NPU] Padded from %s to %s tokens", actual_tokens, embeddings.shape[0]
            )

        return embeddings

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        logger.info("[Qwen2.5VL Vision] Skipping weight loading (using NPU backend)")
        return set()
