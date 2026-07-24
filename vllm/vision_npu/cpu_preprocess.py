# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Factory for model-specific CPU preprocessors used by NPU vision backends.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from vllm.vision_npu.models.qwen2_5_vl_cpu_preprocess import Qwen2_5_VLCpuPreprocessor


class CpuPreprocessor(Protocol):
    """Interface for CPU-side vision preprocessing before NPU execution."""

    def preprocess(self, pixel_values: torch.Tensor) -> np.ndarray: ...

    def postprocess(self, npu_output: np.ndarray) -> np.ndarray: ...


def get_cpu_preprocessor(model_path: str) -> CpuPreprocessor:
    """Return the CPU preprocessor for the vision bundle (.rai path)."""
    return Qwen2_5_VLCpuPreprocessor(model_path)
