# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at b02df0db8 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/glm/__init__.py

"""Public API for the GLM-5 indexed decode MonoKernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.models.deepseek_v32.amd.mono.glm.op import Glm5MonoKernel
    from vllm.models.deepseek_v32.amd.mono.weights import LayerWeights

__all__ = ["Glm5MonoKernel", "LayerWeights"]


def __getattr__(name: str):
    """Load GPU wrappers only when callers request them."""
    if name == "Glm5MonoKernel":
        from vllm.models.deepseek_v32.amd.mono.glm.op import Glm5MonoKernel

        return Glm5MonoKernel
    if name == "LayerWeights":
        from vllm.models.deepseek_v32.amd.mono.weights import LayerWeights

        return LayerWeights
    raise AttributeError(name)
