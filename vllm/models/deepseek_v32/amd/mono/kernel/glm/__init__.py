# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as vendored by
# ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/glm/__init__.py). The original source code was
# licensed under the Apache License 2.0 and included the following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths rewritten to this package.

"""Public API for the GLM-5 indexed decode MonoKernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.models.deepseek_v32.amd.mono.kernel.glm.op import Glm5MonoKernel
    from vllm.models.deepseek_v32.amd.mono.kernel.weights import LayerWeights

__all__ = ["Glm5MonoKernel", "LayerWeights"]


def __getattr__(name: str):
    """Load GPU wrappers only when callers request them."""

    if name == "Glm5MonoKernel":
        from vllm.models.deepseek_v32.amd.mono.kernel.glm.op import Glm5MonoKernel

        return Glm5MonoKernel
    if name == "LayerWeights":
        from vllm.models.deepseek_v32.amd.mono.kernel.weights import LayerWeights

        return LayerWeights
    raise AttributeError(name)
