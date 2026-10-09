# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as vendored by
# ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/weights.py). The original source code was
# licensed under the Apache License 2.0 and included the following copyright notice:
# Copyright (c) 2026 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths rewritten to this package;
#   reduced to the weight container (the checkpoint adapters and expert-storage converters are unused).

"""Host-side weight container of one MonoKernel layer shard."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.models.deepseek_v32.amd.mono.kernel.config import GLM5_CONFIG, LayerConfig


@dataclass
class LayerWeights:
    """One tensor-parallel rank's weights and model geometry."""

    heads: int
    t: dict[str, torch.Tensor]
    config: LayerConfig = GLM5_CONFIG
    rank: int = 0
    npes: int = 1


__all__ = ["LayerWeights"]
