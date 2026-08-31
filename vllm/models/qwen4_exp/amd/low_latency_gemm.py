# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp low-latency GEMM hook for AMD ROCm."""

import torch
from torch import nn


def enable_qwen4_exp_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    """Keep the standard vLLM linear methods on AMD ROCm."""

    del module, dtype
