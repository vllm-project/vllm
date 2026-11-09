# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unquantized GEMM dispatch for the hw-agnostic layers.

Only CPU, which needs its packed weights, and `torch.nn.functional.linear` are
dispatched here. Accelerator-specific GEMMs are left to out-of-tree plugins,
which override the layer rather than adding a branch here.
"""

from collections.abc import Callable

import torch

from vllm.model_executor.layers.utils import (
    cpu_unquantized_gemm,
    default_unquantized_gemm,
)
from vllm.platforms import current_platform


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_cpu():
        return cpu_unquantized_gemm
    return default_unquantized_gemm
