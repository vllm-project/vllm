# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unquantized GEMM dispatch for the hw-agnostic layers.

Only CPU, which needs its packed weights, and `torch.nn.functional.linear` are
dispatched here. Accelerator-specific GEMMs are left to out-of-tree plugins,
which override the layer rather than adding a branch here.
"""

from collections.abc import Callable

import torch

from vllm import _cpu_ops as cpu_ops
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform

logger = init_logger(__name__)


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return layer.cpu_linear(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_cpu():
        return cpu_unquantized_gemm
    return default_unquantized_gemm


def dispatch_cpu_unquantized_gemm(layer: torch.nn.Module) -> None:
    """Set `layer.cpu_linear`, packing `layer.weight` for oneDNN if available.

    A packed weight replaces `layer.weight` with an empty tensor.
    """
    layer.cpu_linear = torch.nn.functional.linear
    if (
        layer.weight.is_meta
        or not cpu_ops._supports_onednn
        or current_platform.get_cpu_architecture() == CpuArchEnum.POWERPC
    ):
        return
    try:
        handler = cpu_ops.create_onednn_mm(layer.weight.t(), 32)
    except RuntimeError as e:
        logger.warning_once(
            "Failed to create oneDNN linear, fallback to torch linear. Exception: %s",
            e,
        )
        return
    layer.cpu_linear = lambda x, weight, bias: cpu_ops.onednn_mm(handler, x, bias)
    layer.weight = torch.nn.Parameter(
        torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device),
        requires_grad=False,
    )
