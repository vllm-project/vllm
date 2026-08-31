# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape-selected fused-A GEMM for unquantized BF16 decode projections.

``dsv3_fused_a_gemm`` beats cuBLAS on a handful of measured local ``(N, K)``
shapes at small token counts. A model opts in by handing its measured table to
:func:`install_fused_a_linear`, which swaps :class:`FusedALinearMethod` onto the
matching layers. Every other shape, token count, or dtype keeps the standard
unquantized linear path.
"""

from __future__ import annotations

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

# Local ``(N, K)`` -> token counts measured faster on the fused-A GEMM.
FusedATable = dict[tuple[int, int], frozenset[int]]


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _operands_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
        and hasattr(torch.ops._C, "dsv3_fused_a_gemm")
    )


def run_fused_a_gemm(
    tokens: frozenset[int],
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    """Fused-A GEMM output, or None when the standard path should run."""
    if x.shape[0] not in tokens or not _operands_ok(x, weight):
        return None
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    ops.dsv3_fused_a_gemm(output, x, weight.t(), enable_pdl=True)
    return output


class FusedALinearMethod(UnquantizedLinearMethod):
    """Unquantized linear taking the fused-A GEMM at measured token counts."""

    def __init__(self, tokens: frozenset[int]) -> None:
        super().__init__()
        self._tokens = tokens

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None:
            output = run_fused_a_gemm(self._tokens, x, layer.weight)
            if output is not None:
                return output
        return super().apply(layer, x, bias)


def install_fused_a_linear(
    module: nn.Module,
    dtype: torch.dtype,
    table: FusedATable,
) -> None:
    """Swap :class:`FusedALinearMethod` onto every layer ``table`` covers."""
    if dtype != torch.bfloat16:
        return
    for child in module.modules():
        if (
            not isinstance(child, LinearBase)
            or type(child.quant_method) is not UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        tokens = table.get((weight.shape[0], weight.shape[1]))
        if tokens:
            child.quant_method = FusedALinearMethod(tokens)
