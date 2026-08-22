# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2 decode GEMM selection for unquantized BF16 on SM103."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform

Backend = Literal["dsv3_fused_a"]


@dataclass(frozen=True, slots=True)
class GLM52ProjectionSpec:
    n: int
    k: int
    dsv3_tokens: frozenset[int] = frozenset()

    def build_plan(self) -> dict[int, Backend]:
        return {num_tokens: "dsv3_fused_a" for num_tokens in self.dsv3_tokens}


GLM52_QKV_A_PROJECTION = GLM52ProjectionSpec(
    n=2624,
    k=6144,
    dsv3_tokens=frozenset(range(3, 17)),
)

GLM52_Q_B_PROJECTION = GLM52ProjectionSpec(
    n=2048,
    k=2048,
    dsv3_tokens=frozenset(range(3, 17)),
)

GLM52_PROJECTIONS = {
    (spec.n, spec.k): spec
    for spec in (
        GLM52_QKV_A_PROJECTION,
        GLM52_Q_B_PROJECTION,
    )
}


def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))


def _is_supported_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and _is_supported_row_major(x)
        and _is_supported_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def run_glm52_plan(
    plan: dict[int, Backend] | None,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    if plan is None or not _runtime_ok(x, weight):
        return None
    if plan.get(x.shape[0]) is None:
        return None
    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        return None
    output = torch.empty(
        (x.shape[0], weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )
    ops.dsv3_fused_a_gemm(output, x, weight.t(), enable_pdl=True)
    return output


def build_glm52_plan(
    weight: torch.Tensor | None, dtype: torch.dtype
) -> dict[int, Backend] | None:
    """Plan for a weight the walk below cannot reach (a plain ``nn.Linear``)."""
    if dtype != torch.bfloat16 or not _is_sm103():
        return None
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return None
    spec = GLM52_PROJECTIONS.get(tuple(weight.shape))
    return spec.build_plan() if spec is not None else None


class GLM52LowLatencyLinearMethod(UnquantizedLinearMethod):
    def __init__(self, plan: dict[int, Backend]) -> None:
        super().__init__()
        self._plan = plan

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None:
            output = run_glm52_plan(self._plan, x, layer.weight)
            if output is not None:
                return output
        return super().apply(layer, x, bias)


def enable_glm52_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    if dtype != torch.bfloat16 or not _is_sm103():
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
        spec = GLM52_PROJECTIONS.get(tuple(weight.shape))
        if spec is None:
            continue
        child.quant_method = GLM52LowLatencyLinearMethod(spec.build_plan())
