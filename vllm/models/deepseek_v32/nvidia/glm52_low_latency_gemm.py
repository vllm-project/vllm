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
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform

Backend = Literal["cute", "dsv3_fused_a"]
ResolvedCall = tuple[Backend, SkinnyGemmConfig | None]


@dataclass(frozen=True, slots=True)
class GLM52ProjectionSpec:
    n: int
    k: int
    cute_configs: tuple[tuple[int, SkinnyGemmConfig], ...]
    dsv3_tokens: frozenset[int] = frozenset()

    def build_plan(self) -> dict[int, ResolvedCall]:
        plan: dict[int, ResolvedCall] = {
            num_tokens: ("cute", config) for num_tokens, config in self.cute_configs
        }
        plan.update(
            (num_tokens, ("dsv3_fused_a", None)) for num_tokens in self.dsv3_tokens
        )
        return plan


GLM52_QKV_A_PROJECTION = GLM52ProjectionSpec(
    n=2624,
    k=6144,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 128, 4, static_k=6144)),
        (2, SkinnyGemmConfig(2, 128, 2)),
    ),
    dsv3_tokens=frozenset(range(3, 17)),
)

GLM52_Q_B_PROJECTION = GLM52ProjectionSpec(
    n=2048,
    k=2048,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 128, 4, static_k=2048)),
        (2, SkinnyGemmConfig(2, 64, 2, k_unroll=2)),
    ),
    dsv3_tokens=frozenset(range(3, 17)),
)

# The MTP eh_proj is a plain nn.Linear, so it gets its plan through
# build_glm52_plan rather than a quant_method swap. cuBLAS wins from M=4.
GLM52_EH_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=12288,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 256, 2, vector_width=4, static_k=12288)),
        (2, SkinnyGemmConfig(2, 64, 2)),
        (3, SkinnyGemmConfig(3, 64, 2)),
    ),
)

GLM52_PROJECTIONS = {
    (spec.n, spec.k): spec
    for spec in (
        GLM52_QKV_A_PROJECTION,
        GLM52_Q_B_PROJECTION,
        GLM52_EH_PROJECTION,
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
    plan: dict[int, ResolvedCall] | None,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    if plan is None or not _runtime_ok(x, weight):
        return None
    entry = plan.get(x.shape[0])
    if entry is None:
        return None

    backend, config = entry
    if backend == "cute":
        if not shape_dynamic_skinny_gemm.is_available():
            return None
        return shape_dynamic_skinny_gemm(x, weight, config)

    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        return None
    output = torch.empty(
        (x.shape[0], weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )
    ops.dsv3_fused_a_gemm(output, x, weight.t(), enable_pdl=True)
    return output


def _request_warmup(dtype: torch.dtype, configs: set[SkinnyGemmConfig]) -> None:
    if configs and shape_dynamic_skinny_gemm.is_available():
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, configs)


def build_glm52_plan(
    weight: torch.Tensor | None, dtype: torch.dtype
) -> dict[int, ResolvedCall] | None:
    """Plan for a weight the walk below cannot reach (a plain ``nn.Linear``)."""
    if dtype != torch.bfloat16 or not _is_sm103():
        return None
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return None
    spec = GLM52_PROJECTIONS.get(tuple(weight.shape))
    if spec is None:
        return None
    _request_warmup(dtype, {config for _, config in spec.cute_configs})
    return spec.build_plan()


class GLM52LowLatencyLinearMethod(UnquantizedLinearMethod):
    def __init__(self, plan: dict[int, ResolvedCall]) -> None:
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

    warmup_configs: set[SkinnyGemmConfig] = set()
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
        warmup_configs.update(config for _, config in spec.cute_configs)

    _request_warmup(dtype, warmup_configs)
