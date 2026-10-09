# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2 decode GEMM selection for unquantized BF16 on SM10x (incl. B200/SM100)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

Backend = Literal["cute", "dsv3_fused_a"]
ResolvedCall = tuple[Backend, SkinnyGemmConfig | None]

# Short human-readable description used in the startup enable/disable log lines.
FEATURE_NAME = "GLM-5.2 low-latency GEMM plan"

# Environment-variable kill switch for the whole GLM-5.2 low-latency GEMM plan.
# Setting it to "1" disables the feature entirely and falls back to the stock
# UnquantizedLinearMethod -> F.linear (cuBLASLt) path.
VLLM_DISABLE_GLM52_LOW_LATENCY_GEMM = "VLLM_DISABLE_GLM52_LOW_LATENCY_GEMM"


def _feature_disabled_by_env() -> bool:
    return os.getenv(VLLM_DISABLE_GLM52_LOW_LATENCY_GEMM, "0") == "1"


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

# Dense gate_up (layers 0-2 at TP=4): cute M=1 only, no dsv3 specialization.
GLM52_DENSE_GATE_UP_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=6144,
    cute_configs=((1, SkinnyGemmConfig(1, 128, 4, static_k=6144)),),
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
        GLM52_DENSE_GATE_UP_PROJECTION,
        GLM52_EH_PROJECTION,
    )
}


def _is_sm10x() -> bool:
    return current_platform.is_device_capability_family(100)


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
    # Range guard (not equality): avoids ConstraintViolationError under torch.compile.
    if x.shape[0] > 16:
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
    if _feature_disabled_by_env():
        return None
    if dtype != torch.bfloat16 or not _is_sm10x():
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
    if _feature_disabled_by_env():
        logger.info("%s is DISABLED (env override)", FEATURE_NAME)
        return
    if dtype != torch.bfloat16 or not _is_sm10x():
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    installed = 0
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
        logger.debug(
            "GLM-5.2 low-latency GEMM: %s shape=%s",
            child.prefix,
            tuple(weight.shape),
        )
        child.quant_method = GLM52LowLatencyLinearMethod(spec.build_plan())
        installed += 1
        warmup_configs.update(config for _, config in spec.cute_configs)

    if installed == 0:
        logger.warning(
            "GLM-5.2 low-latency GEMM plan is ENABLED but no projection matched "
            "(0 switched)"
        )
    else:
        logger.info(
            "%s is ENABLED (%d projection(s) switched)", FEATURE_NAME, installed
        )
    _request_warmup(dtype, warmup_configs)
