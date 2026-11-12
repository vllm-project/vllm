# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 BF16 projection choices measured for GLM-5.3-Flash TP4 decode on H20."""

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform

# (local N, K) -> measured token count and kernel layout. Larger batches keep
# cuBLAS. See benchmarks/kernels/benchmark_glm5next_decode.py.
_SM90_PROJECTIONS = {
    (6416, 4096): {
        1: SkinnyGemmConfig(1, 128, 2),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=2),
    },
    (4096, 2048): {1: SkinnyGemmConfig(1, 128, 2)},
    (2048, 4096): {
        1: SkinnyGemmConfig(1, 128, 2),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=2),
    },
    (4096, 1536): {1: SkinnyGemmConfig(1, 64, 4, k_unroll=2)},
}


class Glm5NextLowLatencyLinearMethod(UnquantizedLinearMethod):
    def __init__(self, plan: dict[int, SkinnyGemmConfig]) -> None:
        super().__init__()
        self.plan = plan

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config = self.plan.get(x.shape[0]) if x.ndim == 2 else None
        if (
            config is not None
            and bias is None
            and not envs.VLLM_BATCH_INVARIANT
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and layer.weight.is_contiguous()
            and layer.weight.dtype == x.dtype
            and layer.weight.device == x.device
        ):
            return shape_dynamic_skinny_gemm(x, layer.weight, config)
        return super().apply(layer, x, bias)


def enable_sm90_low_latency_gemm(module: nn.Module, dtype: torch.dtype) -> None:
    if (
        dtype != torch.bfloat16
        or envs.VLLM_BATCH_INVARIANT
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability(90)
        or not shape_dynamic_skinny_gemm.is_available()
    ):
        return
    configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        if (
            not isinstance(child, LinearBase)
            or type(child.quant_method) is not UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.ndim != 2 or weight.dtype != dtype:
            continue
        plan = _SM90_PROJECTIONS.get(tuple(weight.shape))
        if plan is not None:
            child.quant_method = Glm5NextLowLatencyLinearMethod(plan)
            configs.update(plan.values())
    if configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, configs)
