# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash decode GEMM selection on NVIDIA GPUs.

Dispatch follows Qwen4Exp and uses the local ``(N, K)`` shape and token count.
Plans contain measured CUDA graph capture sizes; other token counts use the
standard linear implementation.
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    row_stride_ok,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

# DGX Spark (GB10) plans for TP=2 local shapes.
GLM5NEXT_SM121_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # Shared-expert fused gate/up and MLA fused q_a/kv_a projections, TP=2.
    (2048, 4096): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, static_k=4096),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4, vector_width=4, static_k=4096),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=2, static_k=4096),
    },
    # Shared-expert down projection, TP=2.
    (4096, 1024): {
        1: SkinnyGemmConfig(1, 64, 1, vector_width=2, static_k=1024),
        2: SkinnyGemmConfig(2, 64, 1, vector_width=2, static_k=1024),
        4: SkinnyGemmConfig(4, 64, 1, vector_width=2, static_k=1024),
    },
    # KDA output projection, TP=2.
    (4096, 4096): {
        1: SkinnyGemmConfig(1, 32, 2, k_unroll=4, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4, vector_width=2, static_k=4096),
        4: SkinnyGemmConfig(4, 64, 1, vector_width=2, static_k=4096),
    },
    # MLA output projection, TP=2.
    (4096, 8192): {
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, vector_width=2, static_k=8192),
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=2, static_k=8192),
    },
    # MLA q_b projection, TP=2.
    (8192, 1536): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, vector_width=2, static_k=1536),
        2: SkinnyGemmConfig(2, 32, 1, k_unroll=2, vector_width=2, static_k=1536),
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=2, vector_width=2, static_k=1536),
    },
    # KDA fused QKV/beta/f_a/g_a input projection, TP=2.
    (12576, 4096): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=2, vector_width=2),
        2: SkinnyGemmConfig(2, 256, 1),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4, static_k=4096),
    },
    # LM head, TP=2.
    (77440, 4096): {
        1: SkinnyGemmConfig(1, 256, 1, static_k=4096),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=2),
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=2),
    },
}


GLM5NEXT_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int] | None, dict[tuple[int, int], dict[int, SkinnyGemmConfig]]
] = {
    (12, 1): GLM5NEXT_SM121_GEMM_PLANS,
}


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(
    x: torch.Tensor, weight: torch.Tensor, config: SkinnyGemmConfig
) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and x.dim() == 2
        and row_stride_ok(x, config)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


class _Glm5NextLowLatencyApply:
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.glm5next_low_latency_gemm(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Glm5NextLowLatencyLinearMethod(_Glm5NextLowLatencyApply, UnquantizedLinearMethod):
    pass


class Glm5NextLowLatencyEmbeddingMethod(
    _Glm5NextLowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


# A quant config that excludes ``lm_head`` (the modelopt NVFP4 checkpoints do)
# hands ParallelLMHead an UnquantizedLinearMethod rather than an
# UnquantizedEmbeddingMethod, so the replacement is chosen by the method the
# layer actually carries.
_LOW_LATENCY_METHODS: dict[type, type] = {
    UnquantizedLinearMethod: Glm5NextLowLatencyLinearMethod,
    UnquantizedEmbeddingMethod: Glm5NextLowLatencyEmbeddingMethod,
}


def _glm5next_low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    plans = GLM5NEXT_GEMM_PLANS_BY_CAPABILITY.get(
        current_platform.get_device_capability(), {}
    )
    plan = plans.get((weight.shape[0], weight.shape[1]))
    config = None if plan is None else plan.get(x.shape[0])
    if (
        config is not None
        and _runtime_ok(x, weight, config)
        and shape_dynamic_skinny_gemm.is_available()
    ):
        return shape_dynamic_skinny_gemm(x, weight, config)
    return torch.nn.functional.linear(x, weight)


def _glm5next_low_latency_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="glm5next_low_latency_gemm",
    op_func=_glm5next_low_latency_gemm,
    fake_impl=_glm5next_low_latency_gemm_fake,
)


def enable_glm5next_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    plans = GLM5NEXT_GEMM_PLANS_BY_CAPABILITY.get(
        current_platform.get_device_capability(), {}
    )
    if dtype != torch.bfloat16 or not plans:
        return
    if not shape_dynamic_skinny_gemm.is_available():
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        if not isinstance(child, (LinearBase, ParallelLMHead)):
            continue
        method_cls = _LOW_LATENCY_METHODS.get(type(child.quant_method))
        if method_cls is None:
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        plan = plans.get((weight.shape[0], weight.shape[1]))
        if plan is None:
            continue
        child.quant_method = method_cls()
        warmup_configs.update(plan.values())

    if warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)
