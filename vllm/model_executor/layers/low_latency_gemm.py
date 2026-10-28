# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-provided skinny GEMM plans for unquantized linear layers."""

from collections.abc import Callable

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


class _LowLatencyApply:
    def __init__(
        self, op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        super().__init__()
        self._op = op

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return self._op(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class LowLatencyLinearMethod(_LowLatencyApply, UnquantizedLinearMethod):
    pass


class LowLatencyEmbeddingMethod(_LowLatencyApply, UnquantizedEmbeddingMethod):
    pass


# Excluding lm_head from quantization can give it UnquantizedLinearMethod
# instead of UnquantizedEmbeddingMethod. Preserve the original method family.
_LOW_LATENCY_METHODS: dict[type, type] = {
    UnquantizedLinearMethod: LowLatencyLinearMethod,
    UnquantizedEmbeddingMethod: LowLatencyEmbeddingMethod,
}


def _low_latency_gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


class LowLatencyGemm:
    """Bind a model's measured plans to a custom op and layer installer.

    Create one instance at module scope with a unique op name. Calling
    ``enable`` on multiple modules reuses the registered op.
    """

    def __init__(
        self,
        *,
        op_name: str,
        plans_by_capability: dict[
            tuple[int, int] | None, dict[tuple[int, int], dict[int, SkinnyGemmConfig]]
        ],
    ) -> None:
        self._plans_by_capability = plans_by_capability

        def low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return self._run(x, weight)

        direct_register_custom_op(
            op_name=op_name,
            op_func=low_latency_gemm,
            fake_impl=_low_latency_gemm_fake,
        )
        self._op = getattr(torch.ops.vllm, op_name).default

    def _run(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        plans = self._plans_by_capability.get(
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

    def enable(self, module: nn.Module, dtype: torch.dtype) -> None:
        plans = self._plans_by_capability.get(
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
            child.quant_method = method_cls(self._op)
            warmup_configs.update(plan.values())

        if warmup_configs:
            shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)
