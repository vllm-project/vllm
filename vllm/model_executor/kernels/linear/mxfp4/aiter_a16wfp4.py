# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Automatic gfx950 AITER A16WFP4 fallback for dynamic MXFP4 layers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

logger = init_logger(__name__)


def _prepare_aiter_a16wfp4_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> Any:
    # Keep the concrete import here: importing AITER/FlyDSL at module scope can
    # initialize HIP before the engine core forks.
    from aiter.ops.flydsl.gemm_a16wfp4 import prepare_gemm_a16wfp4_weight

    return prepare_gemm_a16wfp4_weight(weight, weight_scale)


# Registering this wrapper is safe because the concrete AITER/FlyDSL import is
# deferred until execution. Old AITER installs therefore remain importable and
# fall back while weights are prepared.
if is_aiter_found_and_supported():
    from vllm.utils.torch_utils import direct_register_custom_op

    def aiter_a16wfp4_gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        n: int,
        k: int,
    ) -> torch.Tensor:
        from aiter.ops.flydsl.gemm_a16wfp4 import (
            PreshuffledA16WFP4Weight,
            flydsl_gemm_a16wfp4,
        )

        prepared = PreshuffledA16WFP4Weight(
            weight=weight,
            scale=weight_scale,
            n=n,
            k=k,
        )
        return flydsl_gemm_a16wfp4(x, prepared)

    def aiter_a16wfp4_gemm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        n: int,
        k: int,
    ) -> torch.Tensor:
        del weight, weight_scale, k
        return torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)

    direct_register_custom_op(
        op_name="aiter_a16wfp4_gemm",
        op_func=aiter_a16wfp4_gemm,
        mutates_args=[],
        fake_impl=aiter_a16wfp4_gemm_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class AiterA16Wfp4LinearKernel(MxFp4LinearKernel):
    """QDQ activations, then use fused A16WFP4 or emulation automatically."""

    _logged_routes: set[tuple[str, int, int, str | None]] = set()

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.quant_dequant_func = quant_dequant_mxfp4

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "only supports ROCm"

        if compute_capability is None:
            capability = current_platform.get_device_capability()
            if capability is not None:
                compute_capability = capability[0] * 10 + capability[1]
        if compute_capability != 95:
            return False, f"requires gfx950, got capability {compute_capability}"
        if not is_aiter_found_and_supported():
            return False, "AITER is not found or supported"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "only supports dynamic MXFP4 activations"
        return True, None

    @classmethod
    def _log_route(
        cls,
        route: str,
        n: int,
        k: int,
        reason: str | None = None,
    ) -> None:
        key = (route, n, k, reason)
        if key in cls._logged_routes:
            return
        if reason is None:
            logger.info(
                "AiterA16Wfp4LinearKernel route=%s N=%d K=%d",
                route,
                n,
                k,
            )
        else:
            logger.warning(
                "AiterA16Wfp4LinearKernel route=%s N=%d K=%d: %s",
                route,
                n,
                k,
                reason,
            )
        cls._logged_routes.add(key)

    @staticmethod
    def _set_fallback_weights(
        layer: torch.nn.Module,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> str:
        if envs.VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD:
            decoded = dequant_mxfp4(weight, weight_scale, torch.bfloat16)
            layer.weight = Parameter(decoded.contiguous(), requires_grad=False)
            route = "cached-emulation"
        else:
            layer.weight = Parameter(weight, requires_grad=False)
            route = "per-call-emulation"
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        return route

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        n, packed_k = weight.shape
        k = packed_k * 2
        layer._aiter_a16wfp4_n = n
        layer._aiter_a16wfp4_k = k
        layer._aiter_a16wfp4_prepared = False

        fallback_reason: str | None = None
        if not envs.VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD:
            fallback_reason = "dequant-at-load is disabled"
        elif getattr(layer, "params_dtype", None) != torch.bfloat16:
            params_dtype = getattr(layer, "params_dtype", None)
            fallback_reason = f"layer params dtype is {params_dtype}, not BF16"
        elif n % 256 or k % 256:
            fallback_reason = "AITER requires N and K divisible by 256"
        else:
            try:
                prepared = _prepare_aiter_a16wfp4_weight(weight, weight_scale)
            except Exception as exc:
                fallback_reason = (
                    f"AITER FlyDSL preparation failed ({type(exc).__name__}: {exc})"
                )
            else:
                layer.weight = Parameter(prepared.weight, requires_grad=False)
                layer.weight_scale = Parameter(prepared.scale, requires_grad=False)
                layer._aiter_a16wfp4_prepared = True
                layer._aiter_a16wfp4_n = prepared.n
                layer._aiter_a16wfp4_k = prepared.k
                self._log_route("flydsl", n, k)
                return

        route = self._set_fallback_weights(layer, weight, weight_scale)
        self._log_route(route, n, k, fallback_reason)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qdq_x = self.quant_dequant_func(x)
        if not layer._aiter_a16wfp4_prepared:
            if layer.weight.element_size() >= 2:
                weight = layer.weight.to(qdq_x.dtype)
            else:
                weight = dequant_mxfp4(
                    layer.weight,
                    layer.weight_scale,
                    qdq_x.dtype,
                )
            return F.linear(qdq_x, weight, bias)

        original_shape = qdq_x.shape[:-1]
        qdq_x_2d = qdq_x.reshape(-1, qdq_x.shape[-1]).to(
            dtype=torch.bfloat16,
            memory_format=torch.contiguous_format,
        )
        y = torch.ops.vllm.aiter_a16wfp4_gemm(
            qdq_x_2d,
            layer.weight,
            layer.weight_scale,
            layer._aiter_a16wfp4_n,
            layer._aiter_a16wfp4_k,
        )
        y = y.reshape(*original_shape, layer._aiter_a16wfp4_n)
        if bias is not None:
            y = y + bias
        return y
