# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.base.w16a16 import Kernel as BaseKernel
from vllm.platforms import CpuArchEnum, current_platform

logger = init_logger(__name__)


def _check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    return (
        torch.cpu._is_amx_tile_supported()
        and (dtype in (torch.bfloat16, torch.int8))
        and k % 32 == 0
        and n % 16 == 0
    )


class Kernel(BaseKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "CPU platform not available"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.is_meta:
            layer.cpu_linear = torch.nn.functional.linear
            return

        N, K = layer.weight.size()
        dtype = layer.weight.dtype

        if current_platform.is_zen_cpu() and hasattr(
            torch.ops.zentorch, "zentorch_linear_unary"
        ):
            zen_weight = layer.weight.detach()
            is_prepacked = False

            if envs.VLLM_ZENTORCH_WEIGHT_PREPACK and hasattr(
                torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
            ):
                zen_weight = torch.ops.zentorch.zentorch_weight_prepack_for_linear(
                    zen_weight
                )
                is_prepacked = True

            layer.cpu_linear = lambda x, weight, bias, _p=is_prepacked: (
                torch.ops.zentorch.zentorch_linear_unary(
                    x, zen_weight, bias, is_weight_prepacked=_p
                )
            )
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            return

        if envs.VLLM_CPU_SGL_KERNEL and _check_cpu_sgl_kernel(N, K, dtype):
            packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
            bias_f32 = (
                layer.bias.to(torch.float32)
                if getattr(layer, "bias", None) is not None
                else None
            )
            layer.cpu_linear = lambda x, weight, bias: torch.ops._C.weight_packed_linear(
                x, packed_weight, bias_f32 if bias is not None else None, True
            )
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            return

        if (
            ops._supports_onednn
            and current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC
        ):
            try:
                handler = ops.create_onednn_mm(layer.weight.t(), 32)
                layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(
                    handler, x, bias
                )
                layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
                return
            except RuntimeError as e:
                logger.warning_once(
                    "Failed to create oneDNN linear, fallback to torch linear."
                    f" Exception: {e}"
                )

        layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
            x, weight, bias
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return layer.cpu_linear(x, layer.weight, bias)
