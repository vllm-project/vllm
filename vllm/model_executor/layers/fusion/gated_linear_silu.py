# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured gate/up GEMM epilogue for unquantized SM89 dense prefills."""

import torch
import torch.nn.functional as F

from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.utils import cuda_unquantized_gemm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _gated_linear_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        x.ndim == 2
        and weight.shape == (18432, 2560)
        and x.shape[1] == 2560
        and 768 <= x.shape[0] <= 4096
        and x.dtype == weight.dtype == torch.bfloat16
        and x.device == weight.device
        and x.is_contiguous()
        and weight.is_contiguous()
    ):
        from vllm.kernels.triton.gated_linear_silu import gated_linear_silu

        return gated_linear_silu(x, weight)
    projected = F.linear(x, weight)
    out = torch.empty(
        (*projected.shape[:-1], projected.shape[-1] // 2),
        dtype=projected.dtype,
        device=projected.device,
    )
    torch.ops._C.silu_and_mul(out, projected)
    return out


def _gated_linear_silu_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0] // 2))


direct_register_custom_op(
    op_name="gated_linear_silu",
    op_func=_gated_linear_silu,
    fake_impl=_gated_linear_silu_fake,
    dispatch_key="CUDA",
)


@CustomOp.register("gated_linear_silu")
class GatedLinearSilu(CustomOp):
    def __init__(self):
        super().__init__(compile_native=False)

    def forward_native(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return SiluAndMul.forward_native(F.linear(x, weight))

    def forward_cuda(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.gated_linear_silu(x, weight)


def create_gated_linear_silu(
    gate_up: MergedColumnParallelLinear,
    down: RowParallelLinear,
    activation: SiluAndMul,
) -> GatedLinearSilu | None:
    config = get_current_vllm_config_or_none()
    if (
        config is None
        or "+gated_linear_silu" not in config.compilation_config.custom_ops
        or not current_platform.is_cuda()
        or current_platform.get_device_capability() != (8, 9)
        or envs.VLLM_BATCH_INVARIANT
        or config.lora_config is not None
        or config.parallel_config.tensor_parallel_size != 1
        or config.kernel_config.linear_backend != "auto"
        or config.offload_config.uva.cpu_offload_gb > 0
        or config.offload_config.prefetch.offload_group_size > 0
        or not GatedLinearSilu.enabled()
        or not SiluAndMul.enabled()
        or type(activation) is not SiluAndMul
        or type(gate_up) is not MergedColumnParallelLinear
        or type(down) is not RowParallelLinear
        or gate_up.tp_size != 1
        or down.tp_size != 1
        or type(gate_up.quant_method) is not UnquantizedLinearMethod
        or type(down.quant_method) is not UnquantizedLinearMethod
        or gate_up.quant_method._gemm_impl is not cuda_unquantized_gemm
        or down.quant_method._gemm_impl is not cuda_unquantized_gemm
        or gate_up.bias is not None
        or down.bias is not None
        or gate_up.weight.shape != (18432, 2560)
        or down.weight.shape != (2560, 9216)
        or gate_up.weight.dtype != torch.bfloat16
        or down.weight.dtype != torch.bfloat16
        or not gate_up.weight.is_cuda
        or gate_up.weight.device != down.weight.device
    ):
        return None
    return GatedLinearSilu()
