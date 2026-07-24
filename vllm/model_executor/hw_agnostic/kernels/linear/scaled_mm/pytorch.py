# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.config import CompilationMode, get_current_vllm_config
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


def _get_num_tokens(output_shape: list) -> int:
    return math.prod(output_shape[:-1])


class TorchFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (current_platform.is_cuda_alike() or current_platform.is_cpu()):
            return False, "requires ROCm, CUDA or CPU."

        if compute_capability is not None and compute_capability < 89:
            return False, "requires compute capability 89 and above."

        return True, None

    def get_output_padding(self) -> int | None:
        vllm_config = get_current_vllm_config().compilation_config
        pad_output = vllm_config.mode < CompilationMode.VLLM_COMPILE
        return 17 if pad_output else None


class ChannelWiseTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):
    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if per_tensor_activation_scales and per_tensor_weight_scales:
            return False, "cannot be used with per tensor activation and weight scales."

        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        dummy_tensor = torch.ones(1, dtype=torch.float32, device=A.device)

        output = torch._scaled_mm(
            A,
            B,
            scale_a=dummy_tensor,
            scale_b=dummy_tensor,
            out_dtype=torch.float32,
        )
        if type(output) is tuple and len(output) == 2:
            output = output[0]

        num_tokens = _get_num_tokens(output_shape)
        output = torch.narrow(output, 0, 0, num_tokens)
        x_scale = torch.narrow(As, 0, 0, num_tokens)

        output = output * x_scale * Bs.t()
        if bias is not None:
            output = output + bias
        return output.to(out_dtype).view(*output_shape)
