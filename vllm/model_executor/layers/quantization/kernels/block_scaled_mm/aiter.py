# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

from .BlockScaledMMKernel import Fp8BlockScaledMMKernel
from .triton import TritonBlockScaledMMKernel


class AiterBlockScaledMMKernel(Fp8BlockScaledMMKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return (
            rocm_aiter_ops.is_linear_enabled(),
            "Only supported on ROCm platform \
                with aiter package installed.",
        )

    @classmethod
    def ordered_fallback_kernels(cls) -> list[type["Fp8BlockScaledMMKernel"]]:
        return [TritonBlockScaledMMKernel]

    def process_weights_after_loading(self, layer):
        return super().process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale_inv = params.weight_scale_inv
        input_scale = params.input_scale
        scale_up = params.input_scale_ub

        n, k = weight.shape

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], weight.shape[0]]
        output_dtype = x.dtype

        use_triton = (
            not current_platform.is_fp8_fnuz()
            and rocm_aiter_ops.is_triton_gemm_w8a8_tuned(n, k)
        )

        q_input, input_scale = self.input_quant(
            input_2d, input_scale, scale_up, use_triton=use_triton
        )

        output = self.apply_block_scaled_mm(
            A=q_input,
            B=weight,
            out_dtype=output_dtype,
            As=input_scale,
            Bs=weight_scale_inv,
        )

        if bias is not None:
            output = output + bias
        return output.to(dtype=output_dtype).view(*output_shape)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        use_triton = kwargs.get("use_triton", False)

        if use_triton:
            gemm_a8w8_blockscale_op = rocm_aiter_ops.triton_gemm_a8w8_blockscale
        else:
            gemm_a8w8_blockscale_op = rocm_aiter_ops.gemm_a8w8_blockscale

        return gemm_a8w8_blockscale_op(
            A, B, As, Bs, list(self.weight_group_shape), output_dtype=out_dtype
        )
