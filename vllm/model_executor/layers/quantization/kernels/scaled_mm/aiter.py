# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import Fp8BlockMMScaledConfig, Fp8BlockScaledMMKernel
from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from .triton import TritonFp8BlockScaledMMKernel


class AiterInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is not None and compute_capability < 90:
            return False, "requires compute capability 90 and above."

        # IS_AITER_FOUND checks both platform compability
        # and existence of aiter package.
        if not IS_AITER_FOUND:
            return False, "Requires ROCm platfrom with installed aiter package."

        if not rocm_aiter_ops.is_linear_enabled():
            return (
                False,
                "requires setting `VLLM_ROCM_USE_AITER=1` "
                "and `VLLM_ROCM_USE_AITER_LINEAR=1`. "
                "`VLLM_ROCM_USE_AITER_LINEAR` default is True.",
            )
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "supports symmetric quantization only."
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        `AiterInt8ScaledMMLinearKernel` implements a fused version of
            `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
        where scale_a * a and scale_b * b are implemented using numpy-style
        broadcasting.
        Currently only support per-tensor-per-tensor GEMM
        and per-token-per-channel GEMM through AITER
        w8a8 scaled gemm. `AiterInt8ScaledMMLinearKernel` also does not support
        ATIER block scaled GEMM and mix-precision GEMM.
        """
        params = self._get_layer_params(layer)
        w_q, w_s = params.weight, params.weight_scale
        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = params.azp_adj is None
        assert symmetric, (
            "AiterInt8ScaledMMLinearKernel only supports symmetric quantization."
        )
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x, params.input_scale, params.input_zero_point, symmetric=symmetric
        )

        assert x_zp is None, (
            "AiterInt8ScaledMMLinearKernel only supports symmetric quantization."
        )
        out_dtype = x.dtype

        assert w_q.shape[0] % 16 == 0 and w_q.shape[1] % 16 == 0
        assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
        assert bias is None or bias.shape[0] == w_q.shape[1] and bias.dtype == out_dtype

        m = x_q.shape[0]  # a
        n = w_q.shape[1]  # b

        per_tensor_scale_a = x_s.numel() == 1
        per_tensor_scale_b = w_s.numel() == 1
        per_token_scale_a = x_s.numel() == m
        per_channel_scale_b = w_s.numel() == n

        # @TODO:
        # Maybe broadcast the per-tensor-scale into per-channel-scale
        # if one of the scale is a per-channel-scale.
        # For now, it only supports:
        # - per-tensor-per-tensor a8w8 scaled GEMM, and
        # - per-token-per-channel a8w8 scaled GEMM
        assert (per_tensor_scale_a and per_tensor_scale_b) or (
            per_token_scale_a and per_channel_scale_b
        ), (
            "Currently only support per-tensor-per-tensor GEMM "
            " and per-token-per-channel GEMM through AITER"
            " w8a8 scaled gemm. `AiterInt8ScaledMMLinearKernel` "
            "does not support AITER block scaled GEMM."
        )

        # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects
        # a to be [M, K]
        # b to be [N, K]
        # CutlassInt8ScaledMMLinearKernel prepare weight `w_q` in [K, N] format
        return self.apply_scaled_mm(
            A=x_q,
            B=w_q.t(),
            As=x_s,
            Bs=w_s,
            bias=bias,
            out_dtype=out_dtype,
            output_shape=[],
        )

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
        return rocm_aiter_ops.gemm_a8w8(A, B.t(), As, Bs, bias, out_dtype)


class AiterFp8BlockScaledMMKernel(Fp8BlockScaledMMKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return (
            rocm_aiter_ops.is_linear_enabled(),
            "Only supported on ROCm platform \
                with aiter package installed.",
        )

    @classmethod
    def can_implement(cls, config: Fp8BlockMMScaledConfig):
        act_quant_desc = config.activation_quant_key.scale
        if (
            act_quant_desc.group_shape != GroupShape(1, 12)
            and not act_quant_desc.static
        ):
            return (
                False,
                "Supports only dynamic per token group activation \
                quantization with group_shape=(1,12).",
            )

    @classmethod
    def ordered_fallback_kernels(cls):
        return [TritonFp8BlockScaledMMKernel]

    def apply_weights(
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

        q_input, input_scale = self.input_quant_op(
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
