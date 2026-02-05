# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from .cutlass import CutlassInt8ScaledMMLinearKernel
from .ScaledMMLinearKernel import Int8ScaledMMLinearLayerConfig


class AiterInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "Requires ROCm."

        if compute_capability is not None and compute_capability < 90:
            return False, "requires compute capability 90 and above."

        try:
            import aiter  # noqa: F401 # deliberately attempt to import aiter
        except Exception:
            return False, "requires `aiter` to be installed."

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
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        assert symmetric, (
            "AiterInt8ScaledMMLinearKernel only supports symmetric quantization."
        )
        x_q, x_s, x_zp = ops.scaled_int8_quant(x, i_s, i_zp, symmetric=symmetric)

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
        return rocm_aiter_ops.gemm_a8w8(x_q, w_q.t(), x_s, w_s, bias, out_dtype)


class AiterFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return (
            rocm_aiter_ops.is_linear_enabled(),
            "Only supported on ROCm platform \
                with aiter package installed.",
        )

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
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
        # TODO This import is to avoid circular import
        # this import can be global
        # after all scaled MM kernels inherit from base
        from .triton import TritonFp8BlockScaledMMKernel

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
