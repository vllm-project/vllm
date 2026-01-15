# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

from .cutlass import CutlassScaledMMLinearKernel
from .ScaledMMLinearKernel import ScaledMMLinearLayerConfig


class AiterScaledMMLinearKernel(CutlassScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return (
                False,
                "AiterScaledMMLinearKernel requires `aiter` which is not "
                + "currently supported on non-ROCm platform.",
            )
        if compute_capability is None:
            _cc = current_platform.get_device_capability()
            if _cc is not None:
                compute_capability = _cc.major * 10 + _cc.minor
        if compute_capability is not None and compute_capability < 90:
            return False, f"requires capability 90, got {compute_capability}"

        try:
            import aiter  # noqa: F401 # deliberately attempt to import aiter
        except Exception:
            return (
                False,
                "AiterScaledMMLinearKernel requires `aiter` which is not "
                + "installed on ROCm.",
            )

        if not rocm_aiter_ops.is_linear_enabled():
            return (
                False,
                "AiterScaledMMLinearKernel is disabled. "
                + "Enable by setting `VLLM_ROCM_USE_AITER=1` "
                + "and `VLLM_ROCM_USE_AITER_LINEAR=1`. "
                + "`VLLM_ROCM_USE_AITER_LINEAR` default is True.",
            )

        return True, None

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return (
                False,
                "AiterScaledMMLinearKernel only supports symmetric " + "quantization.",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        `AiterScaledMMLinearKernel` implements a fused version of
            `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
        where scale_a * a and scale_b * b are implemented using numpy-style
        broadcasting.
        Currently only support per-tensor-per-tensor GEMM
        and per-token-per-channel GEMM through AITER
        w8a8 scaled gemm. `AiterScaledMMLinearKernel` also does not support
        ATIER block scaled GEMM and mix-precision GEMM.
        """
        w_q, w_s, i_s, i_zp, azp_adj = self._get_weight_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        assert symmetric, (
            "AiterScaledMMLinearKernel only supports symmetric quantization."
        )
        x_q, x_s, x_zp = ops.scaled_int8_quant(x, i_s, i_zp, symmetric=symmetric)

        assert x_zp is None, (
            "AiterScaledMMLinearKernel only supports symmetric quantization."
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
            + " and per-token-per-channel GEMM through AITER"
            " w8a8 scaled gemm. `AiterScaledMMLinearKernel` "
            + "does not support AITER block scaled GEMM."
        )

        # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects
        # a to be [M, K]
        # b to be [N, K]
        # CutlassScaledMMLinearKernel prepare weight `w_q` in [K, N] format
        return rocm_aiter_ops.gemm_a8w8(x_q, w_q.t(), x_s, w_s, bias, out_dtype)
