# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import (
    rocm_aiter_ops,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    prepare_int8_scaled_mm_weights,
)
from vllm.platforms import current_platform

from .base import (
    Int8LinearKernel,
    Int8LinearLayerConfig,
)


class AiterInt8LinearKernel(Int8LinearKernel):
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
                (
                    "requires setting `VLLM_ROCM_USE_AITER=1` "
                    "and `VLLM_ROCM_USE_AITER_LINEAR=1`. "
                    "`VLLM_ROCM_USE_AITER_LINEAR` default is True."
                ),
            )
        return True, None

    @classmethod
    def can_implement(cls, c: Int8LinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "supports symmetric quantization only."
        # AITER actually supports only per-token-per-channel GEMM.
        # Fused per-tensor weight scales become per-channel after loading
        # (see `prepare_int8_scaled_mm_weights`).
        weight_per_channel = c.is_channelwise or c.is_fused_module
        if c.is_static_input_scheme or not weight_per_channel:
            return (
                False,
                "supports only per-token activations with per-channel weights.",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepare_int8_scaled_mm_weights(layer, self.config, self.layer_param_names)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`AiterInt8LinearKernel` implements a fused version of
            `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
        where scale_a * a and scale_b * b are implemented using numpy-style
        broadcasting.
        Currently only supports per-token-per-channel GEMM through AITER
        w8a8 scaled gemm. `AiterInt8LinearKernel` also does not support
        AITER block scaled GEMM and mix-precision GEMM.
        """
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        assert symmetric, "AiterInt8LinearKernel only supports symmetric quantization."
        x_q, x_s, x_zp = ops.scaled_int8_quant(x, i_s, i_zp, symmetric=symmetric)

        assert x_zp is None, (
            "AiterInt8LinearKernel only supports symmetric quantization."
        )
        out_dtype = x.dtype

        assert w_q.shape[0] % 16 == 0 and w_q.shape[1] % 16 == 0
        assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
        assert bias is None or bias.shape[0] == w_q.shape[1] and bias.dtype == out_dtype

        m = x_q.shape[0]  # a
        n = w_q.shape[1]  # b

        per_token_scale_a = x_s.numel() == m
        per_channel_scale_b = w_s.numel() == n
        assert per_token_scale_a and per_channel_scale_b, (
            "Currently only support per-token-per-channel GEMM through AITER"
            " w8a8 scaled gemm."
        )

        # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects
        # a to be [M, K]
        # b to be [N, K]
        # prepare_int8_scaled_mm_weights leaves `w_q` in [K, N] format
        return rocm_aiter_ops.w8a8_gemm(x_q, w_q.t(), x_s, w_s, bias, out_dtype)
