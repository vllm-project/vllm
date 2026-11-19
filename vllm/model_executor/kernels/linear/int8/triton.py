# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    prepare_int8_scaled_mm_weights,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .base import (
    Int8LinearKernel,
    Int8LinearLayerConfig,
)


class TritonInt8LinearKernel(Int8LinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            return True, None
        return False, "requires ROCm, CUDA or XPU."

    @classmethod
    def can_implement(cls, c: Int8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepare_int8_scaled_mm_weights(layer, self.config, self.layer_param_names)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=symmetric
        )

        out = torch.ops.vllm.int8_scaled_mm_func(x_q, w_q, x_s, w_s, x.dtype, bias)

        if azp_adj is not None:
            # Asymmetric quantization: subtract the zero-point correction.
            # D = scale_a * scale_b * (A_q @ B_q - azp * azp_adj) + bias
            # triton_scaled_mm already computed scale_a * scale_b * (A_q @ B_q) + bias
            # so we subtract scale_a * scale_b * azp * azp_adj
            #
            # x_s: [M, 1] or scalar, w_s: [N, 1] or scalar, azp_adj: [1, N]
            # Reshape w_s from [N, 1] to [1, N] for proper broadcasting.
            w_s_row = w_s.view(1, -1) if w_s.dim() > 0 else w_s
            static = i_zp is not None
            if not static and x_zp is not None:
                # Dynamic per-token: azp is per-token, azp_adj is per-channel
                # x_zp: [M, 1], azp_adj: [1, N]
                out -= x_s * w_s_row * (x_zp * azp_adj).to(x.dtype)
            else:
                # Static per-tensor: azp already folded into azp_adj
                out -= (x_s * w_s_row * azp_adj).to(x.dtype)

        return out


def _int8_scaled_mm_func(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return triton_scaled_mm(
        qx,
        weight,
        scale_a=x_scale,
        scale_b=weight_scale,
        out_dtype=output_dtype,
        bias=bias,
    )


def _int8_scaled_mm_fake(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(
        (qx.size(0), weight.size(1)), dtype=output_dtype, device=qx.device
    )


direct_register_custom_op(
    "int8_scaled_mm_func",
    _int8_scaled_mm_func,
    fake_impl=_int8_scaled_mm_fake,
)
