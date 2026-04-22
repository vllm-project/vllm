# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)


class TritonInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.is_cuda_alike():
            return True, None
        return False, "requires ROCm or CUDA."

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)

        azp_adj = params.azp_adj
        w_q, w_s = params.weight, params.weight_scale
        i_s, i_zp = params.input_scale, params.input_zero_point

        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=symmetric
        )

        out = triton_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )

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


class TritonFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda_alike():
            return False, "only cuda like devices are supported."
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
            A,
            B,
            As,
            Bs,
            list(self.weight_group_shape),
            self.config.out_dtype,
        )


# TODO we should be able to change the type of block_size to GroupShape
# after we resolve GroupShape compilation issue
# https://github.com/vllm-project/vllm/issues/25270
def _w8a8_triton_block_scaled_mm_func(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    return w8a8_triton_block_scaled_mm(
        qx, weight, x_scale, weight_scale, block_size, output_dtype
    )


def _w8a8_triton_block_scaled_mm_fake(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        (qx.size(0), weight.size(0)), dtype=output_dtype, device=qx.device
    )


direct_register_custom_op(
    "w8a8_triton_block_scaled_mm_func",
    _w8a8_triton_block_scaled_mm_func,
    fake_impl=_w8a8_triton_block_scaled_mm_fake,
)
