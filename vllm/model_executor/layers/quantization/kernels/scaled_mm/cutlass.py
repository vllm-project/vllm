# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)


class CutlassInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        return True, None

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        config = self.config

        # INPUT SCALE
        if config.is_static_input_scheme and not config.input_symmetric:
            input_scale = params.input_scale
            i_s_name = params.INPUT_SCALE
            i_zp_name = params.INPUT_ZERO_POINT
            input_zero_point = params.input_zero_point

            assert input_zero_point is not None
            # reconstruct the ranges
            int8_traits = torch.iinfo(torch.int8)
            azps = input_zero_point.to(dtype=torch.int32)
            range_max = (input_scale * (int8_traits.max - azps)).max()
            range_min = (input_scale * (int8_traits.min - azps)).min()

            scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
            replace_parameter(
                layer, i_s_name, torch.nn.Parameter(scale, requires_grad=False)
            )

            # AZP loaded as int8 but used as int32
            azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
            replace_parameter(
                layer, i_zp_name, torch.nn.Parameter(azp, requires_grad=False)
            )

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # For more details, see csrc/quantization/w8a8/cutlass/Epilogues.md
        # https://github.com/vllm-project/vllm/blob/main/csrc/quantization/w8a8/cutlass/Epilogues.md
        if not config.input_symmetric:
            weight = getattr(layer, params.WEIGHT)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if config.is_static_input_scheme:
                # cutlass_w8a8 requires azp to be folded into azp_adj
                # in the per-tensor case
                azp_adj = getattr(layer, i_zp_name) * azp_adj
            setattr(
                layer,
                params.AZP_ADJ,
                torch.nn.Parameter(azp_adj, requires_grad=False),
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = params.azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(),
            params.input_scale,
            params.input_zero_point,
            symmetric=symmetric,
        )

        if x_zp is not None:
            # Currently, static is always per-tensor and dynamic is per-token
            static = params.input_zero_point is not None
            azp = None if static else x_zp
            return ops.cutlass_scaled_mm_azp(
                x_q,
                params.weight,
                scale_a=x_s,
                scale_b=params.weight_scale,
                out_dtype=x.dtype,
                azp_adj=params.azp_adj,
                azp=azp,
                bias=bias,
            )
        return ops.cutlass_scaled_mm(
            x_q,
            params.weight,
            scale_a=x_s,
            scale_b=params.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )


class CutlassFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
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
        # Fused GEMM_DQ
        output = ops.cutlass_scaled_mm(
            A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
        )
        return output.view(*output_shape)


class CutlassFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    is_hopper: bool = current_platform.is_device_capability(90)

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not CUTLASS_BLOCK_FP8_SUPPORTED:
            return (
                False,
                f"The device compute capability of \
                {compute_capability} is not supported.",
            )

    @classmethod
    def ordered_fallback_kernels(cls) -> list[type["Fp8BlockScaledMMLinearKernel"]]:
        # TODO This import is to avoid circular import
        # this import can be global
        # after all scaled MM kernels inherit from base
        from .triton import TritonFp8BlockScaledMMKernel

        return [TritonFp8BlockScaledMMKernel]

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.is_hopper:
            return torch.ops.vllm.padded_cutlass(
                A,
                B,
                As,
                Bs,
                list(self.weight_group_shape),
                out_dtype,
            )
        else:
            return ops.cutlass_scaled_mm(
                A,
                B.T,
                out_dtype=out_dtype,
                scale_a=As,
                scale_b=Bs.T,
            )


# We need to pass in the is_hopper flag as argument because the function
# current_platform.is_device_capability() is not supported by Torch compiler.
def cutlass_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return ops.cutlass_scaled_mm(
        A,
        B.T,
        out_dtype=output_dtype,
        scale_a=As,
        scale_b=Bs.T,
    )


def _padded_cutlass(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    pad_multiple = 4
    dim = qx.shape[0]
    padded = (
        dim if dim % pad_multiple == 0 else dim + pad_multiple - (dim % pad_multiple)
    )

    has_pad = padded > dim

    if has_pad:
        padded_shape = [padded, *qx.shape[1:]]
        padded_qx = torch.zeros(padded_shape, device=qx.device, dtype=qx.dtype)
        padded_qx[0 : qx.shape[0], ...].copy_(qx)

        padded_x_scale_shape = [*x_scale.shape[1:], padded]
        padded_x_scale = torch.ones(
            padded_x_scale_shape, device=x_scale.device, dtype=x_scale.dtype
        ).permute(-1, -2)
        padded_x_scale[0 : x_scale.shape[0], ...].copy_(x_scale)

        output = cutlass_scaled_mm(
            padded_qx, weight, padded_x_scale, weight_scale, block_size, output_dtype
        )
        return output[0 : qx.shape[0], ...]
    else:
        return cutlass_scaled_mm(
            qx, weight, x_scale, weight_scale, block_size, output_dtype
        )


def _padded_cutlass_fake(
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
    "padded_cutlass",
    _padded_cutlass,
    fake_impl=_padded_cutlass_fake,
)
