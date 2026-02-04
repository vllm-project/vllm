# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_gemm_nt,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
    should_use_deepgemm_for_fp8_linear,
)
from vllm.utils.torch_utils import direct_register_custom_op

from ...utils.fp8_utils import deepgemm_post_process_fp8_weight_block
from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from .cutlass import CutlassFp8BlockScaledMMKernel
from .triton import TritonFp8BlockScaledMMKernel


def _fp8_gemm_nt_op(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    fp8_gemm_nt(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
        is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
    )


def _fp8_gemm_nt_op_fake(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    return None


direct_register_custom_op(
    "fp8_gemm_nt_op",
    _fp8_gemm_nt_op,
    mutates_args=["output"],
    fake_impl=_fp8_gemm_nt_op_fake,
)


class DeepGemmFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        self.use_deep_gemm_e8m0 = is_deep_gemm_e8m0_used()
        act_scale_descriptor = config.activation_quant_key.scale
        self.is_deep_gemm_supported = is_deep_gemm_supported()
        self.input_quant_op = QuantFP8(
            static=False,
            group_shape=act_scale_descriptor.group_shape,
            use_ue8m0=is_deep_gemm_e8m0_used(),
        )

    @classmethod
    def ordered_fallback_kernels(cls) -> list[type["Fp8BlockScaledMMLinearKernel"]]:
        return [CutlassFp8BlockScaledMMKernel, TritonFp8BlockScaledMMKernel]

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda():
            return False, "DeepGEMM is only supported on cuda platform"
        if not is_deep_gemm_supported():
            return False, "Currently, only Hopper and Blackwell GPUs are supported."

        return True, None

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)

        if should_use_deepgemm_for_fp8_linear(self.config.out_dtype, params.weight):
            weight_scale_invs = params.weight_scale_inv
            scale_attr = (
                params.WEIGHT_SCALE_INV
                if weight_scale_invs is not None
                else params.WEIGHT_SCALE
            )
            dg_weight, dg_weight_scale = deepgemm_post_process_fp8_weight_block(
                wq=params.weight,
                ws=weight_scale_invs
                if weight_scale_invs is not None
                else params.weight_scale,
                quant_block_shape=(
                    self.weight_group_shape.row,
                    self.weight_group_shape.col,
                ),
                use_e8m0=is_deep_gemm_e8m0_used(),
            )
            replace_parameter(layer, params.WEIGHT, dg_weight)
            replace_parameter(layer, scale_attr, dg_weight_scale)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        output = torch.empty(
            (A.shape[0], B.shape[0]),
            dtype=out_dtype,
            device=A.device,
        )

        torch.ops.vllm.fp8_gemm_nt_op(A, As, B, Bs, output, self.use_deep_gemm_e8m0)

        return output
