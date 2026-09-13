# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    kFp8Dynamic128Sym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_gemm_nt,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
    should_auto_disable_deep_gemm,
    should_use_deepgemm_for_fp8_linear,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


class DeepGemmFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        self.use_deep_gemm_e8m0 = is_deep_gemm_e8m0_used()
        act_scale_descriptor = config.activation_quant_key.scale
        self.is_deep_gemm_supported = is_deep_gemm_supported()
        self.quant_fp8 = QuantFP8(
            static=False,
            group_shape=act_scale_descriptor.group_shape,
            use_ue8m0=self.use_deep_gemm_e8m0,
            tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES,
            column_major_scales=True,
        )

    def input_quant_key(self) -> QuantKey | None:
        if self.config.activation_quant_key == kFp8Dynamic128Sym:
            return kFp8Dynamic128Sym
        return None

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda():
            return False, "DeepGEMM is only supported on cuda platform"
        if not is_deep_gemm_supported():
            return False, "Currently, only Hopper and Blackwell GPUs are supported."
        return True, None

    @classmethod
    def can_implement(cls, config):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason
        if config.out_dtype != torch.bfloat16:
            return (False, "Supports only output dtype of bfloat16")

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )
        model_config = get_current_vllm_config().model_config

        if model_config is None:
            return False, "Model configuration is required."

        model_type = getattr(model_config.hf_text_config, "model_type", None)
        if should_auto_disable_deep_gemm(model_type):
            return False, f"Should not use deepgemm for model {model_type}"

        if not should_use_deepgemm_for_fp8_linear(
            config.out_dtype, config.weight_shape
        ):
            return False, "The provided metadata is not supported."
        return True, None

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        assert layer.weight_block_size is not None

        is_bmm = getattr(layer, "is_bmm", False)

        if self.is_deep_gemm_supported:
            dg_weight, dg_weight_scale = deepgemm_post_process_fp8_weight_block(
                wq=params.weight,
                ws=params.block_scale,
                quant_block_shape=tuple(layer.weight_block_size),
                use_e8m0=self.use_deep_gemm_e8m0,
                is_bmm=is_bmm,
                bmm_batch_size=getattr(layer, "bmm_batch_size", 0),
            )
            replace_parameter(layer, params.WEIGHT, dg_weight)
            replace_parameter(layer, params.block_scale_attr, dg_weight_scale)
        # bmm layers bypass this kernel and run through fp8_einsum.
        if not is_bmm:
            layer.deep_gemm_warmup_provider = self

    def get_deep_gemm_warmup_weights(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self._get_layer_params(layer)
        return params.weight, params.block_scale

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        output = torch.empty(
            (A.shape[0], B.shape[0]),
            dtype=out_dtype,
            device=A.device,
        )
        torch.ops.vllm.fp8_gemm_nt_op(A, As, B, Bs, output, self.use_deep_gemm_e8m0)
        return output


def _fp8_gemm_nt_op(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    if use_deep_gemm_e8m0 and input_scale.dtype == torch.int32:
        logical_rows = q_input.shape[0]
        aligned_rows = (logical_rows + 3) // 4 * 4
        if input_scale.shape[0] == aligned_rows and aligned_rows != logical_rows:
            input_scale = input_scale[:logical_rows]
    fp8_gemm_nt(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
        is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
    )


direct_register_custom_op(
    "fp8_gemm_nt_op",
    _fp8_gemm_nt_op,
    mutates_args=["output"],
)
