# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
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
    # Match FlashInfer: keep BF16→FP8 quant inside the custom op so
    # torch.compile/Inductor cannot rewrite TMA-aligned scale strides.
    apply_input_quant: ClassVar[bool] = False

    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        self.use_deep_gemm_e8m0 = is_deep_gemm_e8m0_used()
        self.is_deep_gemm_supported = is_deep_gemm_supported()
        # Input quant happens inside deep_gemm_fp8_blockscale_mm (not here).

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

        if self.is_deep_gemm_supported:
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
                quant_block_shape=tuple(layer.weight_block_size),
                use_e8m0=self.use_deep_gemm_e8m0,
                is_bmm=getattr(layer, "is_bmm", False),
                bmm_batch_size=getattr(layer, "bmm_batch_size", 0),
            )
            replace_parameter(layer, params.WEIGHT, dg_weight)
            replace_parameter(layer, scale_attr, dg_weight_scale)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # A is BF16 (apply_input_quant=False). As is unused placeholder.
        del As
        group_size = self.weight_group_shape.col
        return torch.ops.vllm.deep_gemm_fp8_blockscale_mm(
            A,
            B,
            Bs,
            group_size,
            self.use_deep_gemm_e8m0,
            bool(envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES),
        )


def _deep_gemm_fp8_blockscale_mm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """BF16→FP8 quant + DeepGEMM, all inside one custom op (FI-style).

    Doing quant in the compiled FX graph lets Inductor rewrite TMA-aligned
    scale strides (e.g. (1, align(M,4)) → (1, M)), which makes DeepGEMM's
    SFA TMA read OOB (CUDA_ERROR_ILLEGAL_ADDRESS). FlashInfer swapAB avoids
    this by owning quant+gemm inside its opaque kernel/op.
    """
    q_input, input_scale = per_token_group_quant_fp8(
        input,
        group_size=group_size,
        column_major_scales=True,
        tma_aligned_scales=tma_aligned_scales,
        use_ue8m0=use_deep_gemm_e8m0,
    )
    output = torch.empty(
        (q_input.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=q_input.device,
    )
    fp8_gemm_nt(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
        is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
    )
    return output


def _deep_gemm_fp8_blockscale_mm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
    tma_aligned_scales: bool,
) -> torch.Tensor:
    return torch.empty(
        input.shape[0],
        weight.shape[0],
        dtype=torch.bfloat16,
        device=input.device,
    )


direct_register_custom_op(
    "deep_gemm_fp8_blockscale_mm",
    _deep_gemm_fp8_blockscale_mm_impl,
    fake_impl=_deep_gemm_fp8_blockscale_mm_fake,
    # Opaque to Dynamo: quant+TMA layout stay inside the op.
    # cudagraph_unsafe: DeepGEMM CUtensorMap uses absolute addresses.
    tags=(torch.Tag.cudagraph_unsafe,),
)


# Backward-compatible alias used by older compiled artifacts / call sites.
def _fp8_gemm_nt_op(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    from vllm.utils.deep_gemm import get_col_major_tma_aligned_tensor

    if input_scale.dim() >= 2:
        input_scale = get_col_major_tma_aligned_tensor(input_scale)
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
    tags=(torch.Tag.needs_fixed_stride_order, torch.Tag.cudagraph_unsafe),
)
