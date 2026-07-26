# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Module

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _quantize_moe_weight_to_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize stacked MoE expert weights ``(E, N, K)`` to NVFP4.

    One FP32 global scale per expert plus per-block (group-16) FP8 scales,
    matching the ModelOpt NVFP4 checkpoint layout. Returns packed FP4 weights
    ``(E, N, K // 2)``, block scales ``(E, N, K // 16)``, and the per-expert
    global scale ``(E,)`` stored as ``amax / (fp4_max * fp8_max)``.
    """
    assert weight.dim() == 3, f"expected 3D expert weights, got {weight.shape}"
    num_experts, n, k = weight.shape
    assert k % 16 == 0, f"last dim must be a multiple of 16, got {k}"

    amax = weight.abs().amax(dim=(1, 2)).to(torch.float32).clamp_min(1e-8)
    global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax
    weight_scale_2 = (1.0 / global_scale).to(torch.float32)

    # scaled_fp4_quant(w, g) == scaled_fp4_quant(w * g, 1), so fold each
    # expert's scale in and quantize all experts in one call (fp32 to keep the
    # large scale precise), rather than looping per expert.
    scaled = (weight.float() * global_scale[:, None, None]).to(weight.dtype)
    scaled = scaled.reshape(-1, k)
    one = torch.ones((), device=weight.device, dtype=torch.float32)
    qweight, block_scale = scaled_fp4_quant(scaled, one, is_sf_swizzled_layout=False)
    return (
        qweight.reshape(num_experts, n, k // 2),
        block_scale.reshape(num_experts, n, k // 16),
        weight_scale_2,
    )


class Nvfp4OnlineMoEMethod(OnlineMoEMethodBase):
    """Online NVFP4 MoE quantization with per-token activation scales.

    Quantizes fp16/bf16 expert weights to NVFP4 at load time; the FlashInfer
    TRTLLM kernel computes per-token activation scales at runtime. Blackwell
    (SM100) only.
    """

    def __init__(
        self,
        *,
        layer: torch.nn.Module,
    ):
        if not current_platform.is_device_capability_family(100):
            raise ValueError(
                "nvfp4_per_token online quantization requires a Blackwell (SM100) GPU."
            )
        super().__init__(layer.moe_config)
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._quantize_weights(layer)
        self._setup_kernel(layer)

        layer._already_called_process_weights_after_loading = True

    def _quantize_weights(self, layer: Module) -> None:
        w13, w13_scale, w13_scale_2 = _quantize_moe_weight_to_nvfp4(layer.w13_weight)
        w2, w2_scale, w2_scale_2 = _quantize_moe_weight_to_nvfp4(layer.w2_weight)

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)

        # Neutral (1.0) activation global scales: the kernel derives per-token
        # scales at runtime, so the output scalars reduce to the weight scales.
        ones = torch.ones(layer.num_experts, device=w13.device, dtype=torch.float32)
        replace_parameter(layer, "w13_input_scale", ones)
        replace_parameter(layer, "w2_input_scale", ones.clone())

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=layer.w2_weight_scale_2,
            a2_scale=layer.w2_input_scale,
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w13_input_scale", a13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
        replace_parameter(layer, "w2_input_scale", a2_scale)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
            layer=layer,
            per_token_activation=True,
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            layer=layer,
        )
