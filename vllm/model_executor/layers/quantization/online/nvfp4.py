# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Module

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    trtllm_nvfp4_hidden_alignment,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
    ref_nvfp4_quant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    amax_for_moe_weight_quant,
    kNvfp4DynamicToken,
    kNvfp4Static,
    weight_amax,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
NVFP4_BLOCK_SIZE = 16


def _quantize_nvfp4(
    weight: torch.Tensor, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4-quantize the rows of `weight` without the SM100+ quant kernel.

    Returns packed E2M1 codes (low nibble first), E4M3 scales per 16 values and
    the fp32 dequantization global scale, in the checkpoint layout.
    """
    global_scale = (
        weight_amax(weight).float() / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
    ).clamp(min=1e-12)
    inv_global_scale = (1.0 / global_scale).reshape(())
    # E2M1 magnitudes 0, 0.5, ..., 6 doubled to integers, mapped to their codes.
    code_of = torch.zeros(13, dtype=torch.uint8, device=weight.device)
    code_of[torch.tensor([0, 1, 2, 3, 4, 6, 8, 12])] = torch.arange(
        8, dtype=torch.uint8
    ).to(weight.device)
    n, k = weight.shape
    packed = torch.empty(n, k // 2, dtype=torch.uint8, device=weight.device)
    scales = torch.empty(
        n, k // NVFP4_BLOCK_SIZE, dtype=torch.float8_e4m3fn, device=weight.device
    )
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        values, block_scales = ref_nvfp4_quant(
            weight[rows], inv_global_scale, NVFP4_BLOCK_SIZE
        )
        codes = code_of[(values.abs() * 2).long()] | ((values < 0).to(torch.uint8) << 3)
        packed[rows] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales[rows] = block_scales.to(torch.float8_e4m3fn)
    return packed, scales, global_scale


class Nvfp4WeightOnlyOnlineLinearMethod(OnlineLinearBase):
    """Online weight-only NVFP4 (W4A16) linear method.

    Quantizes fp16/bf16 weights to NVFP4 during loading, with one global scale
    per weight shard, and runs them on the platform's W4A16 kernel.
    Activations stay in the model dtype.
    """

    activation_quant_key: QuantKey | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % NVFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"NVFP4 requires input_size_per_partition "
                f"({input_size_per_partition}) to be divisible by "
                f"{NVFP4_BLOCK_SIZE}."
            )

        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        self.kernel = init_nvfp4_linear_kernel(use_a16=True)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        weight = self.get_weight_for_quantization(layer)
        packed, scales, global_scale = _quantize_nvfp4(weight)

        replace_parameter(layer, "weight", packed)
        replace_parameter(layer, "weight_scale", scales)
        replace_parameter(layer, "weight_global_scale", global_scale)

        self.kernel.process_weights_after_loading(layer)
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


def _quantize_moe_weight_to_nvfp4(
    weight: torch.Tensor,
    moe_tp_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize stacked MoE expert weights `(E, N, K)` to NVFP4.

    One FP32 global scale per expert plus per-block (group-16) FP8 scales,
    matching the ModelOpt NVFP4 checkpoint layout. Returns packed FP4 weights
    `(E, N, K // 2)`, block scales `(E, N, K // 16)`, and the per-expert
    global scale `(E,)` stored as `amax / (fp4_max * fp8_max)`.
    """
    assert weight.dim() == 3, f"expected 3D expert weights, got {weight.shape}"
    k = weight.shape[-1]
    assert k % 16 == 0, f"last dim must be a multiple of 16, got {k}"

    amax = weight_amax(weight.flatten(1), dim=-1).to(torch.float32)
    amax = amax_for_moe_weight_quant(amax, moe_tp_size).clamp_min(1e-8)
    global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax
    weight_scale_2 = (1.0 / global_scale).to(torch.float32)

    # Keep the original BF16/FP16 values as the quantizer input. Folding each
    # expert's FP32 global scale into the weight would add a BF16/FP16 rounding
    # before the group-16 scale and E2M1 values are selected.
    weight = weight.contiguous()
    quantized_experts = [
        scaled_fp4_quant(
            expert_weight,
            expert_scale,
            is_sf_swizzled_layout=False,
        )
        for expert_weight, expert_scale in zip(
            weight,
            global_scale,
            strict=True,
        )
    ]
    qweight = torch.stack([quantized for quantized, _ in quantized_experts])
    block_scale = torch.stack([block_scale for _, block_scale in quantized_experts])
    return (
        qweight,
        block_scale,
        weight_scale_2,
    )


class Nvfp4OnlineMoEMethod(OnlineMoEMethodBase):
    """Online NVFP4 MoE quantization with per-token activation scales.

    Quantizes fp16/bf16 expert weights to NVFP4 at load time; the FlashInfer
    TRTLLM kernel computes per-token activation scales at runtime. Blackwell
    (SM100) only.
    """

    activation_quant_key = kNvfp4DynamicToken

    def __init__(
        self,
        *,
        moe: FusedMoEConfig,
    ):
        if not current_platform.is_device_capability_family(100):
            raise ValueError(
                "nvfp4_per_token online quantization requires a Blackwell (SM100) GPU."
            )
        super().__init__(moe)
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=self.activation_quant_key,
        )

    @property
    def per_token_activation(self) -> bool:
        return self.activation_quant_key == kNvfp4DynamicToken

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        w13_weight, w2_weight = self.get_weights_for_quantization(layer)
        self._quantize_weights(layer, w13_weight, w2_weight)
        self._setup_kernel(layer)
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def _quantize_weights(
        self,
        layer: Module,
        w13_weight: torch.Tensor | None = None,
        w2_weight: torch.Tensor | None = None,
    ) -> None:
        w13_weight = layer.w13_weight if w13_weight is None else w13_weight
        w2_weight = layer.w2_weight if w2_weight is None else w2_weight
        moe_tp_size = self.moe.tp_size
        w13, w13_scale, w13_scale_2 = _quantize_moe_weight_to_nvfp4(
            w13_weight, moe_tp_size
        )
        w2, w2_scale, w2_scale_2 = _quantize_moe_weight_to_nvfp4(w2_weight, moe_tp_size)

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
            trtllm_hidden_alignment=trtllm_nvfp4_hidden_alignment(
                per_token_activation=self.per_token_activation,
                is_act_and_mul=self.moe.is_act_and_mul,
            ),
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w13_input_scale", a13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
        replace_parameter(layer, "w2_input_scale", a2_scale)

        if self.moe_kernel is None:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.experts_cls is not None
            self.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                backend=self.nvfp4_backend,
                routing_tables=layer._expert_routing_tables(),
                per_token_activation=self.per_token_activation,
            )
            self.moe_kernel.fused_experts.keep_constants(layer)
        else:
            # Reload creates new scale tensors; derived kernel scales must use
            # their new values before layerwise reload restores captured storage.
            assert self.moe_quant_config is not None
            assert self.moe_quant_config.g1_alphas is not None
            assert self.moe_quant_config.g2_alphas is not None
            self.moe_quant_config.g1_alphas.copy_(w13_scale_2)
            self.moe_quant_config.g2_alphas.copy_(w2_scale_2)

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
