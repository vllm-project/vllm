# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online LUT-B quantization (MXFP8 activations) of linear layers and routed
MoE experts."""

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import _get_linear_backend
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lut_b_moe import (
    LutBEmulationTritonExperts,
    LutBSm107Experts,
    make_lut_b_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    LUT_B_BLOCK_K,
    LUT_B_BLOCK_N,
    LUT_B_SM107_TILE_K,
    LUT_B_SM107_TILE_N,
    dequantize_lut_b,
    lut_b_sm107_supported,
    lut_b_to_sm107_tiles,
    mxfp8_qdq,
    quantize_lut_b,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kLutBStatic,
    kMxfp8Dynamic,
)
from vllm.model_executor.utils import is_weights_pre_processed, replace_parameter
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

# The SM107 kernel computes 128-row output tiles.
_LUT_B_SM107_BLOCK_M = 128


def apply_lut_b_linear_sm107(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ W.T (+ bias)`` on Rubin with MXFP8 activations and LUT-B weights.

    ``weight`` is one expert in the SM107 tile layout, ``[1, N / 128, K / 128,
    tile bytes]``, so this runs the MoE grouped GEMM with every row routed to
    expert 0.
    """
    k = weight.size(2) * LUT_B_SM107_TILE_K
    n = weight.size(1) * LUT_B_SM107_TILE_N
    x_2d = x.reshape(-1, k).contiguous()
    num_tokens = x_2d.size(0)
    out = x_2d.new_empty(num_tokens, n)
    padded = round_up(num_tokens, _LUT_B_SM107_BLOCK_M)
    # Rows past num_tokens are padding, which the kernel skips.
    sorted_token_ids = torch.arange(padded, dtype=torch.int32, device=x.device)
    expert_ids = torch.zeros(
        padded // _LUT_B_SM107_BLOCK_M, dtype=torch.int32, device=x.device
    )
    num_tokens_post_padded = torch.full(
        (1,), padded, dtype=torch.int32, device=x.device
    )
    a, a_scales = mxfp8_e4m3_quantize(x_2d)
    ops.lut_b_moe_gemm(
        a,
        a_scales.view(num_tokens, -1),
        weight,
        out,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        1,
    )
    if bias is not None:
        out.add_(bias)
    return out.view(*x.shape[:-1], n)


def apply_lut_b_linear_emulation(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Emulated LUT-B linear: dequantized ``[N, K]`` weight, MXFP8 QDQ input."""
    return F.linear(mxfp8_qdq(x), weight, bias)


class LutBOnlineLinearMethod(OnlineLinearBase):
    """Quantize BF16/FP16 linear weights to LUT-B while loading.

    Weights use 3-bit indices into one E4M3 codebook (with a UE8M0 scale) per
    8x64 tile and activations are MXFP8. Rubin runs the native LUT-B tensor
    core GEMM when both partition dims are multiples of 128; other GPUs and
    shapes emulate the same numerics with dequantized weights.
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
        output_size_per_partition = sum(output_partition_sizes)
        if (
            input_size_per_partition % LUT_B_BLOCK_K
            or output_size_per_partition % LUT_B_BLOCK_N
        ):
            raise ValueError(
                "LUT-B requires the input size per partition to be divisible by "
                f"{LUT_B_BLOCK_K} and the output size per partition by "
                f"{LUT_B_BLOCK_N}, got ({output_size_per_partition}, "
                f"{input_size_per_partition}). Exclude this layer with "
                "quantization_config.ignore."
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

        backend = _get_linear_backend(quantization="lut_b")
        if backend not in ("auto", "emulation"):
            raise ValueError(
                f"linear_backend={backend!r} is not supported for LUT-B; "
                "expected 'auto' or 'emulation'."
            )
        # The native GEMM writes BF16/FP16 in 128 x 128 output tiles.
        native_shape = (
            input_size_per_partition % LUT_B_SM107_TILE_K == 0
            and output_size_per_partition % LUT_B_SM107_TILE_N == 0
            and self.input_dtype in (torch.bfloat16, torch.float16)
        )
        self.use_native = backend == "auto" and native_shape and lut_b_sm107_supported()
        if self.use_native:
            return
        if backend == "auto" and lut_b_sm107_supported():
            logger.info_once(
                "LUT-B linear falls back to emulation for layers whose sizes "
                "are not multiples of 128, or a dtype other than BF16/FP16."
            )
        else:
            logger.warning_once(
                "Using emulated LUT-B linear: weights are stored dequantized and "
                "activations are MXFP8 quantize-dequantized. This validates "
                "accuracy, not memory use or speed."
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if is_weights_pre_processed():
            raise RuntimeError("Online LUT-B cannot use pre-processed weights")
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        weight = self.get_weight_for_quantization(layer)
        packed, codebooks, scales = quantize_lut_b(weight)
        if self.use_native:
            weight = lut_b_to_sm107_tiles(packed[None], codebooks[None], scales[None])
        else:
            weight = dequantize_lut_b(packed, codebooks, scales, self.input_dtype)
        replace_parameter(layer, "weight", weight)
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_native:
            return apply_lut_b_linear_sm107(x, layer.weight, bias)
        return apply_lut_b_linear_emulation(x, layer.weight, bias)


class LutBOnlineMoEMethod(OnlineMoEMethodBase):
    """Quantize BF16/FP16 routed experts to LUT-B while loading.

    Weights use 3-bit indices into one E4M3 codebook (with a UE8M0 scale) per
    8x64 tile and activations are MXFP8. Rubin runs the native LUT-B tensor
    core kernel; other GPUs emulate the same numerics.
    """

    def __init__(self, *, moe: FusedMoEConfig):
        super().__init__(moe)
        if moe.has_bias:
            raise ValueError("LUT-B online MoE quantization does not support bias.")

        if moe.moe_backend not in ("auto", "emulation"):
            raise ValueError(
                f"moe_backend={moe.moe_backend!r} is not supported for LUT-B "
                "MoE; expected 'auto' or 'emulation'."
            )
        native_shape = (
            moe.hidden_dim % LUT_B_SM107_TILE_K == 0
            and moe.intermediate_size_per_partition % LUT_B_SM107_TILE_N == 0
        )
        supported, reason = LutBSm107Experts.is_supported_config(
            LutBSm107Experts,
            moe,
            kLutBStatic,
            kMxfp8Dynamic,
            FusedMoEActivationFormat.Standard,
        )
        self.experts_cls: type[LutBSm107Experts | LutBEmulationTritonExperts]
        if moe.moe_backend == "emulation":
            self.experts_cls = LutBEmulationTritonExperts
        elif supported and native_shape:
            self.experts_cls = LutBSm107Experts
        else:
            self.experts_cls = LutBEmulationTritonExperts
            if reason is None:
                reason = "hidden and intermediate sizes must be multiples of 128"
            logger.info_once("LUT-B MoE falls back to emulation: %s", reason)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        # Both experts quantize activations themselves, and the native weights
        # carry their codebooks and scales inline.
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        w13, w2 = self.get_weights_for_quantization(layer)
        self._zero_padding(layer, w13, w2)
        for name, weight in (("w13_weight", w13), ("w2_weight", w2)):
            packed, codebooks, scales = quantize_lut_b(weight)
            if self.experts_cls is LutBSm107Experts:
                weight = lut_b_to_sm107_tiles(packed, codebooks, scales)
            else:
                weight = dequantize_lut_b(packed, codebooks, scales, weight.dtype)
            replace_parameter(layer, name, weight)
        self.release_requantization_source_weights(layer)

        quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_quant_config = quant_config
        self.moe_kernel = make_lut_b_moe_kernel(
            self.experts_cls,
            self.moe,
            quant_config,
            layer._expert_routing_tables(),
        )
        layer._already_called_process_weights_after_loading = True
