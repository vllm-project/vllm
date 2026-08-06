# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Module, Parameter

from vllm._custom_ops import scaled_fp4_quant
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MarlinNvFp4LinearKernel,
    NvFp4LinearLayerConfig,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearKernel
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    _Fp8OnlineLinearBase,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
    ref_nvfp4_quant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# NVFP4 global scale maps a tensor's amax into the representable range of the
# FP8-E4M3 block scale times the FP4 element max (448 * 6 = 2688). Kernels store
# the global scale in *divisor* form (amax / 2688); ``scaled_fp4_quant`` takes
# its reciprocal (the *multiplier*).
_NVFP4_GLOBAL_SCALE_DENOM = FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX
_NVFP4_GROUP_SIZE = 16

# E2M1 magnitudes, indexed by the 3-bit FP4 magnitude field. The sign bit is
# 0x08. Used to pack ref_nvfp4_quant's dequantized-float output back into the
# 4-bit codes when the native packing kernel is unavailable for this arch.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4_global_scale(amax: torch.Tensor) -> torch.Tensor:
    """Per-tensor global scale in divisor form (amax / 2688)."""
    return (amax.to(torch.float32) / _NVFP4_GLOBAL_SCALE_DENOM).clamp_min(
        torch.finfo(torch.float32).tiny
    )


def _pack_nvfp4_reference(
    weight: torch.Tensor, global_scale_mult: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arch-independent NVFP4 packer built on ``ref_nvfp4_quant``.

    ``ref_nvfp4_quant`` (Triton on CUDA) is the ground-truth quantizer: it
    returns the per-element FP4 value as a float plus the positive FP8-E4M3
    per-group block scale. We map the float FP4 values back to their 4-bit
    codes (sign bit 0x08 + 3-bit E2M1 magnitude index) and pack two per byte,
    low nibble first, matching the layout the Marlin FP4 kernel and
    ``dequantize_to_dtype(swizzle=False)`` expect.
    """
    n, k = weight.shape
    fp4_f, block_scale = ref_nvfp4_quant(
        weight.float(), global_scale_mult, _NVFP4_GROUP_SIZE
    )
    absv = fp4_f.abs()
    idx = torch.zeros_like(absv, dtype=torch.long)
    for i, mag in enumerate(_E2M1_MAGNITUDES):
        idx = torch.where(absv == mag, torch.full_like(idx, i), idx)
    sign = (fp4_f < 0).to(torch.long) * 0x08
    nibbles = (sign | idx).to(torch.uint8).view(n, k // 2, 2)
    packed_weight = (nibbles[..., 0] | (nibbles[..., 1] << 4)).to(torch.uint8)
    return packed_weight, block_scale.to(torch.float8_e4m3fn)


def _pack_nvfp4(
    weight: torch.Tensor, global_scale_mult: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a BF16/FP16 weight to NVFP4 (uint8 + non-swizzled FP8 block scale).

    Prefers the native ``scaled_fp4_quant`` CUDA op. Some builds compile that
    quantization kernel only for Blackwell (it raises "No compiled nvfp4
    quantization kernel for SM XX"), even though the FP4 *packing* is pure
    arithmetic and the Marlin GEMM runs on SM>=75. In that case fall back to a
    reference packer that runs on any CUDA arch.
    """
    try:
        return scaled_fp4_quant(
            weight.contiguous(),
            global_scale_mult,
            is_sf_swizzled_layout=False,
        )
    except RuntimeError as e:
        if "quantization kernel" not in str(e):
            raise
        return _pack_nvfp4_reference(weight, global_scale_mult)


def _quantize_weight_to_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a dense BF16/FP16 weight ``[N, K]`` to NVFP4 storage format.

    Returns ``(packed_weight, block_scale, weight_global_scale)`` where
    ``packed_weight`` is uint8 ``[N, K/2]`` (two FP4 values per byte),
    ``block_scale`` is FP8-E4M3 ``[N, K/16]`` in the non-swizzled layout the
    NVFP4 kernels' ``process_weights_after_loading`` expects, and
    ``weight_global_scale`` is the fp32 per-tensor divisor scale.
    """
    weight_global_scale = _nvfp4_global_scale(weight.abs().max())
    # scaled_fp4_quant takes the multiplier (reciprocal of the divisor scale).
    global_scale_mult = (1.0 / weight_global_scale).to(torch.float32)
    packed_weight, block_scale = _pack_nvfp4(weight, global_scale_mult)
    return packed_weight, block_scale, weight_global_scale


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


class _Nvfp4OnlineLinearBase(_Fp8OnlineLinearBase):
    """Shared setup for online NVFP4 dense linear methods.

    Reuses the online meta-device weight loader from ``_Fp8OnlineLinearBase``
    (bf16/fp16 weights are materialized just-in-time, then quantized in
    ``process_weights_after_loading``). Subclasses pick the kernel (Marlin for
    W4A16, CUTLASS for W4A4).
    """

    kernel: NvFp4LinearKernel

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
        if input_size_per_partition % _NVFP4_GROUP_SIZE != 0:
            raise ValueError(
                f"NVFP4 requires input_size_per_partition "
                f"({input_size_per_partition}) to be divisible by "
                f"{_NVFP4_GROUP_SIZE}."
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
        layer.params_dtype = params_dtype

    def _quantize_and_store_weight(self, layer: Module) -> None:
        packed_weight, block_scale, weight_global_scale = _quantize_weight_to_nvfp4(
            layer.weight
        )
        replace_parameter(layer, "weight", packed_weight)
        replace_parameter(layer, "weight_scale", block_scale)
        layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class Nvfp4W4A16OnlineLinearMethod(_Nvfp4OnlineLinearBase):
    """Online weight-only NVFP4 (W4A16) dense linear.

    Weights are quantized to FP4 at load time; activations stay bf16/fp16 and
    the Marlin FP4 kernel dequantizes on the fly. Runs on any SM>=75 GPU (the
    non-Blackwell path).
    """

    def __init__(self):
        super().__init__()
        # Pin Marlin directly: init_nvfp4_linear_kernel() can pick a W4A4 kernel
        # (e.g. under VLLM_BATCH_INVARIANT / --linear-backend) that expects a
        # runtime activation scale this weight-only path never sets. Matches
        # ModelOptNvFp4W4A16LinearMethod.
        self.kernel = MarlinNvFp4LinearKernel(NvFp4LinearLayerConfig())

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._quantize_and_store_weight(layer)
        # Marlin repacks the weight/scales into its own layout.
        self.kernel.process_weights_after_loading(layer)

        layer._already_called_process_weights_after_loading = True


class Nvfp4W4A4OnlineLinearMethod(_Nvfp4OnlineLinearBase):
    """Online NVFP4 W4A4 dense linear with dynamic activation quant.

    Weights are quantized to FP4 once at load time; activations are quantized to
    FP4 per forward using a dynamically computed per-tensor global scale (no
    calibration data). The CUTLASS FP4 GEMM runs only on Blackwell (SM>=100)
    tensor cores.
    """

    def __init__(self):
        super().__init__()
        self.kernel = init_nvfp4_linear_kernel(use_a16=False)
        # Fail closed: without a native FP4-activation kernel the selector falls
        # back to Marlin (weight-only), which would silently serve W4A16 instead
        # of the requested W4A4.
        if isinstance(self.kernel, MarlinNvFp4LinearKernel):
            raise ValueError(
                "nvfp4_w4a4 (W4A4 with dynamic per-token FP4 activation "
                "quantization) requires a native NVFP4 W4A4 GEMM kernel "
                "(CUTLASS/FlashInfer on Blackwell SM>=100); none is available on "
                "this platform. Use '--quantization nvfp4' for the weight-only "
                "(W4A16) path, which runs on any SM>=75 via Marlin."
            )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._quantize_and_store_weight(layer)

        # Seed the activation global scale with a neutral value; it is refreshed
        # per-forward in apply() since there is no calibration data.
        wgs = layer.weight_global_scale
        layer.input_global_scale = Parameter(wgs.clone(), requires_grad=False)
        layer.input_global_scale_inv = Parameter(
            (1.0 / wgs).to(torch.float32), requires_grad=False
        )
        layer.alpha = Parameter(layer.input_global_scale * wgs, requires_grad=False)

        # CUTLASS swizzles the block scale and pads the weight.
        self.kernel.process_weights_after_loading(layer)

        layer._already_called_process_weights_after_loading = True

    def _activation_scale_args(
        self, layer: torch.nn.Module, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamic per-row (per-token) activation scales as (inv, alpha).

        Each token scales from its own amax, not a shared per-tensor amax:
        under CUDA graphs the padded capture buffer's padding rows would inflate
        a per-tensor amax and corrupt real rows' scales. Computed with plain
        (non in-place) ops so the x -> amax -> scale dataflow is captured and
        replays against the live activation.
        """
        x2d = x.reshape(-1, x.shape[-1])
        amax_row = x2d.abs().amax(dim=-1)
        input_global_scale = _nvfp4_global_scale(amax_row)
        input_global_scale_inv = (1.0 / input_global_scale).to(torch.float32)
        alpha = (input_global_scale * layer.weight_global_scale).to(torch.float32)
        return input_global_scale_inv, alpha

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_global_scale_inv, alpha = self._activation_scale_args(layer, x)
        return self.kernel.apply_weights(
            layer,
            x,
            bias,
            input_global_scale_inv=input_global_scale_inv,
            alpha=alpha,
        )
