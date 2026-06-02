# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear import (  # noqa: E501
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform



class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in {kFp8StaticChannelSym, kFp8StaticTensorSym}:
            return (
                False,
                "XPUFP8ScaledMM only support per-channel and per-tensor quantization",
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # fp8_gemm_w8a16 expects weight in [in, out] layout.
        # Transpose if weight is still in [out, in] layout.
        # For square matrices, use contiguity as tie-breaker:
        # checkpoint weights are contiguous, .t() views are not.
        weight = layer.weight
        out_features, in_features = self.config.weight_shape

        if weight.shape == (out_features, in_features) and (
            in_features != out_features or weight.is_contiguous()
        ):
            replace_parameter(layer, "weight", weight.data.t())
        # else: already in [in, out] layout — no-op

        weight_scale = layer.weight_scale.t().contiguous()
        replace_parameter(layer, "weight_scale", weight_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

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
        pass



class XPUFp8BlockDequantLinearKernel(FP8ScaledMMLinearKernel):
    """Block-scaled W8A16-FP8 linear kernel for XPU (dequant fallback).

    The native XPU FP8 op (``fp8_gemm_w8a16``) only supports per-tensor and
    per-channel weight scales, so block-scaled FP8 checkpoints (e.g.
    ``RedHatAI/gemma-4-31B-it-FP8-block``, which stores a scale per
    ``[block_n, block_k]`` tile) cannot use it. This kernel instead
    dequantizes the FP8 weight to the model compute dtype once after loading
    -- expanding each block scale across its tile -- and runs inference as a
    standard ``F.linear`` with full-precision activations.

    This makes block-scaled compressed-tensors W8A16-FP8 checkpoints load and
    run on XPU. Note the weight is materialized in the compute dtype (e.g.
    bf16), so it uses ~2x the memory of the FP8 weight and gains no FP8-GEMM
    throughput benefit; it is a correctness-first fallback.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFp8BlockDequant only supported on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key.dtype not in {
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        }:
            return False, "XPUFp8BlockDequant only supports FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        # Bypass FP8ScaledMMLinearKernel.__init__, which builds a QuantFP8
        # activation quantizer we do not need (activations stay in full
        # precision for this dequant fallback).
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight  # fp8, shape [N, K] = [out, in]
        n_dim, k_dim = weight.shape

        # For block quant, the CT scheme renames the scale to
        # "weight_scale_inv"; per-tensor/per-channel keep "weight_scale".
        if hasattr(layer, "weight_scale_inv"):
            scale_name = "weight_scale_inv"
        else:
            scale_name = "weight_scale"
        weight_scale = getattr(layer, scale_name).to(torch.float32)

        # Expand a block scale [ceil(N/bn), ceil(K/bk)] to the full [N, K]
        # weight shape. Per-tensor/per-channel scales already broadcast.
        block_size = getattr(layer, "weight_block_size", None)
        if block_size is not None and weight_scale.shape != (n_dim, k_dim):
            block_n, block_k = block_size[0], block_size[1]
            weight_scale = weight_scale.repeat_interleave(block_n, dim=0)
            weight_scale = weight_scale.repeat_interleave(block_k, dim=1)
            weight_scale = weight_scale[:n_dim, :k_dim]

        compute_dtype = self.config.out_dtype or torch.get_default_dtype()
        dequant_weight = (weight.to(torch.float32) * weight_scale).to(compute_dtype)

        replace_parameter(
            layer,
            "weight",
            torch.nn.Parameter(dequant_weight, requires_grad=False),
        )
        # Scales are folded into the weight; replace with a scalar so any
        # downstream reference stays valid but is a no-op.
        replace_parameter(
            layer,
            scale_name,
            torch.nn.Parameter(
                torch.ones(1, dtype=compute_dtype, device=weight.device),
                requires_grad=False,
            ),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight  # dequantized [N, K] = [out, in]
        return F.linear(x, weight.to(x.dtype), bias)

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
        # Unused: this kernel folds scales into the weight and serves via
        # apply_weights / F.linear, so no scaled-mm primitive is needed.
        raise NotImplementedError(
            "XPUFp8BlockDequantLinearKernel uses a dequant + F.linear fallback"
        )

    def _get_layer_params(
        self, layer: torch.nn.Module
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        # Required by the ABC; unused by the dequant fallback since scales are
        # folded into the weight during process_weights_after_loading.
        w, w_s, x_s, x_s_ub = self.layer_param_names
        return (
            getattr(layer, w),
            getattr(layer, w_s, None),
            getattr(layer, x_s, None),
            getattr(layer, x_s_ub, None),
        )

