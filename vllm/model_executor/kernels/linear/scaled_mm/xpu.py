# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from .ScaledMMLinearKernel import FP8ScaledMMLinearKernel, FP8ScaledMMLinearLayerConfig


class XPUW8A8FP8LinearKernel(FP8ScaledMMLinearKernel):
    _SUPPORTED_ACT_QUANT_KEYS = {
        kFp8DynamicTensorSym,
        kFp8DynamicTokenSym,
        kFp8StaticTensorSym,
        kFp8StaticTokenSym,
    }
    _SUPPORTED_WEIGHT_QUANT_KEYS = {
        kFp8StaticChannelSym,
        kFp8StaticTensorSym,
    }

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUW8A8FP8Linear only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in cls._SUPPORTED_WEIGHT_QUANT_KEYS:
            return (
                False,
                "XPUW8A8FP8Linear only support per-channel and per-tensor quantization",
            )
        if c.activation_quant_key not in cls._SUPPORTED_ACT_QUANT_KEYS:
            return (
                False,
                "XPUW8A8FP8Linear only support per-tensor and per-token activation "
                "quantization",
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUW8A8FP8Linear only support FP8 weight dtype"
        if c.activation_quant_key.dtype not in {
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        }:
            return False, "XPUW8A8FP8Linear only support FP8 activation dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Ensure weight is stored as C-contiguous [K, N] (KN layout).

        Checkpoints store weight as [N, K]; fp8_gemm requires [K, N],
        C-contiguous.  Three incoming layouts are possible:
          • [N, K] C-contiguous   ← direct checkpoint   → .t().contiguous()
          • [K, N] Fortran-order  ← fp8.py's weight.t() → .contiguous()
          • [K, N] C-contiguous   ← already correct     → no-op

        For square weights (K == N) the shape is ambiguous; contiguity is used
        as a proxy: C-contiguous ≡ checkpoint [N, K] (needs transpose);
        Fortran-order ≡ fp8.py already transposed (needs only contiguous).
        """
        K = getattr(layer, "input_size_per_partition", self.config.weight_shape[1])
        N = getattr(layer, "output_size_per_partition", self.config.weight_shape[0])
        w = layer.weight

        if w.shape not in {(K, N), (N, K)}:
            raise ValueError(
                f"XPUFP8ScaledMM expects weight shape ({K},{N}) or ({N},{K}), "
                f"but got {tuple(w.shape)}"
            )

        needs_transpose = w.shape == (N, K) if K != N else w.is_contiguous()
        layer_weight = w.t() if needs_transpose else w
        replace_parameter(layer, "weight", layer_weight.contiguous())
        ws = layer.weight_scale
        if ws.numel() == 1:
            replace_parameter(layer, "weight_scale", ws.reshape(1))

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
        # B is C-contiguous [K, N] from process_weights_after_loading.
        # fp8_gemm routes on scale dtype (float32) and numel:
        #   As [1]   → per-tensor  (numel==1 branch)
        #   As [M,1] → per-token   (group={1,K} branch, broadcast across K)
        #   Bs [1]   → per-tensor
        #   Bs [N]   → per-channel (mask=bit1 branch)
        # No shape manipulation needed here.
        output = torch.ops._xpu_C.fp8_gemm(A, B, out_dtype, As, Bs, bias)
        return output.view(*output_shape)


class XPUW8A16FP8LinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUW8A16FP8Linear only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in {kFp8StaticChannelSym, kFp8StaticTensorSym}:
            return (
                False,
                "XPUW8A16FP8Linear only support per-channel and per-tensor "
                "quantization",
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUW8A16FP8Linear only support FP8 weight dtype"
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


class XPUFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFp8BlockScaledMM only support on XPU"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        super().process_weights_after_loading(layer)
        scale_attr = (
            "weight_scale_inv" if hasattr(layer, "weight_scale_inv") else "weight_scale"
        )
        scale = getattr(layer, scale_attr)
        # Models with scale_fmt=ue8m0 (e.g. DeepSeek-V4) store weight scales
        # as float8_e8m0fnu. The oneDNN fp8_gemm kernel dispatches to its
        # "block quant" path only when NEITHER scale is e8m0:
        #
        #   is_block_quant = (m1_sc != e8m0) && (m2_sc != e8m0) && ...
        #
        # Since activation scales are always float32 (use_ue8m0=False on XPU,
        # DeepGEMM requires Hopper/Blackwell), an e8m0 weight scale causes
        # is_block_quant=false and falls into the wrong per-channel path,
        # producing NaN. Converting e8m0→float32 here at load time (one-time,
        # negligible overhead for small scale tensors) ensures the kernel sees
        # matching dtypes and correctly enters the block-quant path with the
        # actual group_size derived from scale tensor shapes.
        if scale.dtype == torch.float8_e8m0fnu:
            scale = scale.to(torch.float32)
        replace_parameter(layer, scale_attr, scale.data.t().contiguous())

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # Weight is [N, K]. Use .t() to create a [K, N] view without copying.
        return torch.ops._xpu_C.fp8_gemm(
            A,
            B.t(),
            self.config.out_dtype,
            As,
            Bs,
            torch.Tensor(),
        )
