# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
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
        replace_parameter(layer, "weight", layer_weight)
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
    # Opt-in W8A16 routing (XPU_BLOCK_FP8_W8A16=1, added by apply-w8a16.py): fp16 activations go straight into
    # oneDNN fp8_gemm_w8a16 with the checkpoint's 128x128 scales, as llm-scaler 0.26 b1 routed block FP8.
    # Unset, or any other value, keeps the stock path (activation quantization, then W8A8 fp8_gemm).
    apply_input_quant = os.environ.get("XPU_BLOCK_FP8_W8A16", "0") != "1"
    _w8a16_min_m = int(os.environ.get("XPU_BLOCK_FP8_W8A16_MIN_M", "1"))
    _w8a16_seen: set = set()  # log keys already printed by this process (a cheap check before info_once)

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

        # Ragged N (N % block_n != 0): oneDNN needs n_blocks to divide N.
        # Weight untouched; only repeat scale rows to a finer N-group gn that
        # divides both N and block_n (gn = gcd(N, block_n)):
        #   scale [ceil(N/block_n), K/block_k] --> [N/gn, K/block_k]
        # oneDNN only accepts gn that is a multiple of 16, and gcd(N, block_n)
        # is a power of two (block_n=128), so gn must be >= 16. No-op when
        # N % block_n == 0.
        block_n, block_k = self.weight_group_shape
        N, K = layer.weight.shape
        if N % block_n != 0:
            gn = math.gcd(N, block_n)
            assert gn % 16 == 0, (
                f"XPU block-scaled FP8: N ({N}) yields group width {gn}, but "
                f"oneDNN only supports multiples of 16; this weight shape is "
                f"unsupported."
            )
            col_start = torch.arange(N // gn, device=scale.device) * gn
            src_idx = torch.div(col_start, block_n, rounding_mode="floor")
            scale = scale.index_select(0, src_idx).contiguous()

        # Ragged K needs the runtime activation scale expanded too, which we
        # don't handle; DeepSeek/GLM keep K block-aligned, so fail loudly.
        assert K % block_k == 0, (
            f"XPU block-scaled FP8 requires K ({K}) to be a multiple of the "
            f"weight block size ({block_k}); ragged-K weights are unsupported."
        )

        # Checkpoint scale is [n_blocks, k_blocks] (one value per block tile).
        # oneDNN fp8_gemm requires contiguous [k_blocks, n_blocks] layout.
        # We store the transposed contiguous buffer as a .t() view so that:
        #   - MLA's scaled_dequantize still sees [n_blocks, k_blocks] shape
        #   - apply_block_scaled_mm recovers the contiguous buffer via .t()
        scale_kn = scale.data.t().contiguous()  # [k_blocks, n_blocks]
        replace_parameter(layer, scale_attr, scale_kn.t())  # view: [n_blocks, k_blocks]

        if getattr(layer, "is_bmm", False):
            self._prepare_bmm_params(layer, scale_kn)

    def _prepare_bmm_params(
        self, layer: torch.nn.Module, scale_kn: torch.Tensor
    ) -> None:
        """Precompute batched weight and scale for grouped fp8_bmm (e.g. wo_a).

        Splits scale [k_blocks, n_blocks] into [G, k_blocks, n_blocks_per_group]
        and weight [N_total, K] into [G, K, N_per_group] for batch GEMM.
        """
        batch = layer.bmm_batch_size
        k_blocks, n_blocks = scale_kn.shape
        layer.bmm_scale = (
            scale_kn.reshape(k_blocks, batch, n_blocks // batch)
            .permute(1, 0, 2)
            .contiguous()
        )
        w = layer.weight
        N_total, K = w.shape
        layer.bmm_weight = w.reshape(batch, N_total // batch, K).permute(
            0, 2, 1
        )  # [G, K, N_per_group]

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # B is [N, K]; .t() gives [K, N] view (no copy).
        # Bs is stored as [n_blocks, k_blocks] view; .t() recovers the
        # contiguous [k_blocks, n_blocks] buffer that oneDNN expects.
        if not self.apply_input_quant:
            n, k = B.shape
            m = A.shape[0]
            seen = self._w8a16_seen
            if (
                n % 128 == 0
                and k % 128 == 0
                and k > 128
                and tuple(self.weight_group_shape) == (128, 128)
                and A.dtype == torch.float16
            ):
                if m >= self._w8a16_min_m:
                    if "active" not in seen:
                        seen.add("active")
                        logger.info_once(
                            "XPU_BLOCK_FP8_W8A16: W8A16 active (oneDNN fp8_gemm_w8a16, fp16 "
                            "activations, 128x128 weight scales); first use m=%d n=%d k=%d, "
                            "min_m=%d",
                            m, n, k, self._w8a16_min_m, scope="process",
                        )
                    scale_kn = Bs.t()
                    if scale_kn.dtype != torch.float32 or not scale_kn.is_contiguous():
                        scale_kn = scale_kn.to(torch.float32).contiguous()
                    return torch.ops._xpu_C.fp8_gemm_w8a16(A.contiguous(), B.t(), scale_kn, None)
                if "below_min_m" not in seen:
                    seen.add("below_min_m")
                    logger.info_once(
                        "XPU_BLOCK_FP8_W8A16: W8A8 below min_m=%d (first m=%d n=%d k=%d)",
                        self._w8a16_min_m, m, n, k, scope="process",
                    )
            elif ("fallback", n, k) not in seen:
                seen.add(("fallback", n, k))
                logger.info_once(
                    "XPU_BLOCK_FP8_W8A16: fallback to W8A8 for n=%d k=%d (group %s, dtype %s)",
                    n, k, str(tuple(self.weight_group_shape)), str(A.dtype), scope="process",
                )
            A, As = self.quant_fp8(A, None, None, use_triton=self.use_triton)
        return torch.ops._xpu_C.fp8_gemm(
            A,
            B.t(),
            self.config.out_dtype,
            As,
            Bs.t(),
            torch.Tensor(),
        )
