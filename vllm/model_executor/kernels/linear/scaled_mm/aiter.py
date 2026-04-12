# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from .cutlass import CutlassInt8ScaledMMLinearKernel
from .ScaledMMLinearKernel import Int8ScaledMMLinearLayerConfig


class AiterInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "Requires ROCm."

        if compute_capability is not None and compute_capability < 90:
            return False, "requires compute capability 90 and above."

        try:
            import aiter  # noqa: F401 # deliberately attempt to import aiter
        except Exception:
            return False, "requires `aiter` to be installed."

        if not rocm_aiter_ops.is_linear_enabled():
            return (
                False,
                "requires setting `VLLM_ROCM_USE_AITER=1` "
                "and `VLLM_ROCM_USE_AITER_LINEAR=1`. "
                "`VLLM_ROCM_USE_AITER_LINEAR` default is True.",
            )
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "supports symmetric quantization only."
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        `AiterInt8ScaledMMLinearKernel` implements a fused version of
            `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
        where scale_a * a and scale_b * b are implemented using numpy-style
        broadcasting.
        Currently only support per-tensor-per-tensor GEMM
        and per-token-per-channel GEMM through AITER
        w8a8 scaled gemm. `AiterInt8ScaledMMLinearKernel` also does not support
        ATIER block scaled GEMM and mix-precision GEMM.
        """
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        assert symmetric, (
            "AiterInt8ScaledMMLinearKernel only supports symmetric quantization."
        )
        x_q, x_s, x_zp = ops.scaled_int8_quant(x, i_s, i_zp, symmetric=symmetric)

        assert x_zp is None, (
            "AiterInt8ScaledMMLinearKernel only supports symmetric quantization."
        )
        out_dtype = x.dtype

        assert w_q.shape[0] % 16 == 0 and w_q.shape[1] % 16 == 0
        assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
        assert bias is None or bias.shape[0] == w_q.shape[1] and bias.dtype == out_dtype

        m = x_q.shape[0]  # a
        n = w_q.shape[1]  # b

        per_tensor_scale_a = x_s.numel() == 1
        per_tensor_scale_b = w_s.numel() == 1
        per_token_scale_a = x_s.numel() == m
        per_channel_scale_b = w_s.numel() == n

        # @TODO:
        # Maybe broadcast the per-tensor-scale into per-channel-scale
        # if one of the scale is a per-channel-scale.
        # For now, it only supports:
        # - per-tensor-per-tensor a8w8 scaled GEMM, and
        # - per-token-per-channel a8w8 scaled GEMM
        assert (per_tensor_scale_a and per_tensor_scale_b) or (
            per_token_scale_a and per_channel_scale_b
        ), (
            "Currently only support per-tensor-per-tensor GEMM "
            " and per-token-per-channel GEMM through AITER"
            " w8a8 scaled gemm. `AiterInt8ScaledMMLinearKernel` "
            "does not support AITER block scaled GEMM."
        )

        # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects
        # a to be [M, K]
        # b to be [N, K]
        # CutlassInt8ScaledMMLinearKernel prepare weight `w_q` in [K, N] format
        return rocm_aiter_ops.gemm_a8w8(x_q, w_q.t(), x_s, w_s, bias, out_dtype)


class AiterFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        n, k = config.weight_shape

        self.use_triton = (
            not current_platform.is_fp8_fnuz()
            and rocm_aiter_ops.is_triton_gemm_w8a8_tuned(n, k)
        )

    @classmethod
    def is_supported(cls, compute_capability=None):
        return (
            rocm_aiter_ops.is_linear_enabled(),
            "Only supported on ROCm platform \
                with aiter package installed.",
        )

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        if self.use_triton:
            gemm_a8w8_blockscale_op = rocm_aiter_ops.triton_gemm_a8w8_blockscale
        else:
            gemm_a8w8_blockscale_op = rocm_aiter_ops.gemm_a8w8_blockscale

        return gemm_a8w8_blockscale_op(
            A, B, As, Bs, list(self.weight_group_shape), output_dtype=out_dtype
        )


class AiterFp8BlockScaledDynamicMMKernel(Fp8BlockScaledMMLinearKernel):
    """
    Dynamic Triton / CK FP8 block-scaled GEMM for ROCm.

    Dispatches between two kernels based on input batch size:
    - Small batches (M <= threshold): Triton for low-concurrency efficiency.
    - Large batches (M > threshold): CK for high-concurrency throughput.

    Threshold is controlled by VLLM_ROCM_W8A8_TRITON_MAX_M (default 16).

    Uses torch.cond for compile-safe runtime branching, following the same
    pattern as FlashInferFp8DeepGEMMDynamicBlockScaledKernel on CUDA.
    Only activated when Triton has a tuned config for the (N, K) shape.
    """

    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        self.use_triton = True

    @classmethod
    def is_supported(cls, compute_capability=None):
        return (
            rocm_aiter_ops.is_linear_enabled(),
            "Only supported on ROCm platform "
            "with aiter package installed.",
        )

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )

        n, k = config.weight_shape
        if current_platform.is_fp8_fnuz():
            return (
                False,
                "Dynamic dispatch not available for fnuz FP8 format.",
            )
        if not rocm_aiter_ops.is_triton_gemm_w8a8_tuned(n, k):
            return (
                False,
                "No tuned Triton config for this (N, K) shape.",
            )

        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        return torch.ops.vllm.dynamic_aiter_triton_ck_blockscale_gemm(
            A, B, As, Bs, out_dtype
        )


def _dynamic_aiter_triton_ck_blockscale_gemm_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Conditional Triton / CK FP8 blockscale GEMM with batch-size dispatch.

    - M <= VLLM_ROCM_W8A8_TRITON_MAX_M: Triton (better at low concurrency).
    - M > threshold: CK (better throughput at high concurrency).

    Uses torch.cond so torch.compile can capture both branches in the graph.
    """
    threshold = envs.VLLM_ROCM_W8A8_TRITON_MAX_M

    def run_triton(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale(
            A, B, As, Bs, out_dtype
        )

    def run_ck(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            A, B, As, Bs, out_dtype
        )

    if envs.VLLM_BATCH_INVARIANT:
        return run_ck(A, B, As, Bs)

    condition = A.shape[0] <= threshold

    return torch.cond(
        condition,
        run_triton,
        run_ck,
        (A, B, As, Bs),
    )


def _dynamic_aiter_triton_ck_blockscale_gemm_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fake/meta implementation for torch.compile graph tracing."""
    return torch.empty(
        A.shape[0], B.shape[0], dtype=out_dtype, device=A.device
    )


direct_register_custom_op(
    "dynamic_aiter_triton_ck_blockscale_gemm",
    _dynamic_aiter_triton_ck_blockscale_gemm_impl,
    fake_impl=_dynamic_aiter_triton_ck_blockscale_gemm_fake,
)
