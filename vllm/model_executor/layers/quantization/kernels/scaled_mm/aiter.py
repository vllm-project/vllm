# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Callable

import torch
from aiter.ops.shuffle import shuffle_weight

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .cutlass import CutlassScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)


def rocm_aiter_gemm_w8a8_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    from aiter import gemm_a8w8_CK

    # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects
    # a to be [M, K]
    # b to be [N, K]
    # CutlassScaledMMLinearKernel prepare weight `w_q` in [K, N] format
    return gemm_a8w8_CK(A, B, As, Bs, bias, output_dtype)


def rocm_aiter_gemm_w8a8_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    m = A.shape[0]
    n = B.shape[0]
    Y = torch.empty(m, n, dtype=output_dtype, device=A.device)
    return Y


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_gemm_w8a8",
        op_func=rocm_aiter_gemm_w8a8_impl,
        fake_impl=rocm_aiter_gemm_w8a8_fake,
    )


# bpshuffle
def rocm_aiter_gemm_a8w8_bpreshuffle_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
) -> torch.Tensor:
    # This AITER function can be used for
    # - per-token activations + per-channel weights
    #   e.g. vllm/model_executor/layers/quantization/utils/w8a8_utils.py
    # accept the weight as # keep the weight as (N, K)
    # NOTE: The weight has to be shuffled in the
    # process_weights_after_loading of the CompressedTensorsW8A8Fp8 class

    from aiter import gemm_a8w8_bpreshuffle_ck

    m = input.shape[0]
    n = weight.shape[0]
    Y = torch.empty(m, n, dtype=out_dtype, device=input.device)
    gemm_a8w8_bpreshuffle_ck(input, weight, scale_a, scale_b, Y)
    return Y


def rocm_aiter_gemm_a8w8_bpreshuffle_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
) -> torch.Tensor:
    m = input.shape[0]
    n = weight.shape[0]
    if out_dtype is None:
        out_dtype = input.dtype
    return torch.empty((m, n), dtype=out_dtype, device=input.device)


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_gemm_a8w8_bpreshuffle",
        op_func=rocm_aiter_gemm_a8w8_bpreshuffle_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_gemm_a8w8_bpreshuffle_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class AiterScaledMMLinearKernel(CutlassScaledMMLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return (
                False,
                "AiterScaledMMLinearKernel requires `aiter` which is not "
                + "currently supported on non-ROCm platform.",
            )

        try:
            import aiter  # noqa: F401 # deliberately attempt to import aiter
        except Exception:
            return (
                False,
                "AiterScaledMMLinearKernel requires `aiter` which is not "
                + "installed on ROCm.",
            )
        # Check if rocm_aiter_gemm_w8a8_scaled_mm is enabled
        if not (envs.VLLM_ROCM_USE_AITER_LINEAR and envs.VLLM_ROCM_USE_AITER):
            return (
                False,
                "AiterScaledMMLinearKernel is disabled. "
                + "Enable by setting `VLLM_ROCM_USE_AITER=1` "
                + "and `VLLM_ROCM_USE_AITER_LINEAR=1`. "
                + "`VLLM_ROCM_USE_AITER_LINEAR` default is True.",
            )

        if not c.input_symmetric:
            return (
                False,
                "AiterScaledMMLinearKernel only supports symmetric " + "quantization.",
            )
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        `AiterScaledMMLinearKernel` implements a fused version of
            `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
        where scale_a * a and scale_b * b are implemented using numpy-style
        broadcasting.
        Currently only support per-tensor-per-tensor GEMM
        and per-token-per-channel GEMM through AITER
        w8a8 scaled gemm. `AiterScaledMMLinearKernel` also does not support
        ATIER block scaled GEMM and mix-precision GEMM.
        """
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        assert symmetric, (
            "AiterScaledMMLinearKernel only supports symmetric quantization."
        )
        x_q, x_s, x_zp = ops.scaled_int8_quant(x, i_s, i_zp, symmetric=symmetric)

        assert x_zp is None, (
            "AiterScaledMMLinearKernel only supports symmetric quantization."
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
            + " and per-token-per-channel GEMM through AITER"
            " w8a8 scaled gemm. `AiterScaledMMLinearKernel` "
            + "does not support AITER block scaled GEMM."
        )

        # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects
        # a to be [M, K]
        # b to be [N, K]
        # CutlassScaledMMLinearKernel prepare weight `w_q` in [K, N] format
        return torch.ops.vllm.rocm_aiter_gemm_w8a8(
            x_q, w_q.t(), x_s, w_s, bias, out_dtype
        )


# bpreshuffle
class AiterBpreshufflePerTokenFp8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    def get_ouput_padding(self) -> int | None:
        # PTPC kernels do not require padding.
        return None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return (False, "AITER bpreshuffle is ROCm-only")
        if not (envs.VLLM_ROCM_USE_AITER_LINEAR and envs.VLLM_ROCM_USE_AITER):
            return (False, "AITER bpreshuffle is disabled by env var")
        try:
            import aiter  # noqa: F401
        except Exception:
            return (False, "AITER not installed")

        # Check if the configuration is PTPC
        is_per_channel_weight = c.weight_quant_key.scale.group_shape.is_per_token()
        is_per_token_activation = (
            c.activation_quant_key.scale.group_shape.is_per_token()
        )
        is_ptpc = is_per_channel_weight and is_per_token_activation

        logger.info_once(f"AiterBpreshuffle: can_implement called. is_ptpc={is_ptpc}")

        if not is_ptpc:
            return (False, "This kernel only handles Per-Token/Per-Channel (PTPC)")

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.info_once("AiterBpreshuffle: SHUFFLING WEIGHTS NOW.")

        w_q, _, _, _ = self._get_layer_params(layer)

        N = w_q.shape[1]
        K = w_q.shape[0]

        if N % 16 == 0 and K % 16 == 0:
            # AITER shuffle_weight expectation [N, K]
            w_q_nk = w_q.t().contiguous()

            # Execute shuffle
            shuffled_w_nk = shuffle_weight(w_q_nk, layout=(16, 16))

            del layer.weight
            layer.register_buffer("weight", shuffled_w_nk)

            logger.info_once("[AiterBpreshuffle: Weight shuffle COMPLETE.")

        else:
            raise ValueError(
                f"Weight shape (N={N}, K={K}) not divisible by 16 "
                "for AITER bpreshuffle."
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Obtain parameters
        w_q, w_s, x_s, x_s_ub = self._get_layer_params(layer)
        # 2. Dynamic quantization input
        qinput, qinput_scale = self.quant_fp8(x, x_s, x_s_ub)

        logger.info_once(
            "AiterBpreshuffle: apply_weights... ABOUT TO CALL C++ KERNEL..."
        )

        # 3. Call the AITER bpreshuffle CK operator.
        output = torch.ops.vllm.rocm_aiter_gemm_a8w8_bpreshuffle(
            qinput,
            w_q,  # Input [N, K] shuffle weights
            out_dtype=self.config.out_dtype,
            scale_a=qinput_scale,
            scale_b=w_s,
        )

        logger.info_once("AiterBpreshuffle: C++ KERNEL CALL SUCCEEDED.")

        if bias is not None:
            output.add_(bias)
        return output

    def get_scaled_mm_func(self) -> Callable[..., torch.Tensor]:
        return rocm_aiter_gemm_a8w8_bpreshuffle_impl


# AITER FP8 CK
class AiterCKPerTokenFp8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """
    AITER PTPC kernel (gemm_a8w8_CK) without pre-shuffling.
    """

    def get_ouput_padding(self) -> int | None:
        return None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return (False, "AITER CK is ROCm-only")
        if not (envs.VLLM_ROCM_USE_AITER_LINEAR and envs.VLLM_ROCM_USE_AITER):
            return (False, "AITER CK is disabled by env var")
        try:
            import aiter  # noqa: F401
        except Exception:
            return (False, "AITER not installed")

        is_per_channel_weight = c.weight_quant_key.scale.group_shape.is_per_token()
        is_per_token_activation = (
            c.activation_quant_key.scale.group_shape.is_per_token()
        )
        is_ptpc = is_per_channel_weight and is_per_token_activation

        logger.info_once(f"AiterCK: can_implement called. is_ptpc={is_ptpc}")

        if not is_ptpc:
            return (False, "This kernel only handles Per-Token/Per-Channel (PTPC)")

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.info_once(
            "AITER CK: process_weights_after_loading... DOING NOTHING (pass)."
        )
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, x_s, x_s_ub = self._get_layer_params(layer)

        qinput, qinput_scale = self.quant_fp8(x, x_s, x_s_ub)

        logger.info_once(
            "AiterCK: apply_weights... "
            "ABOUT TO CALL C++ KERNEL (this is where it hangs)..."
        )

        output = torch.ops.vllm.rocm_aiter_gemm_w8a8(
            qinput, w_q.t(), qinput_scale, w_s, bias, self.config.out_dtype
        )

        logger.info_once("AiterCK: C++ KERNEL CALL SUCCEEDED.")

        return output

    def get_scaled_mm_func(self) -> Callable[..., torch.Tensor]:
        return rocm_aiter_gemm_w8a8_impl
