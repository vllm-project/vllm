# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 block-scaled GEMM with LL kernel for M<=16 decode shapes.

Requires SM100+ because transform_sf_into_required_layout produces
packed ue8m0 int32 scales on SM100 (arch_major==10) but fp32 scales
on SM90 (arch_major==9). The kernel's ue8m0_to_f32 byte extraction
only handles the packed int32 format.

Dispatch to the LL kernel happens inside a custom op (opaque to
torch.compile). apply_block_scaled_mm always calls this single op
with no branches, so the compiled graph is identical in structure
to DeepGemm's (one output alloc + one custom op call).
"""

from __future__ import annotations

import logging

import torch

from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

# ── CuTe DSL availability ────────────────────────────────────────────

_cutedsl_available: bool | None = None


def is_available() -> bool:
    global _cutedsl_available
    if _cutedsl_available is not None:
        return _cutedsl_available
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        _cutedsl_available = True
    except ImportError:
        _cutedsl_available = False
        logger.info("cuteDSL not available, ll_fp8_block_gemm disabled")
    return _cutedsl_available


# ── Compile cache ─────────────────────────────────────────────────────

_compiled_cache: dict[tuple, object] = {}
_cute_ctx = None


def _cute():
    global _cute_ctx
    if _cute_ctx is not None:
        return _cute_ctx
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream

    _cute_ctx = (cute, from_dlpack, CUstream, current_stream)
    return _cute_ctx


def _stream():
    _, _, CUstream, current_stream = _cute()
    return CUstream(current_stream().cuda_stream)


def _get_compiled(a_bf16, b_bf16, out, sa_flat, sb_flat):
    cute, from_dlpack, _, _ = _cute()
    from ._ll_fp8_block_warpspecialized import LLFp8BlockGemm

    # TODO (roberto): add tile_n, tile_k, num_stages to autotuning space
    cache_key = ("fp8_block",)
    if cache_key in _compiled_cache:
        return _compiled_cache[cache_key]

    div = 8
    mA = (
        from_dlpack(a_bf16, assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )
    mB = (
        from_dlpack(b_bf16, assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )
    mC = (
        from_dlpack(out, assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )
    mSA = from_dlpack(
        sa_flat, assumed_align=4, enable_tvm_ffi=True
    ).mark_layout_dynamic()
    mSB = from_dlpack(
        sb_flat, assumed_align=4, enable_tvm_ffi=True
    ).mark_layout_dynamic()

    gemm = LLFp8BlockGemm(tile_n=16, tile_k=256, num_stages=2, num_dma_warps=4)
    compiled = cute.compile(
        gemm, mA, mB, mC, mSA, mSB, _stream(), options="--enable-tvm-ffi"
    )
    _compiled_cache[cache_key] = compiled
    logger.info("Compiled ll_fp8_block_gemm")
    return compiled


def _ll_fp8_block_gemm(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    a_bf16 = q_input.view(torch.bfloat16)
    b_bf16 = weight.view(torch.bfloat16)
    sa_flat = input_scale.as_strided((input_scale.numel(),), (1,))
    sb_flat = weight_scale.as_strided((weight_scale.numel(),), (1,))

    compiled = _get_compiled(a_bf16, b_bf16, output, sa_flat, sb_flat)
    compiled(a_bf16, b_bf16, output, sa_flat, sb_flat, _stream())


# ── Kernel class (registered in scaled_mm registry) ───────────────────


class LLFp8BlockScaledMMKernel(DeepGemmFp8BlockScaledMMKernel):
    """FP8 block-scaled kernel that dispatches to LL kernel for M<=16.

    SM100+ only: scale format is packed ue8m0 int32 (not fp32 like SM90).
    """

    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)

    @classmethod
    def can_implement(cls, config):
        if not is_available():
            return False, "CuTe DSL not available"
        if not current_platform.is_device_capability_family(100):
            return False, "requires SM100+ (packed ue8m0 scale format)"
        can_base, reason = super().can_implement(config)
        if not can_base:
            return False, reason
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        output = torch.empty(
            (A.shape[0], B.shape[0]),
            dtype=out_dtype,
            device=A.device,
        )
        torch.ops.vllm.ll_fp8_block_dispatch_op(
            A, As, B, Bs, output, self.use_deep_gemm_e8m0
        )
        return output


# ── Custom op (opaque to torch.compile) ───────────────────────────────


def _ll_fp8_block_dispatch(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    if weight.ndim != 2:
        from vllm.utils.deep_gemm import fp8_gemm_nt

        fp8_gemm_nt(
            (q_input, input_scale),
            (weight, weight_scale),
            output,
            is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
        )
        return
    M = q_input.shape[0]
    K_fp8 = q_input.shape[1]
    N = weight.shape[0]
    if M <= 16 and K_fp8 <= 4096 and K_fp8 % 256 == 0 and N <= 4096:
        _ll_fp8_block_gemm(q_input, input_scale, weight, weight_scale, output)
    else:
        from vllm.utils.deep_gemm import fp8_gemm_nt

        fp8_gemm_nt(
            (q_input, input_scale),
            (weight, weight_scale),
            output,
            is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
        )


def _ll_fp8_block_dispatch_fake(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    return None


from vllm.utils.torch_utils import direct_register_custom_op

direct_register_custom_op(
    "ll_fp8_block_dispatch_op",
    _ll_fp8_block_dispatch,
    mutates_args=["output"],
    fake_impl=_ll_fp8_block_dispatch_fake,
)
