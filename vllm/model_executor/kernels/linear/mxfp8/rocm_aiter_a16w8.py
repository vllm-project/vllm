# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 linear for decode on AMD CDNA3 (gfx942) via AITER's asm A16W8 GEMM.

gfx942 has no MX matrix cores, so MXFP8 checkpoints otherwise run through
``EmulationMxfp8LinearKernel``: the weights are dequantized to BF16 once and
every GEMM is a BF16 hipBLASLt call. At decode sizes those calls are latency
bound and read twice the checkpoint's weight bytes.

``aiter.gemm_a16w8_mxfp8_asm`` reads the FP8 weight bytes and E8M0 block
scales directly, dequantizes them in registers (exactly) and keeps the
activations in BF16, for 1 to 64 rows. AITER ships a tuned table; this kernel
is chosen only for weights whose (N, K) is in it, and a call goes to AITER only
when its row count is tuned. Everything else (prefill chunks, untuned row
counts) runs the same BF16 linear as the emulation kernel, so the BF16 copy of
the weight is kept next to the MXFP8 one.

Opt in with ``VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MXFP8_ASM_GEMM=1``.
"""

import functools

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    dequant_mxfp8_to_bf16,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

logger = init_logger(__name__)

# The asm kernels take at most 64 rows; AITER's table says which are tuned.
_MAX_ROWS = 64


@functools.cache
def _aiter_a16w8():
    """AITER's MXFP8 A16W8 functions, or None if this AITER does not have them."""
    try:
        from aiter.ops.gemm_op_a16w8_mxfp8 import (
            gemm_a16w8_mxfp8_asm,
            gemm_a16w8_mxfp8_prepare_weight,
            is_gemm_a16w8_mxfp8_tuned,
        )
    except ImportError:
        return None
    return (
        gemm_a16w8_mxfp8_asm,
        gemm_a16w8_mxfp8_prepare_weight,
        is_gemm_a16w8_mxfp8_tuned,
    )


@functools.cache
def _tuned(rows: int, n: int, k: int) -> bool:
    funcs = _aiter_a16w8()
    return funcs is not None and 0 < rows <= _MAX_ROWS and funcs[2](rows, n, k)


def _any_tuned(n: int, k: int) -> bool:
    return any(_tuned(rows, n, k) for rows in range(1, _MAX_ROWS + 1))


def _rocm_aiter_a16w8_mxfp8_linear_impl(
    x: torch.Tensor,
    weight_bf16: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    rows = x.shape[0]
    n, k = weight.shape
    if x.dtype == torch.bfloat16 and _tuned(rows, n, k):
        return _aiter_a16w8()[0](x.contiguous(), weight, weight_scale)
    return torch.nn.functional.linear(x, weight_bf16.to(x.dtype))


def _rocm_aiter_a16w8_mxfp8_linear_fake(
    x: torch.Tensor,
    weight_bf16: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


# An opaque op: the choice between the asm kernel and the BF16 linear depends
# on the token count, which a compiled caller must not freeze at its trace-time
# value. CUDA graphs capture whichever path the captured size takes.
direct_register_custom_op(
    op_name="rocm_aiter_a16w8_mxfp8_linear",
    op_func=_rocm_aiter_a16w8_mxfp8_linear_impl,
    fake_impl=_rocm_aiter_a16w8_mxfp8_linear_fake,
)


class RocmAiterA16W8Mxfp8LinearKernel(Mxfp8LinearKernel):
    """gfx942 MXFP8 linear: AITER asm A16W8 GEMM for decode, BF16 otherwise."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "not ROCm"
        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_asm_mxfp8_gemm_enabled():
            return (
                False,
                "needs gfx942, VLLM_ROCM_USE_AITER=1 and "
                "VLLM_ROCM_USE_AITER_MXFP8_ASM_GEMM=1",
            )
        if _aiter_a16w8() is None:
            return False, "this AITER has no gemm_a16w8_mxfp8_asm"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        n, k = c.weight_shape
        if n % 16 or k % MXFP8_BLOCK_SIZE:
            return False, f"weight {n} x {k}: needs N % 16 == 0 and K % 32 == 0"
        if not _any_tuned(n, k):
            return False, f"weight {n} x {k} is not in AITER's tuned table"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K] e4m3fn
        n, k = weight.shape
        weight_scale = layer.weight_scale.data[:n, : k // MXFP8_BLOCK_SIZE].contiguous()
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"{type(self).__name__} expects float8_e4m3fn weights, "
                f"got {weight.dtype}"
            )
        prepare = _aiter_a16w8()[1]
        weight_mx, scale_mx = prepare(weight, weight_scale)
        # The BF16 copy serves prefill and untuned row counts.
        weight_bf16 = dequant_mxfp8_to_bf16(weight.contiguous(), weight_scale)
        layer.weight = Parameter(weight_bf16.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        layer.weight_mxfp8 = Parameter(weight_mx, requires_grad=False)
        layer.weight_scale_mxfp8 = Parameter(scale_mx, requires_grad=False)
        logger.debug(
            "AITER MXFP8 A16W8 asm GEMM for %d x %d up to %d rows",
            n,
            k,
            max((r for r in range(1, _MAX_ROWS + 1) if _tuned(r, n, k)), default=0),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = layer.weight.shape[0]
        out = torch.ops.vllm.rocm_aiter_a16w8_mxfp8_linear(
            x.reshape(-1, x.shape[-1]),
            layer.weight,
            layer.weight_mxfp8,
            layer.weight_scale_mxfp8,
        ).reshape(*x.shape[:-1], n)
        if bias is not None:
            out = out + bias
        return out
