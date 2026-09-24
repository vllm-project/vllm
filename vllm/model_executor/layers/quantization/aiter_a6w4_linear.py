# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A6 dense linear on AITER's ASM a6w4 GEMM (gfx950).

MXFP6-E2M3 activations x MXFP4 weights, per-1x32 E8M0 scales, via
``aiter.ops.gemm_op_a6w4.gemm_a6w4``. Unlike the FlyDSL path this needs nothing
beyond released AITER, imposes no N/K divisibility constraint, and gets a native
fused activation quantizer instead of a PyTorch reference.

**The operands carry a Hadamard rotation.** Both ``quant_mxfp6_gemm`` (A) and
``quant_mxfp4_gemm`` (B) apply a block-diagonal 32-point Walsh-Hadamard along K
before quantizing; the two rotations cancel inside the GEMM. The consequence for
a *statically* quantized checkpoint is that its stored MXFP4 weights cannot be
handed to the kernel as-is: they must be dequantized and requantized through
``quant_mxfp4_gemm`` so the rotation is applied. That is a second, independent
quantization, and its error adds in quadrature with the checkpoint's own --
measured on gfx950 as 1.62e-1 against a bf16 reference where the checkpoint's
own floor is 1.12e-1, i.e. ~1.45x, consistently across shapes.

So this path trades accuracy for speed: ~37 us/layer (24 us GEMM + 13 us
activation quant at N=K=4096) against ~527 us for the FlyDSL path on stock
AITER, whose fp6 activation quantizer falls back to a ~510 us PyTorch
reference. Callers that need the checkpoint's exact numerics should prefer the
FlyDSL path where available, or the high-precision emulation.
"""

from __future__ import annotations

import functools

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_IMPORT_ERROR: str | None = None
try:
    from aiter.ops.gemm_op_a6w4 import gemm_a6w4, quant_mxfp4_gemm
    from aiter.ops.gemm_op_a6w6 import quant_mxfp6_gemm
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    _IMPORTED = True
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = repr(exc)
    _IMPORTED = False


@functools.cache
def is_aiter_a6w4_supported() -> bool:
    """Whether AITER's ASM a6w4 GEMM can run here (gfx950 + importable)."""
    if not _IMPORTED or not current_platform.is_rocm():
        return False
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950"
    except Exception:  # noqa: BLE001
        return False


def why_unavailable() -> str:
    """Human-readable reason the AITER a6w4 path is off, for one-time logging."""
    if not _IMPORTED:
        return f"aiter.ops.gemm_op_a6w4 not importable ({_IMPORT_ERROR})"
    if not current_platform.is_rocm():
        return "not on ROCm"
    return "device is not gfx950"


def prepare_weight(weight: torch.Tensor, weight_scale: torch.Tensor):
    """Checkpoint MXFP4 -> the kernel's packed, Hadamard-rotated B operand.

    Done once at load. ``weight`` is ``[N, K//2]`` packed MXFP4 codes and
    ``weight_scale`` is ``[N, K//32]`` E8M0, exactly as Quark exports them.

    The dequantization deliberately does not go through
    ``quantization/utils/mxfp4_utils.dequant_mxfp4``: that dispatches to Quark's
    HIP kernel, which expects a different scale layout and silently returns
    wrong values for this one. Reconstructing from AITER's own
    ``mxfp4_to_f32`` / ``e8m0_to_f32`` matches the layout Quark wrote.

    Returns ``(packed_b, packed_b_scale)`` flat uint8 buffers.

    Dequantizing in row chunks rather than whole: ``mxfp4_to_f32``, the
    ``repeat_interleave`` of the scales and their product are each a full-size
    fp32 tensor, so the naive expression peaks at 7.5x the bf16 operand it
    produces (measured 3.75 GiB for a 16384x16384 layer whose operand is 0.50
    GiB). On the largest shape AITER itself tunes for, 106496x16384, that is
    ~24 GiB of transient allocation per layer at load.
    """
    n, k = weight.shape[0], weight.shape[1] * 2
    out = torch.empty((n, k), dtype=torch.bfloat16, device=weight.device)
    # ~256 MiB of fp32 scratch per chunk, and at least one row.
    rows = max(1, min(n, (1 << 26) // max(k, 1)))
    for i in range(0, n, rows):
        j = min(i + rows, n)
        out[i:j] = (
            mxfp4_to_f32(weight[i:j])
            * e8m0_to_f32(weight_scale[i:j].repeat_interleave(32, dim=1))
        ).bfloat16()
    return quant_mxfp4_gemm(out)


def _aiter_a6w4_linear_impl(
    x: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """MXFP6-E2M3(x) @ MXFP4(weight).T through AITER's ASM a6w4 GEMM."""
    m = x.shape[0]
    # gemm_a6w4 rejects non-positive dims; an empty batch is legitimate input
    # from vLLM (an empty sequence, or a [B, 0, K] reshape), so answer it here.
    if m == 0:
        return torch.empty((0, n), dtype=torch.bfloat16, device=x.device)
    a_packed, a_scale = quant_mxfp6_gemm(x.contiguous())
    out = gemm_a6w4(a_packed, b_packed, a_scale, b_scale, m, n, k)
    # gemm_a6w4 returns out[:M, :N] of a tile-padded [padM, padN] buffer. That
    # slice is non-contiguous when N is unaligned, so .contiguous() copies. When
    # N *is* aligned but M < padM it is already contiguous, so .contiguous()
    # would be a no-op and the small logical result would keep the whole padded
    # allocation alive -- 16 MiB behind a 64 KiB answer at M=1, N=32768, on
    # every decode step. Clone when the backing storage is materially larger.
    if out.untyped_storage().size() > 2 * out.numel() * out.element_size():
        return out.clone()
    return out.contiguous()


def _aiter_a6w4_linear_fake(
    x: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    return torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)


if _IMPORTED:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="aiter_a6w4_linear",
        op_func=_aiter_a6w4_linear_impl,
        mutates_args=[],
        fake_impl=_aiter_a6w4_linear_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def aiter_a6w4_linear(
    x: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Public entrypoint: dispatches through the registered custom op.

    The kernel emits bfloat16 only, so callers must not route fp16 layers here.
    """
    return torch.ops.vllm.aiter_a6w4_linear(x, b_packed, b_scale, n, k)
