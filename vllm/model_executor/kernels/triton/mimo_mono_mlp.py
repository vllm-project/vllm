# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental gfx950 MiMo dense-MPL mono kernel.

The kernel fuses SwiGLU, dynamic FP8 group-128 activation quantization and the
block-scaled FP8 down projection. It consumes the checkpoint's unshuffled
``[N, K]`` E4M3 weights and ``[N/128, K/128]`` FP32 dequant scales.

One output tile recomputes the SwiGLU activation. This is deliberate: it avoids
materializing the ``[M, 16384]`` activation but trades redundant gate/up reads
for output parallelism. It is an experimental decode kernel, not a claim that
it improves the production AITER path.
"""

import torch
from torch import Tensor

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

_HIDDEN = 6144
_INTERMEDIATE = 16384
_BLOCK = 128
_MAX_ROWS = 16

_LOGGER = init_logger(__name__)
_logged = False


@triton.jit
def _mimo_fp8_mono_mlp_kernel(
    gate_up_ptr,
    weight_ptr,
    weight_scale_ptr,
    out_ptr,
    rows,
    weight_stride_n,
    BLOCK_N: tl.constexpr,
    HIDDEN: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    tile_n = tl.program_id(1)
    n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for block_k in tl.static_range(0, INTERMEDIATE, BLOCK_K):
        offset_k = block_k + k
        gate = tl.load(gate_up_ptr + row * (2 * INTERMEDIATE) + offset_k).to(tl.float32)
        up = tl.load(
            gate_up_ptr + row * (2 * INTERMEDIATE) + INTERMEDIATE + offset_k
        ).to(tl.float32)
        activation = gate * tl.sigmoid(gate) * up
        scale = tl.maximum(tl.max(tl.abs(activation), axis=0), 1e-10) / 448.0
        activation_fp8 = tl.clamp(activation / scale, -448.0, 448.0).to(
            weight_ptr.dtype.element_ty
        )
        weight = tl.load(
            weight_ptr + n[:, None] * weight_stride_n + offset_k[None, :],
            mask=n[:, None] < HIDDEN,
            other=0.0,
        )
        weight_scale = tl.load(
            weight_scale_ptr
            + (n // BLOCK_K) * (INTERMEDIATE // BLOCK_K)
            + (block_k // BLOCK_K),
            mask=n < HIDDEN,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(
            weight.to(tl.float32) * activation_fp8[None, :].to(tl.float32), axis=1
        ) * (scale * weight_scale)

    tl.store(
        out_ptr + row * HIDDEN + n,
        acc.to(out_ptr.dtype.element_ty),
        mask=n < HIDDEN,
    )


def can_use_mimo_fp8_mono_mlp(
    gate_up: Tensor,
    down_weight: Tensor,
    down_scale: Tensor,
    *,
    tensor_parallel_size: int,
    is_capturing: bool,
) -> bool:
    """Return whether the explicit experimental layout contract is met."""
    return (
        not is_capturing
        and tensor_parallel_size in (1, 8)
        and gate_up.device.type == "cuda"
        and gate_up.dtype is torch.bfloat16
        and down_weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        and down_scale.dtype is torch.float32
        and gate_up.ndim == down_weight.ndim == down_scale.ndim == 2
        and 0 < gate_up.shape[0] <= _MAX_ROWS
        and gate_up.shape[1] == 2 * down_weight.shape[1]
        and down_weight.shape[0] == _HIDDEN
        and down_weight.shape[1] % _BLOCK == 0
        and down_scale.shape == (_HIDDEN // _BLOCK, down_weight.shape[1] // _BLOCK)
        and gate_up.is_contiguous()
        and down_weight.stride(1) == 1
        and down_weight.stride(0) >= down_weight.shape[1]
        and down_scale.is_contiguous()
    )


def mimo_fp8_mono_mlp(
    gate_up: Tensor, down_weight: Tensor, down_scale: Tensor
) -> Tensor:
    """Launch the experimental one-kernel MiMo SwiGLU/down projection."""
    global _logged
    if not _logged:
        _LOGGER.info("Using experimental MiMo FP8 blockscale mono MLP")
        _logged = True
    output = torch.empty(
        (gate_up.shape[0], _HIDDEN), dtype=gate_up.dtype, device=gate_up.device
    )
    intermediate = down_weight.shape[1]
    _mimo_fp8_mono_mlp_kernel[(gate_up.shape[0], _HIDDEN // _BLOCK)](
        gate_up,
        down_weight,
        down_scale,
        output,
        gate_up.shape[0],
        down_weight.stride(0),
        BLOCK_N=_BLOCK,
        HIDDEN=_HIDDEN,
        INTERMEDIATE=intermediate,
        BLOCK_K=_BLOCK,
        num_warps=4,
    )
    return output
