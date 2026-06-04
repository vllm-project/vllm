# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SwiGLU-OAI activation (split layout) for AMD ROCm via Triton.

SwiGLU-OAI on a ``[*, 2I]`` split-layout input (gate = first half, up = second
half):

    gate = clamp(gate, max=limit)
    up   = clamp(up, -limit, +limit)
    out  = gate * sigmoid(alpha * gate) * (up + beta)

On ROCm the dense MLP and the native MXFP8 MoE (between its two GEMMs) fell back
to a chain of elementwise PyTorch ops with fp32 intermediates: vLLM's shared
``SiluAndMulWithClamp`` blanket-routes ROCm to ``forward_native``, and the MoE
applies the activation inline in PyTorch. This Triton kernel collapses that into
a single pass producing the ``[*, I]`` output directly, and computes in fp32
(rel ~1e-6 vs reference).

Note: the vectorized ``torch.ops._C.silu_and_mul_with_clamp`` op IS built on
ROCm and is ~1.2-2.2x faster in isolation, but the win is launch overhead that
HIP graphs already eliminate — measured end-to-end throughput is identical
(within noise), so we keep the fp32-accurate Triton kernel.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _swiglu_oai_kernel(
    g_ptr,
    out_ptr,
    n_inter,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    alpha,
    beta,
    limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    pid_i = tl.program_id(1)
    cols = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < n_inter
    gate = tl.load(
        g_ptr + row * stride_gm + cols * stride_gn, mask=mask, other=0.0
    ).to(tl.float32)
    up = tl.load(
        g_ptr + row * stride_gm + (n_inter + cols) * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    tl.store(
        out_ptr + row * stride_om + cols * stride_on,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def swiglu_oai_split(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """SwiGLU-OAI on a split-layout ``[*, 2I]`` tensor -> ``[*, I]``."""
    orig_shape = gate_up.shape
    two_i = orig_shape[-1]
    n_inter = two_i // 2
    x2 = gate_up.reshape(-1, two_i)
    m = x2.shape[0]
    dt = out_dtype if out_dtype is not None else gate_up.dtype
    out = torch.empty((m, n_inter), dtype=dt, device=gate_up.device)
    # Tile tuned on gfx950. The SwiGLU intermediate is sharded across tensor
    # parallel ranks (per-rank n_inter = I / tp: dense I=12288, MoE I=3072), and
    # a 512-wide tile (4 warps, ~2 elems/lane) only helps once the per-rank slice
    # is large enough to be bandwidth-bound — at TP=1 prefill that is ~1.25-1.35x
    # faster than 256. For small sharded slices (high TP) the kernel is launch-
    # bound (~12us) and a wide tile can slightly regress, so fall back to 256.
    # Decode is launch-bound at every TP. num_warps=8 underfills this tile, so it
    # is pinned to 4.
    block_i = 512 if n_inter >= 2048 else 256
    grid = (m, triton.cdiv(n_inter, block_i))
    _swiglu_oai_kernel[grid](
        x2,
        out,
        n_inter,
        x2.stride(0),
        x2.stride(1),
        out.stride(0),
        out.stride(1),
        float(alpha),
        float(beta),
        0.0 if limit is None else float(limit),
        HAS_LIMIT=limit is not None,
        BLOCK_I=block_i,
        num_warps=4,
    )
    return out.reshape(*orig_shape[:-1], n_inter)
