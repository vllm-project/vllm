# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-expert RMSNorm written once per route, in the ``[T * top_k, H]`` layout
the ROCm MXFP4 (AITER top-1 per route) experts take."""

import torch

from vllm.triton_utils import tl, triton

# Set on the returned first-route view; holds the full expanded buffer.
_EXPANDED_ATTR = "_muse_route_expanded"


@triton.jit
def _rmsnorm_expand_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    D: tl.constexpr,
    eps,
    gain_center,
    REPEATS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    dm = d < D
    x = tl.load(x_ptr + t * D + d, mask=dm, other=0.0).to(tl.float32)
    if HAS_WEIGHT:
        w = tl.load(w_ptr + d, mask=dm, other=0.0).to(tl.float32) + gain_center
        if INPUT_SCALE:
            x = x * w
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
    if HAS_WEIGHT and not INPUT_SCALE:
        y = y * w
    y = y.to(out_ptr.dtype.element_ty)
    for r in tl.static_range(REPEATS):
        tl.store(out_ptr + (t * REPEATS + r) * D + d, y, mask=dm)


def fused_rmsnorm_expand(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    gain_center: float,
    input_scale: bool,
    repeats: int,
) -> torch.Tensor:
    """RMSNorm of ``x`` written ``repeats`` times per row."""
    assert x.dim() == 2 and x.is_contiguous()
    T, D = x.shape
    out = torch.empty((T, repeats, D), dtype=x.dtype, device=x.device)
    if T > 0:
        block_d = triton.next_power_of_2(D)
        _rmsnorm_expand_kernel[(T,)](
            x,
            weight if weight is not None else x,
            out,
            D=D,
            eps=eps,
            gain_center=gain_center,
            REPEATS=repeats,
            BLOCK_D=block_d,
            HAS_WEIGHT=weight is not None,
            INPUT_SCALE=input_scale and weight is not None,
            num_warps=8 if block_d >= 4096 else 4,
        )
    first = out[:, 0]
    setattr(first, _EXPANDED_ATTR, out)
    return first


def take_expanded(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor | None:
    """The ``[T * repeats, D]`` buffer behind a :func:`fused_rmsnorm_expand`
    result, or ``None`` if ``hidden_states`` is not one."""
    out = getattr(hidden_states, _EXPANDED_ATTR, None)
    if (
        out is None
        or out.data_ptr() != hidden_states.data_ptr()
        or out.shape != (hidden_states.shape[0], repeats, hidden_states.shape[1])
        or hidden_states.stride() != (repeats * hidden_states.shape[1], 1)
    ):
        return None
    return out.view(-1, hidden_states.shape[1])
