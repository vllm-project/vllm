# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm import ir
from vllm.platforms import current_platform

OINK_AVAILABLE = current_platform.has_device_capability(100) and hasattr(
    torch.ops, "oink"
)


def has_oink_op(name: str) -> bool:
    """Check if a specific oink op is registered."""
    return OINK_AVAILABLE and hasattr(torch.ops.oink, name)


def _can_view_as_2d(x: torch.Tensor) -> bool:
    """Return True if x.view(-1, x.shape[-1]) is viewable (no copy)."""
    if x.dim() < 2:
        return False
    if x.dim() == 2:
        return True
    # For a view(-1, N) to be valid, all leading dims must be contiguous with
    # respect to each other (size-1 dims are ignored).
    for dim in range(x.dim() - 1):
        # Strides for size-1 dims are irrelevant and can be arbitrary.
        if x.size(dim + 1) != 1 and x.stride(dim) != x.stride(dim + 1) * x.size(
            dim + 1
        ):
            return False
    return True


def _is_oink_stride_compatible_2d(x_2d: torch.Tensor) -> bool:
    """Return True if x_2d meets Oink's pointer-path stride constraints."""
    if x_2d.dim() != 2:
        return False
    if x_2d.stride(1) != 1:
        return False
    # Match Oink's vectorization constraint: stride(0) divisible by 256b.
    if x_2d.dtype in (torch.float16, torch.bfloat16):
        divby = 16
    elif x_2d.dtype == torch.float32:
        divby = 8
    else:
        return False
    return (x_2d.stride(0) % divby) == 0


oink_rms_supported = (
    lambda x, w, e, var_size=None: var_size is None
    and w is not None
    and x.dim() >= 2
    and x.dtype == w.dtype
    and w.is_contiguous()
    and _can_view_as_2d(x)
    and _is_oink_stride_compatible_2d(x.view(-1, x.shape[-1]))
)
"""
Oink rms only supports 2d-like inputs with contiguous weight 
and no variance_size override.
"""


@ir.ops.rms_norm.register_impl(
    "oink", supports_args=oink_rms_supported, supported=has_oink_op("rms_norm")
)
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> torch.Tensor:
    assert variance_size is None
    x_2d = x.view(-1, x.shape[-1])
    return torch.ops.oink.rmsnorm(x_2d, weight, epsilon).view_as(x)
