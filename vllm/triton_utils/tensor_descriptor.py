# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def use_tensor_descriptor(override: bool | None = None) -> bool:
    """Tri-state VLLM_TRITON_USE_TD: unset=auto (on for XPU), 1/0=force on/off."""
    from vllm import envs
    from vllm.platforms import current_platform

    if override is None:
        override = envs.VLLM_TRITON_USE_TD
    if override is not None:
        return override
    return current_platform.is_xpu()


def tensor_descriptor_compatible(*tensors: "torch.Tensor | None") -> bool:
    """Whether every tensor can back a tensor descriptor spanning its last dim.
    ``None`` (unused optional operand) is accepted."""
    for t in tensors:
        if t is None:
            continue
        elem = t.element_size()
        width = t.shape[-1] * elem
        # Intel 2D block IO: width/pitch >= 64 B, multiples of 16 B (covers TMA).
        # Size-1 dim strides are checked too: kernels may pass them as a pitch.
        min_pitch = max(64, width)
        if (
            t.stride(-1) != 1
            or t.data_ptr() % 16 != 0
            or width < 64
            or width % 16 != 0
            or any(s * elem < min_pitch or s * elem % 16 for s in t.stride()[:-1])
        ):
            return False
    return True
