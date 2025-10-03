# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom ops for FP8 collective operations.

This module registers custom ops for FP8-optimized collective operations that
enable pattern matching in torch.compile's FX graph. While the implementations
are functionally identical to their non-FP8 counterparts, having separate op
registrations allows the compiler to distinguish between BF16 and FP8 code paths
for applying different fusion strategies.
"""

import torch

from vllm.distributed import get_tp_group
from vllm.utils import direct_register_custom_op


def vllm_all_gather_fp8_impl(
    x: torch.Tensor,
    dim: int,
    world_size: int,
    group_name: str,
) -> torch.Tensor:
    """All-gather FP8 tensor across tensor-parallel group.

    This is functionally identical to torch.ops.vllm.all_gather, but
    is registered as a separate op to enable FP8-specific pattern matching
    in the AsyncTP fusion pass.

    Args:
        x: Input FP8 tensor to gather (typically float8_e4m3fn)
        dim: Dimension along which to gather (typically 0 for sequence dim)
        world_size: Number of ranks in the tensor-parallel group
        group_name: Name of the tensor-parallel process group

    Returns:
        Gathered tensor with shape expanded by world_size along dim
    """
    return get_tp_group().all_gather(x, dim)


def vllm_all_gather_fp8_fake(
    x: torch.Tensor,
    dim: int,
    world_size: int,
    group_name: str,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return x.repeat_interleave(world_size, dim=dim)


# Register custom op for FP8 AllGather
direct_register_custom_op(
    op_name="vllm_all_gather_fp8",
    op_func=vllm_all_gather_fp8_impl,
    mutates_args=[],
    fake_impl=vllm_all_gather_fp8_fake,
)

# Export op
vllm_all_gather_fp8 = torch.ops.vllm.vllm_all_gather_fp8.default
