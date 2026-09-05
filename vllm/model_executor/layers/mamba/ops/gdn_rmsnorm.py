# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_SUPPORTED_HIDDEN_SIZES = frozenset({64, 96, 128, 192, 256})
_LARGE_CONFIG_ROWS = 262144
_MAX_WIDE_DIM_ROWS = 131072
_IS_SM103 = current_platform.get_device_capability() == (10, 3)


@triton.jit
def _fused_gdn_rmsnorm_gated_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    output_ptr,
    rows,
    stride_z_token,
    stride_z_head,
    eps,
    HEADS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    column = tl.arange(0, BLOCK_D)
    mask = (row[:, None] < rows) & (column[None, :] < HIDDEN_SIZE)

    x = tl.load(
        x_ptr + row[:, None] * HIDDEN_SIZE + column[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / HIDDEN_SIZE
    reciprocal_std = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + column, mask=column < HIDDEN_SIZE, other=0.0).to(
        tl.float32
    )

    token = row // HEADS
    head = row % HEADS
    z_offsets = (
        token[:, None] * stride_z_token
        + head[:, None] * stride_z_head
        + column[None, :]
    )
    z = tl.load(z_ptr + z_offsets, mask=mask, other=0.0).to(tl.float32)
    output = x * reciprocal_std[:, None] * weight[None, :]
    output *= z * tl.sigmoid(z)
    tl.store(
        output_ptr + row[:, None] * HIDDEN_SIZE + column[None, :],
        output,
        mask=mask,
    )


def _launch_config(hidden_size: int, rows: int) -> tuple[int, int]:
    if hidden_size == 64:
        return (32 if rows >= _LARGE_CONFIG_ROWS else 16), 4
    if hidden_size == 96:
        return (32 if rows >= _LARGE_CONFIG_ROWS else 8), 4
    if hidden_size == 128:
        return (16 if rows >= _LARGE_CONFIG_ROWS else 8), 4
    if hidden_size in (192, 256):
        return (8 if rows >= _LARGE_CONFIG_ROWS else 4), 4
    raise ValueError(f"Unsupported GDN RMSNorm hidden size: {hidden_size}")


def fused_gdn_rmsnorm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply RMSNorm and a strided SiLU gate in one Triton kernel."""
    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    block_m, num_warps = _launch_config(hidden_size, rows)
    output = torch.empty_like(x)
    _fused_gdn_rmsnorm_gated_kernel[(triton.cdiv(rows, block_m),)](
        x,
        z,
        weight,
        output,
        rows,
        z.stride(0),
        z.stride(1),
        eps,
        HEADS=x.shape[1],
        HIDDEN_SIZE=hidden_size,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        BLOCK_M=block_m,
        num_warps=num_warps,
    )
    return output


def _fused_gdn_rmsnorm_gated_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="fused_gdn_rmsnorm_gated",
    op_func=fused_gdn_rmsnorm_gated,
    mutates_args=[],
    fake_impl=_fused_gdn_rmsnorm_gated_fake,
)


def fused_gdn_rmsnorm_gated_op(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.ops.vllm.fused_gdn_rmsnorm_gated(x, z, weight, eps)


def try_fused_gdn_rmsnorm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int | None,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor | None:
    """Return the SM103 fast path output, or ``None`` when unsupported."""
    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    eligible = (
        _IS_SM103
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and z.dtype == x.dtype
        and weight.dtype == x.dtype
        and x.device == z.device == weight.device
        and x.ndim == 3
        and x.is_contiguous()
        and z.shape == x.shape
        and z.stride(-1) == 1
        and weight.shape == (hidden_size,)
        and weight.stride(0) == 1
        and hidden_size in _SUPPORTED_HIDDEN_SIZES
        and group_size is None
        and norm_before_gate
        and activation in ("silu", "swish")
        and not (hidden_size in (192, 256) and rows > _MAX_WIDE_DIM_ROWS)
    )
    if not eligible:
        return None
    return fused_gdn_rmsnorm_gated_op(x, z, weight, eps)
