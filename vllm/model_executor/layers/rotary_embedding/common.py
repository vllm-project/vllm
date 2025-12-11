# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable
from functools import cache
from importlib.util import find_spec

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

if current_platform.is_cuda():
    from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

logger = init_logger(__name__)


# common functions
def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_rotary_emb_dispatch(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox_style: bool
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    if current_platform.is_cuda():
        return apply_rotary_emb(x.unsqueeze(0), cos, sin, not is_neox_style).squeeze(0)
    else:
        return apply_rotary_emb_torch(x, cos, sin, is_neox_style)


@cache
def dispatch_rotary_emb_function(
    default: Callable[..., torch.Tensor] | None = None,
) -> Callable[..., torch.Tensor]:
    if current_platform.is_cuda():
        return apply_rotary_emb

    # if torch compile is not enabled
    # use rotary embedding function from flash_attn package
    # otherwise use the naive pytorch embedding implementation
    # is faster when torch compile is enabled.
    if current_platform.is_rocm() and not torch.compiler.is_compiling():
        if find_spec("flash_attn") is not None:
            from flash_attn.ops.triton.rotary import apply_rotary

            return apply_rotary
        else:
            logger.warning(
                "flash_attn is not installed. Falling back to PyTorch "
                "implementation for rotary embeddings."
            )
    if default is not None:
        return default

    return apply_rotary_emb_torch


# yarn functions
# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
    truncate: bool = True,
) -> tuple[float | int, float | int]:
    low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype
) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def _flashinfer_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Custom op wrapper for flashinfer's rotary embedding.

    This is an in-place operation that modifies query and key tensors directly.
    """
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


def _flashinfer_rotary_embedding_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    return


# Register flashinfer rotary embedding custom op
direct_register_custom_op(
    op_name="flashinfer_rotary_embedding",
    op_func=_flashinfer_rotary_embedding,
    mutates_args=["query", "key"],  # These tensors are modified in-place
    fake_impl=_flashinfer_rotary_embedding_fake,
)
