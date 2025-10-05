# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def is_rocm_triton_rotary_embedding_enabled() -> bool:
    return (
        current_platform.is_rocm()
        and envs.VLLM_ROCM_USE_AITER
        and envs.VLLM_ROCM_USE_TRITON_ROPE
    )


def rocm_aiter_rotary_emb_with_key_forward_triton_impl(
    positions: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rotate_style: int = 0,
    is_nope_first: bool = False,
) -> None:
    import aiter.ops.triton.rope as ops

    ops.rope_cached_thd_positions_2c_fwd_inplace(
        query,
        key,
        cos,
        sin,
        positions,
        rotate_style,
        reuse_freqs_front_part=True,
        nope_first=is_nope_first,
    )


def rocm_aiter_rotary_emb_with_key_forward_triton_fake(
    positions: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rotate_style: int = 0,
    is_nope_first: bool = False,
) -> None:
    pass


if is_rocm_triton_rotary_embedding_enabled():
    direct_register_custom_op(
        op_name="rocm_aiter_rotary_emb_with_key_forward_triton",
        op_func=rocm_aiter_rotary_emb_with_key_forward_triton_impl,
        mutates_args=["key", "query"],
        fake_impl=rocm_aiter_rotary_emb_with_key_forward_triton_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def rocm_aiter_rotary_emb(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
):
    num_tokens = positions.numel()
    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    query_shape = query.shape
    key_shape = key.shape
    rotate_style = 0 if is_neox_style else 1

    query = query.view(num_tokens, -1, head_size)
    key = key.view(num_tokens, -1, head_size)
    query_ = query[..., :rotary_dim]
    key_ = key[..., :rotary_dim]
    positions = positions.view(*query.shape[:1])
    torch.ops.vllm.rocm_aiter_rotary_emb_with_key_forward_triton(
        positions,
        sin,
        cos,
        query_,
        key_,
        rotate_style,
        False,
    )
    query = query.view(query_shape)
    key = key.view(key_shape)
