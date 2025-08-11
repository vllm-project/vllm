# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def is_rocm_rotary_embedding_enabled() -> bool:
    return (current_platform.is_rocm() and envs.VLLM_ROCM_USE_AITER)


def rocm_aiter_rotary_emb_without_key_forward_hip_impl(
    positions: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    query: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    rotate_style: int = 0,
    is_nope_first: bool = False,
) -> None:
    import aiter as ops
    if offsets is None:
        ops.rope_cached_positions_fwd_inplace(
            query,
            cos,
            sin,
            positions,
            rotate_style,
            reuse_freqs_front_part=True,
            nope_first=is_nope_first,
        )
    else:
        ops.rope_cached_positions_offsets_fwd_inplace(
            query,
            cos,
            sin,
            positions,
            offsets,
            rotate_style,
            reuse_freqs_front_part=True,
            nope_first=is_nope_first,
        )


def rocm_aiter_rotary_emb_with_key_forward_hip_impl(
    positions: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    rotate_style: int = 0,
    is_nope_first: bool = False,
) -> None:
    import aiter as ops
    if offsets is None:
        ops.rope_cached_positions_2c_fwd_inplace(
            query,
            key,
            cos,
            sin,
            positions,
            rotate_style,
            reuse_freqs_front_part=True,
            nope_first=is_nope_first,
        )
    else:
        ops.rope_cached_positions_offsets_2c_fwd_inplace(
            query,
            key,
            cos,
            sin,
            positions,
            offsets,
            rotate_style,
            reuse_freqs_front_part=True,
            nope_first=is_nope_first,
        )


def rocm_aiter_rotary_emb_with_key_forward_hip_fake(
    positions: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    rotate_style: int = 0,
    is_nope_first: bool = False,
) -> None:
    pass


def rocm_aiter_rotary_emb_without_key_forward_hip_fake(
    positions: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    query: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    rotate_style: int = 0,
    is_nope_first: bool = False,
) -> None:
    pass


if is_rocm_rotary_embedding_enabled():

    direct_register_custom_op(
        op_name="rocm_aiter_rotary_emb_with_key_forward_hip",
        op_func=rocm_aiter_rotary_emb_with_key_forward_hip_impl,
        mutates_args=["key", "query"],
        fake_impl=rocm_aiter_rotary_emb_with_key_forward_hip_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_rotary_emb_without_key_forward_hip",
        op_func=rocm_aiter_rotary_emb_without_key_forward_hip_impl,
        mutates_args=["query"],
        fake_impl=rocm_aiter_rotary_emb_without_key_forward_hip_fake,
        dispatch_key=current_platform.dispatch_key,
    )