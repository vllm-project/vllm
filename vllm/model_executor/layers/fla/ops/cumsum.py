# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import warnings

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .utils import check_shared_mem, input_guard

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
        p_o = tl.make_block_ptr(
            o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h * T) * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * H + i_h * T) * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    else:
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
        "chunk_size must be a power of 2"
    )
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        g_org,
        g,
        cu_seqlens,
        chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
        "chunk_size must be a power of 2"
    )

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    # keep cumulative normalizer in fp32
    # this kernel is equivalent to
    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
    chunk_local_cumsum_vector_kernel[grid](
        g_org,
        g,
        cu_seqlens,
        chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g


@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    if not head_first and g.shape[1] < g.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({g.shape[1]}) < num_heads ({g.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            stacklevel=2,
        )
    if cu_seqlens is not None:
        assert g.shape[0] == 1, (
            "Only batch size 1 is supported when cu_seqlens are provided"
        )
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g, chunk_size, reverse, cu_seqlens, head_first, output_dtype
        )
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g, chunk_size, reverse, cu_seqlens, head_first, output_dtype
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}. "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )
