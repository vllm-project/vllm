# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

from vllm.triton_utils import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(
    cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None = None
) -> torch.Tensor:
    if cu_seqlens_cpu is not None:
        return cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(
                prepare_lens(cu_seqlens, cu_seqlens_cpu), chunk_size
            ).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(
        cu_seqlens, non_blocking=True
    )


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_zeros(1), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
