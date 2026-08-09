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

from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.triton_utils import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    # Per-sequence chunk counts are needed as Python ints, both to build the
    # aranges below and because the caller uses `len()` of the result as a
    # Triton grid dimension. Avoiding this would mean deriving the counts from
    # a host-side cu_seqlens at the vLLM call site. Note `@tensor_cache`
    # compares args by identity, so a fresh cu_seqlens each step misses.
    with gpu_sync_allowed():
        chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    indices = torch.cat([torch.arange(n) for n in chunk_counts])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
