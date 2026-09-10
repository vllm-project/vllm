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
    # Inferring seq_id from local-index wraparound (eq(0).cumsum) is wrong when
    # a sequence has zero chunks: the next sequence's first chunk is attributed
    # to the empty slot. Assign seq_id from the sequence index instead.
    # https://github.com/vllm-project/vllm/pull/51540
    with gpu_sync_allowed():
        chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    if not any(chunk_counts):
        return torch.empty(0, 2, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    seq_ids = torch.repeat_interleave(
        torch.arange(len(chunk_counts), dtype=torch.long),
        torch.as_tensor(chunk_counts, dtype=torch.long),
    )
    loc_ids = torch.cat(
        [torch.arange(n, dtype=torch.long) for n in chunk_counts if n]
    )
    return torch.stack([seq_ids, loc_ids], 1).to(
        device=cu_seqlens.device, dtype=cu_seqlens.dtype, non_blocking=True
    )


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_zeros(1), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
