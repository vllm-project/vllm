# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass

import torch


@dataclass
class Mamba2Metadata:
    chunk_size: int
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor


def prepare_mamba2_metadata(seq_idx: torch.Tensor,
                            chunk_size: int) -> Mamba2Metadata:
    # convert seq_idx to chunk indices and offsets
    # - derive the cu_seqlens
    _, cu_seqlens = torch.where(seq_idx.diff())
    cu_seqlens += 1

    # outputs will have length expansion of chunks that do not divide
    # chunk_size
    N = math.ceil(seq_idx.shape[-1] / chunk_size) + (cu_seqlens % chunk_size
                                                     > 0).sum()
    chunk_indices = torch.arange(N, dtype=torch.int, device=seq_idx.device)
    chunk_offsets = torch.zeros((N, ), dtype=torch.int, device=seq_idx.device)

    cu_seqlens = cu_seqlens.tolist() + [seq_idx.shape[-1]]
    p = 0  # num of insertions
    for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):

        # if does not divide chunk_size, then there is one chunk insertion
        p += (s % chunk_size > 0)

        # get the dimensions
        # - the + 1 for _e is to shift the boundary by one chunk
        # - this shifting is not needed if chunk_size divides e
        _s, _e = s // chunk_size + p, e // chunk_size + p + (e % chunk_size
                                                             > 0)

        # adjust inidces and offsets
        chunk_indices[_s:_e] -= p
        chunk_offsets[_s] = s % chunk_size

    return Mamba2Metadata(chunk_size=chunk_size,
                          chunk_indices=chunk_indices,
                          chunk_offsets=chunk_offsets)
