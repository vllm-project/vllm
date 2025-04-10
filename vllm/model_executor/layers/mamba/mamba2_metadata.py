# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass

import torch


@dataclass
class Mamba2Metadata:
    chunk_size: int
    seq_idx: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor


def _seq_idx_to_chunk_indices_offsets(seq_idx, chunk_size: int):

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

    return chunk_indices, chunk_offsets


def prepare_mamba2_metadata(
    chunk_size: int,
    has_prefills: bool,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> Mamba2Metadata:

    seq_idx = None
    chunk_indices, chunk_offsets = None, None
    if has_prefills:
        seq_idx = torch.zeros_like(input_ids, dtype=torch.int32)
        for i, (srt,
                end) in enumerate(zip(
                    query_start_loc,
                    query_start_loc[1:],
                )):
            seq_idx[srt:end] = i
        seq_idx.unsqueeze_(0)

        # compute metadata for chunked prefill.
        # actually this is only needed if there are initial states,
        # but this is determinable only from attention metadata yet
        # unavailable from the top-level model forward. Rather than
        # complicating things to extract said metadata, we simply just
        # compute them once at the top level model forward and reuse
        # them in mamba layers. If not needed, they will be ignored
        # inside mamba kernels.
        chunk_indices, chunk_offsets = _seq_idx_to_chunk_indices_offsets(
            seq_idx, chunk_size)

    return Mamba2Metadata(chunk_size=chunk_size,
                          seq_idx=seq_idx,
                          chunk_indices=chunk_indices,
                          chunk_offsets=chunk_offsets)
