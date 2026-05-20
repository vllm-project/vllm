# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Etha chunk abstraction — the unit of weight transfer.

A `Chunk` is one piece of work for one rank: "I am sending this slice of
this tensor to that rank", or "I am receiving into this slice from that
rank", or "I am locally copying from this slice to that slice." It is
deliberately transport-agnostic — Chunks know nothing about NCCL,
PyNcclCommunicator, NIXL, or any wire format. A transport module
(e.g. `etha_nccl_transport.chunk_comm`) consumes a list of Chunks and
moves the bytes.

`map_to_chunk_ops` is the bridge between the M-to-N planner output
(an abstract `m2m_map: src_rank → src_chunk_idx → [(dst_rank, dst_chunk_idx)]`,
keyed in chunk-index space) and concrete Chunks for one rank, by
specializing the abstract indices to actual tensor slices.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class Chunk:
    """A planned send / recv / self-copy on the local tensor.

    `is_source` is True for sends and self-copies (this rank is producing
    bytes); False for recvs.
    """

    is_source: bool
    src_rank: int
    dst_rank: int
    slice_tuples: tuple[slice, ...]
    tensor: torch.Tensor
    transfer_dtype: torch.dtype
    # For self-copy only: the slice of `tensor` to read from.
    src_slice_tuples: tuple[slice, ...] | None = None

    @property
    def is_self_copy(self) -> bool:
        return self.src_slice_tuples is not None


def _slicer_tuples(
    tensor_shape: tuple[int, ...], num_slicers: Iterable[int]
) -> list[tuple[slice, ...]]:
    """Slice tuples for chunking a tensor into num_slicers pieces per dim."""
    per_dim: list[list[slice]] = []
    for d, n in enumerate(num_slicers):
        size = tensor_shape[d] // n
        per_dim.append([slice(i * size, (i + 1) * size) for i in range(n)])
    return list(itertools.product(*per_dim))


def _idx_to_linear(idx: tuple[int, ...], shape: list[int] | tuple[int, ...]) -> int:
    """Row-major flatten of a multi-dim index."""
    flat = 0
    for c, s in zip(idx, shape):
        flat = flat * s + c
    return flat


def map_to_chunk_ops(
    m2m_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    rank: int,
    src_num_slicers: list[int],
    tgt_num_slicers: list[int],
    src_tensor: torch.Tensor | None,
    tgt_tensor: torch.Tensor | None,
    transfer_dtype: torch.dtype,
    role: Literal["src", "tgt"],
) -> list[Chunk]:
    """Specialize an abstract M2M map into per-rank send/recv chunks.

    The `m2m_map` is keyed in chunk-index space (in units of
    src_num_slicers / tgt_num_slicers), so it is shape-independent.
    This function applies it to a specific tensor by computing the
    actual slice ranges from the tensor's shape.

    Fan-out broadcasts (one source chunk going to multiple destinations)
    are emitted as N separate P2P sends. This is the MVP shortcut.
    """
    def _build_slicers(
        tensor: torch.Tensor | None, num_slicers: list[int]
    ) -> list[tuple[slice, ...]] | None:
        if tensor is None:
            return None
        pad = [1] * (len(tensor.shape) - len(num_slicers))
        return _slicer_tuples(tuple(tensor.shape), num_slicers + pad)

    src_slicers = _build_slicers(src_tensor, src_num_slicers)
    tgt_slicers = _build_slicers(tgt_tensor, tgt_num_slicers)

    chunks: list[Chunk] = []
    for src_rank, sub in m2m_map.items():
        for src_idx, dst_list in sub.items():
            for dst_rank, dst_idx in dst_list:
                if src_rank == rank and dst_rank == rank:
                    # Self-copy: only meaningful when this rank holds both
                    # ends of the transfer (rare; possible if source and
                    # target meshes overlap on a rank).
                    if (
                        tgt_tensor is None
                        or src_tensor is None
                        or tgt_slicers is None
                        or src_slicers is None
                    ):
                        # Caller has only one side of the tensor on this rank;
                        # nothing to do here from their perspective.
                        continue
                    chunks.append(
                        Chunk(
                            is_source=True,
                            src_rank=src_rank,
                            dst_rank=dst_rank,
                            slice_tuples=tgt_slicers[
                                _idx_to_linear(dst_idx, tgt_num_slicers)
                            ],
                            src_slice_tuples=src_slicers[
                                _idx_to_linear(src_idx, src_num_slicers)
                            ],
                            tensor=tgt_tensor,
                            transfer_dtype=transfer_dtype,
                        )
                    )
                elif src_rank == rank and role == "src":
                    assert src_tensor is not None and src_slicers is not None
                    chunks.append(
                        Chunk(
                            is_source=True,
                            src_rank=src_rank,
                            dst_rank=dst_rank,
                            slice_tuples=src_slicers[
                                _idx_to_linear(src_idx, src_num_slicers)
                            ],
                            tensor=src_tensor,
                            transfer_dtype=transfer_dtype,
                        )
                    )
                elif dst_rank == rank and role == "tgt":
                    assert tgt_tensor is not None and tgt_slicers is not None
                    chunks.append(
                        Chunk(
                            is_source=False,
                            src_rank=src_rank,
                            dst_rank=dst_rank,
                            slice_tuples=tgt_slicers[
                                _idx_to_linear(dst_idx, tgt_num_slicers)
                            ],
                            tensor=tgt_tensor,
                            transfer_dtype=transfer_dtype,
                        )
                    )
    return chunks
