# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointExporter,
)
from vllm.utils.torch_utils import async_tensor_h2d


def kda_prefill_checkpoint_alignment(backend: str) -> int | None:
    return 16 if backend == "flashkda" else None


class FlashKDAPrefillCheckpointExporter(MambaPrefillCheckpointExporter):
    """Store FlashKDA recurrent and convolution checkpoint states."""


# The split export below can snapshot a row at any token of its query.
SPLIT_KDA_CHECKPOINT_ALIGNMENT = 1


@dataclass
class KDAPrefillCheckpointSplit:
    """Host plan for exporting checkpoints by splitting prefill rows.

    Each checkpointed prefill row is cut in two at its checkpoint: the head's
    final state is the checkpoint, and the tail is rerun from it. Unlike the
    GDN split, no chunk-kernel metadata is carried: the KDA chunk kernels
    derive their chunk indices from the segment cu_seqlens themselves.

    Attributes:
        seg_bounds: cu_seqlens of the non-spec rows, each checkpointed row
            cut in two.
        first_seg: first segment of each non-spec row.
        ckpt_rows: non-spec row of each checkpoint.
        tail_tokens: token indices of the tails into the non-spec tensors.
        tail_bounds: cu_seqlens of the tails.
        rows: checkpoint of each row of ``MambaPrefillCheckpointMetadata``,
            0 if none.

    """

    seg_bounds: torch.Tensor
    first_seg: torch.Tensor
    ckpt_rows: torch.Tensor
    tail_tokens: torch.Tensor
    tail_bounds: torch.Tensor
    rows: torch.Tensor


def build_kda_prefill_checkpoint_split(
    query_start_loc: list[int],
    offsets: list[int],
    num_decodes: int,
    device: torch.device,
) -> KDAPrefillCheckpointSplit | None:
    """Plan the split on the host, so the forward pass does not sync.

    Args:
        query_start_loc: cu_seqlens of the non-spec rows, decodes first.
        offsets: checkpoint offset into each row's query, 0 for none.
        num_decodes: number of leading decode rows, kept as their own
            segments because the KDA chunk path does not peel them.
        device: device of the returned tensors.

    Returns:
        The split plan, or ``None`` when no row carries a checkpoint.

    """
    assert not any(offsets[:num_decodes])
    seg_bounds: list[int] = []
    first_seg: list[int] = []
    ckpt_rows: list[int] = []
    tail_tokens: list[int] = []
    tail_bounds = [0]
    rows = [0] * len(offsets)
    for row in range(len(offsets)):
        start = query_start_loc[row]
        end = query_start_loc[row + 1]
        offset = offsets[row]
        first_seg.append(len(seg_bounds))
        seg_bounds.append(start)
        if offset:
            assert 0 < offset < end - start
            rows[row] = len(ckpt_rows)
            ckpt_rows.append(row)
            seg_bounds.append(start + offset)
            tail_tokens.extend(range(start + offset, end))
            tail_bounds.append(tail_bounds[-1] + end - start - offset)
    if not ckpt_rows:
        return None
    seg_bounds.append(query_start_loc[-1])

    def to_device(data: list[int]) -> torch.Tensor:
        return async_tensor_h2d(data, device=device, dtype=torch.int64)

    return KDAPrefillCheckpointSplit(
        seg_bounds=to_device(seg_bounds),
        first_seg=to_device(first_seg),
        ckpt_rows=to_device(ckpt_rows),
        tail_tokens=to_device(tail_tokens),
        tail_bounds=to_device(tail_bounds),
        rows=to_device(rows),
    )


def chunk_kda_with_checkpoints(
    chunk_kda: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    split: KDAPrefillCheckpointSplit,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``chunk_kda`` on the split rows of a prefill batch.

    Args:
        chunk_kda: the platform's ``chunk_kda_with_fused_gate``.
        split: plan from :func:`build_kda_prefill_checkpoint_split`.
        q: query tensor, ``[1, T, H, K]``.
        k: key tensor, ``[1, T, H, K]``.
        v: value tensor, ``[1, T, H, V]``.
        raw_g: raw gate projection, ``[1, T, H, K]``.
        beta: pre-sigmoided fp32 beta, ``[1, T, H]``.
        initial_state: per-row initial recurrent state.
        **kwargs: forwarded to ``chunk_kda`` unchanged.

    Returns:
        ``(o, final_state, checkpoint_state)``: ``o`` and ``final_state`` as
        for the unsplit rows, and one checkpoint state per checkpoint.

    """

    def run(inputs, state, cu_seqlens):
        return chunk_kda(
            q=inputs[0],
            k=inputs[1],
            v=inputs[2],
            raw_g=inputs[3],
            beta=inputs[4],
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )

    inputs = (q, k, v, raw_g, beta)
    num_segs = split.seg_bounds.numel() - 1
    split_state = initial_state.new_zeros((num_segs, *initial_state.shape[1:]))
    split_state.index_copy_(0, split.first_seg, initial_state)
    o, split_final_state = run(inputs, split_state, split.seg_bounds)
    # A checkpointed row's first segment ends at its checkpoint.
    final_state = split_final_state.index_select(0, split.first_seg)
    checkpoint_state = final_state.index_select(0, split.ckpt_rows)

    # The tails above started from a zero state; rerun them from the checkpoint.
    tails = split.tail_tokens
    tail_o, tail_final_state = run(
        tuple(x.index_select(1, tails) for x in inputs),
        checkpoint_state.to(initial_state.dtype),
        split.tail_bounds,
    )
    o.index_copy_(1, tails, tail_o)
    final_state.index_copy_(0, split.ckpt_rows, tail_final_state)
    return o, final_state, checkpoint_state
