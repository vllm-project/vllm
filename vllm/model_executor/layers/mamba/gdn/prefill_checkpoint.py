# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Internal prefill checkpoints for GDN chunk kernels that only return final
states (Triton/FLA, AITER FlyDSL).

Each checkpointed prefill row is cut in two at its checkpoint: the head's final
state is the checkpoint, and the tail, shorter than one block, is rerun from it.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.utils.torch_utils import async_tensor_h2d

# A row can be cut at any token.
GDN_SPLIT_CHECKPOINT_ALIGNMENT = 1

# (chunk_indices, chunk_offsets, aiter_prefill_metadata) of one cu_seqlens.
ChunkKernelMetadata = tuple[torch.Tensor | None, torch.Tensor | None, object | None]


@dataclass
class GDNPrefillCheckpointSplit:
    # cu_seqlens of the prefill rows, with each checkpointed row cut in two.
    seg_bounds: torch.Tensor
    seg_kernel_metadata: ChunkKernelMetadata
    # First segment of each prefill row.
    first_seg: torch.Tensor
    # Prefill row of each checkpoint.
    ckpt_rows: torch.Tensor
    # Prefill-relative token indices of the tails, and their cu_seqlens.
    tail_tokens: torch.Tensor
    tail_bounds: torch.Tensor
    tail_kernel_metadata: ChunkKernelMetadata
    # Checkpoint of each row of MambaPrefillCheckpointMetadata, 0 if none.
    rows: torch.Tensor


def build_gdn_prefill_checkpoint_split(
    query_start_loc: list[int],
    offsets: list[int],
    num_decodes: int,
    device: torch.device,
    kernel_metadata: Callable[[torch.Tensor, torch.Tensor], ChunkKernelMetadata],
) -> GDNPrefillCheckpointSplit | None:
    """Plan the split on the host, so the forward pass does not sync.

    Args:
        query_start_loc: cu_seqlens of the non-spec rows.
        offsets: checkpoint offset into each row's query, 0 for none.
        num_decodes: number of leading decode rows.
        device: device of the returned tensors.
        kernel_metadata: chunk-kernel metadata of (cu_seqlens, cu_seqlens_cpu).

    """
    assert not any(offsets[:num_decodes])
    base = query_start_loc[num_decodes]
    seg_bounds: list[int] = []
    first_seg: list[int] = []
    ckpt_rows: list[int] = []
    tail_tokens: list[int] = []
    tail_bounds = [0]
    rows = [0] * len(offsets)
    for prefill_row, row in enumerate(range(num_decodes, len(offsets))):
        start = query_start_loc[row] - base
        end = query_start_loc[row + 1] - base
        offset = offsets[row]
        first_seg.append(len(seg_bounds))
        seg_bounds.append(start)
        if offset:
            assert 0 < offset < end - start
            rows[row] = len(ckpt_rows)
            ckpt_rows.append(prefill_row)
            seg_bounds.append(start + offset)
            tail_tokens.extend(range(start + offset, end))
            tail_bounds.append(tail_bounds[-1] + end - start - offset)
    if not ckpt_rows:
        return None
    seg_bounds.append(query_start_loc[-1] - base)

    def to_device(data: list[int]) -> torch.Tensor:
        return async_tensor_h2d(data, device=device, dtype=torch.int64)

    def with_metadata(bounds: list[int]) -> tuple[torch.Tensor, ChunkKernelMetadata]:
        bounds_cpu = torch.tensor(bounds, dtype=torch.int32)
        bounds_gpu = async_tensor_h2d(bounds_cpu, device=device)
        return bounds_gpu, kernel_metadata(bounds_gpu, bounds_cpu)

    seg_bounds_gpu, seg_kernel_metadata = with_metadata(seg_bounds)
    tail_bounds_gpu, tail_kernel_metadata = with_metadata(tail_bounds)
    return GDNPrefillCheckpointSplit(
        seg_bounds=seg_bounds_gpu,
        seg_kernel_metadata=seg_kernel_metadata,
        first_seg=to_device(first_seg),
        ckpt_rows=to_device(ckpt_rows),
        tail_tokens=to_device(tail_tokens),
        tail_bounds=tail_bounds_gpu,
        tail_kernel_metadata=tail_kernel_metadata,
        rows=to_device(rows),
    )


def chunk_gated_delta_rule_with_checkpoints(
    chunk_gated_delta_rule: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    split: GDNPrefillCheckpointSplit,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``chunk_gated_delta_rule`` on the split rows.

    Returns ``(o, final_state, checkpoint_state)``: ``o`` and ``final_state`` as
    for the unsplit rows, and one checkpoint state per checkpoint.
    """

    def run(inputs, state, cu_seqlens, kernel_metadata):
        chunk_indices, chunk_offsets, aiter_prefill_metadata = kernel_metadata
        q, k, v, g, beta = inputs
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            aiter_prefill_metadata=aiter_prefill_metadata,
        )

    inputs = (q, k, v, g, beta)
    num_segs = split.seg_bounds.numel() - 1
    split_state = initial_state.new_zeros((num_segs, *initial_state.shape[1:]))
    split_state.index_copy_(0, split.first_seg, initial_state)
    o, split_final_state = run(
        inputs, split_state, split.seg_bounds, split.seg_kernel_metadata
    )
    # A checkpointed row's first segment ends at its checkpoint.
    final_state = split_final_state.index_select(0, split.first_seg)
    checkpoint_state = final_state.index_select(0, split.ckpt_rows)

    # The tails above started from a zero state; rerun them from the checkpoint.
    tails = split.tail_tokens
    tail_o, tail_final_state = run(
        tuple(x.index_select(1, tails) for x in inputs),
        checkpoint_state.to(initial_state.dtype),
        split.tail_bounds,
        split.tail_kernel_metadata,
    )
    o.index_copy_(1, tails, tail_o)
    final_state.index_copy_(0, split.ckpt_rows, tail_final_state)
    return o, final_state, checkpoint_state
