# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch
import torch.nn.functional as F

from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors


def compact_sampling_mask(
    processed_logits: torch.Tensor,
    top_k: np.ndarray,
) -> SamplingMaskTensors:
    num_rows, vocab_size = processed_logits.shape
    if (
        top_k.shape != (num_rows,)
        or not np.issubdtype(top_k.dtype, np.integer)
        or np.any((top_k <= 0) | (top_k > vocab_size))
    ):
        raise ValueError("top_k must contain one normalized value per logits row")

    sparse_row_indices = np.flatnonzero(top_k < vocab_size).tolist()
    packed_row_indices = np.flatnonzero(top_k == vocab_size).tolist()
    counts = torch.empty(num_rows, dtype=torch.int32, device=processed_logits.device)

    if sparse_row_indices:
        max_top_k = int(top_k[sparse_row_indices].max())
        topk_values, topk_indices = torch.topk(processed_logits, max_top_k, dim=-1)
        sparse_values = topk_values[sparse_row_indices]
        sparse_token_ids = topk_indices[sparse_row_indices].to(torch.int32)
        counts[sparse_row_indices] = torch.isfinite(sparse_values).sum(
            dim=-1, dtype=torch.int32
        )
    else:
        sparse_token_ids = torch.empty(
            (0, 0), dtype=torch.int32, device=processed_logits.device
        )

    if packed_row_indices:
        packed_logits = (
            processed_logits
            if len(packed_row_indices) == num_rows
            else processed_logits[packed_row_indices]
        )
        packed_mask, packed_counts = _pack_sampling_mask(packed_logits)
        counts[packed_row_indices] = packed_counts
    else:
        packed_mask = torch.empty(
            (0, (vocab_size + 7) // 8),
            dtype=torch.uint8,
            device=processed_logits.device,
        )

    return SamplingMaskTensors(
        sparse_token_ids=sparse_token_ids,
        sparse_row_indices=sparse_row_indices,
        packed_mask=packed_mask,
        packed_row_indices=packed_row_indices,
        counts=counts,
        vocab_size=vocab_size,
    )


def _pack_sampling_mask(
    processed_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kept_mask = torch.isfinite(processed_logits)
    counts = kept_mask.sum(dim=-1, dtype=torch.int32)
    vocab_size = processed_logits.shape[-1]
    padding = -vocab_size % 8
    if padding:
        kept_mask = F.pad(kept_mask, (0, padding))
    bit_shifts = torch.arange(8, device=processed_logits.device, dtype=torch.uint8)
    byte_mask = kept_mask.reshape(kept_mask.shape[0], -1, 8).to(torch.uint8)
    byte_mask.bitwise_left_shift_(bit_shifts)
    packed_mask = byte_mask.sum(dim=-1, dtype=torch.uint8)
    return packed_mask, counts
