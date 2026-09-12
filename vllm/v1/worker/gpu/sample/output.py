# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors, SamplingMaskLists


@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor | None
    num_rejected: torch.Tensor | None = None
    sampling_mask_tensors: SamplingMaskTensors | None = None


@triton.jit
def _compact_sampling_mask_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    num_sampled_tokens_ptr,
    token_ids_ptr,
    token_ids_row_stride,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    max_num_kept,
    BLOCK_SIZE: tl.constexpr,
):
    """Per row: first ``max_num_kept`` finite-logit ids, the full count, the bitmask."""
    req_idx = tl.program_id(0)
    is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
            mask=offsets < vocab_size,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        keep_i32 = keep.to(tl.int32)
        pos = count + tl.cumsum(keep_i32, axis=0) - keep_i32
        tl.store(
            token_ids_ptr + req_idx * token_ids_row_stride + pos,
            offsets.to(tl.int32),
            mask=keep & (pos < max_num_kept),
        )
        count += tl.sum(keep_i32)

        bits = tl.reshape(keep_i32, (BLOCK_SIZE // 8, 8)) << tl.arange(0, 8)[None, :]
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
            tl.sum(bits, axis=1).to(tl.uint8),
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + req_idx, count)


# Bounds the [num_reqs, width] int32 buffer; wider rows use the bitmask instead.
MAX_COMPACT_SUPPORT = 2048


class SamplingMaskTensors(NamedTuple):
    """Device-side masks pending async D2H: compact ids, plus the bitmask as
    the exact fallback for rows wider than ``max_num_kept``."""

    # [num_requests, max_num_kept]
    token_ids: torch.Tensor
    # [num_requests, ceil(vocab_size / 8)]
    packed_mask: torch.Tensor
    # [num_requests]
    counts: torch.Tensor
    vocab_size: int

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        num_sampled_tokens: torch.Tensor,
        max_num_kept: int,
    ) -> SamplingMaskTensors:
        """Capture the finite-logit support of every row with a sampled token."""
        num_reqs, vocab_size = logits.shape
        max_num_kept = min(max_num_kept, vocab_size, MAX_COMPACT_SUPPORT)
        device = logits.device

        token_ids = torch.empty(
            (num_reqs, max_num_kept), dtype=torch.int32, device=device
        )
        packed_mask = torch.empty(
            (num_reqs, (vocab_size + 7) // 8), dtype=torch.uint8, device=device
        )
        counts = torch.empty(num_reqs, dtype=torch.int32, device=device)
        _compact_sampling_mask_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            logits.stride(1),
            num_sampled_tokens,
            token_ids,
            token_ids.stride(0),
            packed_mask,
            packed_mask.stride(0),
            counts,
            vocab_size,
            max_num_kept,
            BLOCK_SIZE=8192,
        )
        return cls(token_ids, packed_mask, counts, vocab_size)

    def to_cpu_nonblocking(self) -> SamplingMaskTensors:
        if self.token_ids.device.type == "cpu":
            return self
        return SamplingMaskTensors(
            self.token_ids.to("cpu", non_blocking=True),
            self.packed_mask.to("cpu", non_blocking=True),
            self.counts.to("cpu", non_blocking=True),
            self.vocab_size,
        )

    def tolists(self) -> SamplingMaskLists:
        """CSR over all requests; rows without a sampled token are empty."""
        counts = self.counts.cpu().numpy()
        token_ids = self.token_ids.cpu().numpy()
        packed_mask = self.packed_mask.cpu().numpy()
        width = token_ids.shape[1]

        def support(row: int) -> np.ndarray:
            if counts[row] <= width:
                return token_ids[row, : counts[row]]
            # Wider than the compact row (ties or a huge top_k): use the bitmask.
            bits = np.unpackbits(
                packed_mask[row], count=self.vocab_size, bitorder="little"
            )
            return np.flatnonzero(bits).astype(np.int32, copy=False)

        supports = [support(row) for row in range(len(counts))]
        offsets = np.zeros(len(supports) + 1, dtype=np.int64)
        np.cumsum([len(s) for s in supports], out=offsets[1:])
        return SamplingMaskLists(np.concatenate(supports), offsets)
