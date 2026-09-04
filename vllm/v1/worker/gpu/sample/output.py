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
    cu_num_logits_ptr,
    num_sampled_tokens_ptr,
    token_ids_ptr,
    token_ids_row_stride,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    max_num_kept,
    ROWS_PER_REQUEST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per row: first ``max_num_kept`` finite-logit ids, the full count, the bitmask."""
    output_row = tl.program_id(0)
    req_idx = output_row // ROWS_PER_REQUEST
    slot_idx = output_row % ROWS_PER_REQUEST
    source_row = tl.load(cu_num_logits_ptr + req_idx) + slot_idx
    request_end = tl.load(cu_num_logits_ptr + req_idx + 1)
    is_active = (slot_idx < tl.load(num_sampled_tokens_ptr + req_idx)) & (
        source_row < request_end
    )
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            logits_ptr + source_row * logits_row_stride + offsets * logits_col_stride,
            mask=(offsets < vocab_size) & is_active,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        keep_i32 = keep.to(tl.int32)
        pos = count + tl.cumsum(keep_i32, axis=0) - keep_i32
        tl.store(
            token_ids_ptr + output_row * token_ids_row_stride + pos,
            offsets.to(tl.int32),
            mask=keep & (pos < max_num_kept),
        )
        count += tl.sum(keep_i32)

        bits = tl.reshape(keep_i32, (BLOCK_SIZE // 8, 8)) << tl.arange(0, 8)[None, :]
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + output_row * packed_mask_row_stride + byte_offsets,
            tl.sum(bits, axis=1).to(tl.uint8),
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + output_row, count)


# Bounds the [num_reqs, width] int32 buffer; wider rows use the bitmask instead.
MAX_COMPACT_SUPPORT = 2048


class SamplingMaskTensors(NamedTuple):
    """Device-side masks pending async D2H: compact ids, plus the bitmask as
    the exact fallback for rows wider than ``max_num_kept``."""

    # [num_requests * rows_per_request, max_num_kept]
    token_ids: torch.Tensor
    # [num_requests * rows_per_request, ceil(vocab_size / 8)]
    packed_mask: torch.Tensor
    # [num_requests * rows_per_request]
    counts: torch.Tensor
    vocab_size: int
    # Fixed request-major output slots. Ordinary sampling has one slot.
    rows_per_request: int = 1

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_sampled_tokens: torch.Tensor,
        max_num_kept: int,
        rows_per_request: int = 1,
    ) -> SamplingMaskTensors:
        """Capture committed target supports in fixed request-major slots."""
        num_reqs = num_sampled_tokens.shape[0]
        vocab_size = logits.shape[1]
        max_num_kept = min(max_num_kept, vocab_size, MAX_COMPACT_SUPPORT)
        num_output_rows = num_reqs * rows_per_request
        device = logits.device

        token_ids = torch.empty(
            (num_output_rows, max_num_kept), dtype=torch.int32, device=device
        )
        packed_mask = torch.empty(
            (num_output_rows, (vocab_size + 7) // 8),
            dtype=torch.uint8,
            device=device,
        )
        counts = torch.empty(num_output_rows, dtype=torch.int32, device=device)
        if num_output_rows:
            _compact_sampling_mask_kernel[(num_output_rows,)](
                logits,
                logits.stride(0),
                logits.stride(1),
                cu_num_logits,
                num_sampled_tokens,
                token_ids,
                token_ids.stride(0),
                packed_mask,
                packed_mask.stride(0),
                counts,
                vocab_size,
                max_num_kept,
                ROWS_PER_REQUEST=rows_per_request,
                BLOCK_SIZE=8192,
            )
        return cls(token_ids, packed_mask, counts, vocab_size, rows_per_request)

    def to_cpu_nonblocking(self) -> SamplingMaskTensors:
        if self.token_ids.device.type == "cpu":
            return self
        return SamplingMaskTensors(
            self.token_ids.to("cpu", non_blocking=True),
            self.packed_mask.to("cpu", non_blocking=True),
            self.counts.to("cpu", non_blocking=True),
            self.vocab_size,
            self.rows_per_request,
        )

    def tolists(
        self, num_sampled_tokens: np.ndarray | None = None
    ) -> SamplingMaskLists:
        """Convert fixed output rows to the scheduler's CSR representation."""
        counts = self.counts.cpu().numpy()
        token_ids = self.token_ids.cpu().numpy()
        packed_mask = self.packed_mask.cpu().numpy()
        width = token_ids.shape[1]

        cu_num_generated_tokens = None
        if self.rows_per_request == 1:
            sampled_rows = np.arange(len(counts))
        else:
            assert num_sampled_tokens is not None
            num_sampled_tokens = np.asarray(num_sampled_tokens)
            active_slots = (
                np.arange(self.rows_per_request)[None, :]
                < (num_sampled_tokens[:, None])
            )
            sampled_rows = np.flatnonzero(active_slots)
            cu_num_generated_tokens = np.cumsum(
                np.concatenate(([0], num_sampled_tokens))
            ).tolist()

        def support(row: int) -> np.ndarray:
            if counts[row] <= width:
                return token_ids[row, : counts[row]]
            # Wider than the compact row (ties or a huge top_k): use the bitmask.
            bits = np.unpackbits(
                packed_mask[row], count=self.vocab_size, bitorder="little"
            )
            return np.flatnonzero(bits).astype(np.int32, copy=False)

        supports = [support(row) for row in sampled_rows]
        offsets = np.zeros(len(supports) + 1, dtype=np.int64)
        np.cumsum([len(s) for s in supports], out=offsets[1:])
        flattened = (
            np.concatenate(supports) if supports else np.empty(0, dtype=np.int32)
        )
        return SamplingMaskLists(flattened, offsets, cu_num_generated_tokens)
