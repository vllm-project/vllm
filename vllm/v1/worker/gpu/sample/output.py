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
def _pack_sampling_mask_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    num_sampled_tokens_ptr,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        valid = offsets < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
            mask=valid,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        count += tl.sum(keep).to(tl.int32)

        keep = tl.reshape(keep.to(tl.int32), (BLOCK_SIZE // 8, 8))
        bit_shifts = tl.arange(0, 8)[None, :]
        packed = tl.sum(keep << bit_shifts, axis=1).to(tl.uint8)
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
            packed,
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + req_idx, count)


class SamplingMaskTensors(NamedTuple):
    """Bit-packed device-side sampling mask data pending async D2H."""

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
    ) -> SamplingMaskTensors:
        """Pack the finite-logit support for requests that sampled tokens."""
        num_reqs, vocab_size = logits.shape
        packed_width = (vocab_size + 7) // 8

        packed_mask = torch.empty(
            (num_reqs, packed_width), dtype=torch.uint8, device=logits.device
        )
        counts = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
        _pack_sampling_mask_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            logits.stride(1),
            num_sampled_tokens,
            packed_mask,
            packed_mask.stride(0),
            counts,
            vocab_size,
            BLOCK_SIZE=8192,
        )

        return cls(packed_mask, counts, vocab_size)

    def to_cpu_nonblocking(self) -> SamplingMaskTensors:
        if self.packed_mask.device.type == "cpu":
            return self
        return SamplingMaskTensors(
            self.packed_mask.to("cpu", non_blocking=True),
            self.counts.to("cpu", non_blocking=True),
            self.vocab_size,
        )

    def tolists(self, num_sampled_tokens: np.ndarray) -> SamplingMaskLists:
        """Convert the packed masks to the scheduler's CSR representation."""
        sampled_rows = np.flatnonzero(num_sampled_tokens)
        counts = self.counts.cpu().numpy()[sampled_rows]
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, dtype=np.int64, out=offsets[1:])
        unpacked = np.unpackbits(
            self.packed_mask.cpu().numpy()[sampled_rows],
            axis=1,
            count=self.vocab_size,
            bitorder="little",
        )
        token_ids = np.nonzero(unpacked)[1].astype(np.int32, copy=False)
        return SamplingMaskLists(
            token_ids=token_ids,
            offsets=offsets,
            cu_num_generated_tokens=np.cumsum(
                np.concatenate(([0], num_sampled_tokens))
            ).tolist(),
        )
