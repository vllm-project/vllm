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
    """Per row: write the finite-logit token ids in ascending order into
    ``token_ids[row, :count]`` and the same support as a bitmask into
    ``packed_mask[row]``. The bitmask is only read back on the host for
    rows whose support overflows ``max_num_kept`` (top-k ties)."""
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
        keep_i32 = keep.to(tl.int32)
        pos = count + tl.cumsum(keep_i32, axis=0) - keep_i32
        tl.store(
            token_ids_ptr + req_idx * token_ids_row_stride + pos,
            offsets.to(tl.int32),
            mask=keep & (pos < max_num_kept),
        )
        count += tl.sum(keep_i32)

        keep_bits = tl.reshape(keep_i32, (BLOCK_SIZE // 8, 8))
        bit_shifts = tl.arange(0, 8)[None, :]
        packed = tl.sum(keep_bits << bit_shifts, axis=1).to(tl.uint8)
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
            packed,
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + req_idx, count)


# Widest compact row the sampler allocates. Requests whose support is larger
# (a top_k near the vocab size, or top-k ties) fall back to the exact bitmask,
# so this only bounds GPU/host memory: [max_num_reqs, cap] int32.
MAX_COMPACT_SUPPORT = 2048


class SamplingMaskTensors(NamedTuple):
    """Device-side sampling mask data pending async D2H.

    The support is compacted on the GPU so the host normally touches only
    ``num_reqs * max_num_kept`` ids. The bit-packed mask is kept as the
    exact fallback for rows whose support exceeds ``max_num_kept``.
    """

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
        """Capture the finite-logit support for requests that sampled tokens.

        Args:
            logits: Post-processed logits, ``-inf`` outside the support.
            num_sampled_tokens: Per-request sampled token count; rows with
                zero are skipped.
            max_num_kept: Expected upper bound on the support size, normally
                the largest ``top_k`` in the batch; clamped to
                ``MAX_COMPACT_SUPPORT``. Rows exceeding it still round-trip
                exactly through the bitmask.
        """
        num_reqs, vocab_size = logits.shape
        max_num_kept = max(1, min(max_num_kept, vocab_size, MAX_COMPACT_SUPPORT))
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

    def tolists(self, num_sampled_tokens: np.ndarray) -> SamplingMaskLists:
        """Convert the captured masks to the scheduler's CSR representation."""
        sampled_rows = np.flatnonzero(num_sampled_tokens)
        max_num_kept = self.token_ids.shape[1]
        counts = self.counts.cpu().numpy()[sampled_rows]
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, dtype=np.int64, out=offsets[1:])

        overflow = np.flatnonzero(counts > max_num_kept)
        if len(overflow) == 0:
            token_ids = self.token_ids.cpu().numpy()[sampled_rows]
            kept = np.arange(max_num_kept)[None, :] < counts[:, None]
            flat_token_ids = token_ids[kept]
        else:
            # Rare: rebuild the overflowing rows from the exact bitmask.
            flat_token_ids = np.empty(offsets[-1], dtype=np.int32)
            token_ids = self.token_ids.cpu().numpy()[sampled_rows]
            packed_mask = self.packed_mask.cpu().numpy()[sampled_rows]
            for i, count in enumerate(counts):
                if count > max_num_kept:
                    bits = np.unpackbits(
                        packed_mask[i], count=self.vocab_size, bitorder="little"
                    )
                    row = np.flatnonzero(bits).astype(np.int32, copy=False)
                else:
                    row = token_ids[i, :count]
                flat_token_ids[offsets[i] : offsets[i + 1]] = row

        return SamplingMaskLists(
            token_ids=flat_token_ids,
            offsets=offsets,
            cu_num_generated_tokens=np.cumsum(
                np.concatenate(([0], num_sampled_tokens))
            ).tolist(),
        )
