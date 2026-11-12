# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    sampling_mask_tensors: SamplingMaskTensors | SamplingSupportTensors | None = None


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
    """Per row: first `max_num_kept` finite-logit ids, the full count, the bitmask."""
    output_row = tl.program_id(0)
    req_idx = output_row // ROWS_PER_REQUEST
    slot_idx = output_row % ROWS_PER_REQUEST
    if slot_idx >= tl.load(num_sampled_tokens_ptr + req_idx):
        tl.store(counts_ptr + output_row, 0)
        return
    source_row = tl.load(cu_num_logits_ptr + req_idx) + slot_idx

    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            logits_ptr + source_row * logits_row_stride + offsets * logits_col_stride,
            mask=offsets < vocab_size,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf"))
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
    the exact fallback for rows wider than `max_num_kept`."""

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

    @classmethod
    def cat(cls, chunks: Sequence[SamplingMaskTensors]) -> SamplingMaskTensors:
        """Join request-contiguous chunks that share one output layout."""
        first = chunks[0]
        assert len({(c.vocab_size, c.rows_per_request) for c in chunks}) == 1
        return cls(
            torch.cat([chunk.token_ids for chunk in chunks]),
            torch.cat([chunk.packed_mask for chunk in chunks]),
            torch.cat([chunk.counts for chunk in chunks]),
            first.vocab_size,
            first.rows_per_request,
        )

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
            active_slots = (
                np.arange(self.rows_per_request)[None, :] < num_sampled_tokens[:, None]
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
        flat = np.concatenate([np.empty(0, dtype=np.int32), *supports])
        return SamplingMaskLists(flat, offsets, cu_num_generated_tokens)


@triton.jit
def _support_stats_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    cu_num_logits_ptr,
    num_sampled_tokens_ptr,
    counts_ptr,
    logsumexp_ptr,
    vocab_size,
    ROWS_PER_REQUEST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per row: the number of finite logits and their logsumexp."""
    output_row = tl.program_id(0)
    req_idx = output_row // ROWS_PER_REQUEST
    slot_idx = output_row % ROWS_PER_REQUEST
    if slot_idx >= tl.load(num_sampled_tokens_ptr + req_idx):
        tl.store(counts_ptr + output_row, 0)
        return
    source_row = (tl.load(cu_num_logits_ptr + req_idx) + slot_idx).to(tl.int64)

    count = tl.zeros((), dtype=tl.int32)
    running_max = tl.full((), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((), dtype=tl.float32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            logits_ptr + source_row * logits_row_stride + offsets * logits_col_stride,
            mask=offsets < vocab_size,
            other=-float("inf"),
        ).to(tl.float32)
        keep = (logits > -float("inf")) & (logits < float("inf"))
        count += tl.sum(keep.to(tl.int32))

        new_max = tl.maximum(running_max, tl.max(tl.where(keep, logits, -float("inf"))))
        # A block without finite logits leaves the max at -inf; shift by 0 then.
        shift = tl.where(new_max > -float("inf"), new_max, 0.0)
        running_sum = running_sum * tl.exp(running_max - shift) + tl.sum(
            tl.where(keep, tl.exp(logits - shift), 0.0)
        )
        running_max = new_max

    tl.store(counts_ptr + output_row, count)
    tl.store(logsumexp_ptr + output_row, running_max + tl.log(running_sum))


@triton.jit
def _support_compact_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    cu_num_logits_ptr,
    counts_ptr,
    starts_ptr,
    logsumexp_ptr,
    token_ids_ptr,
    logprobs_ptr,
    vocab_size,
    ROWS_PER_REQUEST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per row: its finite-logit ids and their logprobs, written at `starts`."""
    output_row = tl.program_id(0)
    if tl.load(counts_ptr + output_row) == 0:
        return
    req_idx = output_row // ROWS_PER_REQUEST
    slot_idx = output_row % ROWS_PER_REQUEST
    source_row = (tl.load(cu_num_logits_ptr + req_idx) + slot_idx).to(tl.int64)
    pos = tl.load(starts_ptr + output_row)
    logsumexp = tl.load(logsumexp_ptr + output_row)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            logits_ptr + source_row * logits_row_stride + offsets * logits_col_stride,
            mask=offsets < vocab_size,
            other=-float("inf"),
        ).to(tl.float32)
        keep = (logits > -float("inf")) & (logits < float("inf"))
        keep_i64 = keep.to(tl.int64)
        slots = pos + tl.cumsum(keep_i64, axis=0) - keep_i64
        tl.store(token_ids_ptr + slots, offsets.to(tl.int32), mask=keep)
        tl.store(logprobs_ptr + slots, logits - logsumexp, mask=keep)
        pos += tl.sum(keep_i64)


@dataclass
class SamplingSupportTensors:
    """Device-side supports with their logprobs, pending D2H in two phases.

    Rows are packed back to back in flat buffers, so the host copy is the
    step's real total instead of a padded layout. That total is only known on
    the device: the counts ride on the step's async copy, and `tolists` copies
    exactly the filled prefix once they have arrived.
    """

    # Per chunk, flat [capacity] buffers; row i starts at sum(counts[:i]).
    token_ids: list[torch.Tensor]
    logprobs: list[torch.Tensor]
    # Per chunk, [num_requests * rows_per_request].
    counts: list[torch.Tensor]
    # Fixed request-major output slots. Ordinary sampling has one slot.
    rows_per_request: int = 1

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_sampled_tokens: torch.Tensor,
        rows_per_request: int = 1,
    ) -> SamplingSupportTensors:
        """Capture committed target supports, normalized over each support."""
        num_output_rows = num_sampled_tokens.shape[0] * rows_per_request
        num_logits, vocab_size = logits.shape
        device = logits.device

        counts = torch.empty(num_output_rows, dtype=torch.int32, device=device)
        logsumexp = torch.empty(num_output_rows, dtype=torch.float32, device=device)
        _support_stats_kernel[(num_output_rows,)](
            logits,
            logits.stride(0),
            logits.stride(1),
            cu_num_logits,
            num_sampled_tokens,
            counts,
            logsumexp,
            vocab_size,
            ROWS_PER_REQUEST=rows_per_request,
            BLOCK_SIZE=8192,
        )
        starts = counts.cumsum(0, dtype=torch.int64) - counts

        # Committed rows read distinct logits rows, so this cannot overflow.
        # The buffers stay on the device; only their filled prefix is copied.
        capacity = num_logits * vocab_size
        token_ids = torch.empty(capacity, dtype=torch.int32, device=device)
        logprobs = torch.empty(capacity, dtype=torch.float32, device=device)
        _support_compact_kernel[(num_output_rows,)](
            logits,
            logits.stride(0),
            logits.stride(1),
            cu_num_logits,
            counts,
            starts,
            logsumexp,
            token_ids,
            logprobs,
            vocab_size,
            ROWS_PER_REQUEST=rows_per_request,
            BLOCK_SIZE=8192,
        )
        return cls([token_ids], [logprobs], [counts], rows_per_request)

    @classmethod
    def cat(cls, chunks: Sequence[SamplingSupportTensors]) -> SamplingSupportTensors:
        """Join request-contiguous chunks that share one output layout."""
        first = chunks[0]
        assert len({chunk.rows_per_request for chunk in chunks}) == 1
        return cls(
            [buffer for chunk in chunks for buffer in chunk.token_ids],
            [buffer for chunk in chunks for buffer in chunk.logprobs],
            [counts for chunk in chunks for counts in chunk.counts],
            first.rows_per_request,
        )

    def to_cpu_nonblocking(self) -> SamplingSupportTensors:
        return replace(
            self,
            counts=[counts.to("cpu", non_blocking=True) for counts in self.counts],
        )

    def tolists(
        self, num_sampled_tokens: np.ndarray | None = None
    ) -> SamplingMaskLists:
        """Copy the filled prefixes and return the scheduler's CSR representation."""
        chunk_counts = [counts.cpu().numpy() for counts in self.counts]
        counts = np.concatenate(chunk_counts)
        totals = [int(chunk.sum()) for chunk in chunk_counts]
        token_ids = _copy_prefixes(self.token_ids, totals)
        logprobs = _copy_prefixes(self.logprobs, totals)

        cu_num_generated_tokens = None
        if self.rows_per_request != 1:
            assert num_sampled_tokens is not None
            # Uncommitted slots hold no entries, so dropping them keeps the
            # flat buffers aligned.
            active_slots = (
                np.arange(self.rows_per_request)[None, :] < num_sampled_tokens[:, None]
            )
            counts = counts[np.flatnonzero(active_slots)]
            cu_num_generated_tokens = np.cumsum(
                np.concatenate(([0], num_sampled_tokens))
            ).tolist()

        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return SamplingMaskLists(token_ids, offsets, cu_num_generated_tokens, logprobs)


_PREFIX_COPY_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _copy_prefixes(
    buffers: Sequence[torch.Tensor], totals: Sequence[int]
) -> np.ndarray:
    """Host copy of each buffer's first `total` entries, joined."""
    device = buffers[0].device
    prefixes = [buffer[:total] for buffer, total in zip(buffers, totals)]
    if device.type != "cpu":
        # The step's main and copy streams already queue the next step, so the
        # prefixes go through a stream of their own.
        copy_stream = _PREFIX_COPY_STREAMS.get(device)
        if copy_stream is None:
            from vllm.v1.worker.gpu.async_utils import new_d2h_stream

            copy_stream = _PREFIX_COPY_STREAMS[device] = new_d2h_stream(device)
        with torch.cuda.stream(copy_stream):
            prefixes = [prefix.to("cpu", non_blocking=True) for prefix in prefixes]
        copy_stream.synchronize()
    arrays = [prefix.numpy() for prefix in prefixes]
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
