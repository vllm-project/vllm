# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

_INITIAL_POOL_SLOTS = 8


def params_need_penalties(sampling_params: SamplingParams) -> bool:
    return (
        sampling_params.repetition_penalty != 1.0
        or sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
    )


class PenaltiesState:
    """Persistent per-request penalty statistics for the v1 GPU sampler.

    Instead of rebuilding a padded [num_rows, max_output_len] token matrix on
    the CPU every step (O(rows * max_len) per step, the source of multi-ms
    stalls with long outputs), this keeps per-request vocab statistics on the
    sampler device and updates them incrementally:

    - ``output_bin_counts[slot, token]``: occurrence count of ``token`` in the
      request's output so far (frequency penalty; ``>0`` for presence).
    - ``prompt_bin_mask[slot, token // 32]``: packed bitmask of tokens that
      appear in the prompt (repetition penalty).

    Slots are pooled and only allocated for requests that actually use
    penalties; ``row_to_slot`` maps batch rows to pool slots (-1 = row has no
    penalties). Rows move within the batch (swap/condense) by updating the
    mapping only; slot contents never move. The pool grows geometrically on
    demand, so a batch with no penalty requests costs nothing.

    Per-step costs: building the statistics is paid once at request admission
    (O(seq_len) bincount); each step only commits the newly accepted tokens
    (O(batch * spec_len) scatter) and applies penalties with a kernel that
    reads the statistics in place. Draft tokens from the current step are
    never committed; the apply kernel superimposes the per-row draft prefix
    on the fly, which preserves rejection-sampling semantics.
    """

    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size
        self.device = device

        # Batch row -> pool slot; -1 means the row has no penalties.
        self.row_to_slot = np.full(max_num_reqs, -1, dtype=np.int32)
        self.free_slots: list[int] = []
        self.capacity = 0
        # Lazily allocated on first penalty request:
        # output_bin_counts: int32 [capacity, vocab_size]
        # prompt_bin_mask:   int32 [capacity, cdiv(vocab_size, 32)]
        self.output_bin_counts: torch.Tensor | None = None
        self.prompt_bin_mask: torch.Tensor | None = None

        # Admissions staged until the next make_slot_mapping() call:
        # (slot, token_ids copy, prompt_len).
        self._staged: list[tuple[int, np.ndarray, int]] = []

    @property
    def no_penalties(self) -> bool:
        return self.capacity == len(self.free_slots) and not self._staged

    def _ensure_capacity(self) -> None:
        if self.free_slots:
            return
        new_capacity = max(_INITIAL_POOL_SLOTS, self.capacity * 2)
        new_capacity = min(new_capacity, self.max_num_reqs)
        assert new_capacity > self.capacity
        counts = torch.zeros(
            new_capacity, self.vocab_size, dtype=torch.int32, device=self.device
        )
        mask = torch.zeros(
            new_capacity,
            cdiv(self.vocab_size, 32),
            dtype=torch.int32,
            device=self.device,
        )
        if self.capacity > 0:
            assert self.output_bin_counts is not None
            assert self.prompt_bin_mask is not None
            counts[: self.capacity] = self.output_bin_counts
            mask[: self.capacity] = self.prompt_bin_mask
        self.output_bin_counts = counts
        self.prompt_bin_mask = mask
        self.free_slots.extend(range(self.capacity, new_capacity))
        self.capacity = new_capacity

    def add_request(
        self,
        row: int,
        sampling_params: SamplingParams,
        token_ids: np.ndarray,
        prompt_len: int,
    ) -> None:
        """Admit the request at ``row``.

        ``token_ids`` holds the request's full token history so far
        (prompt followed by any already-generated output, e.g. when resuming
        after preemption); statistics are rebuilt from it.
        """
        assert self.row_to_slot[row] == -1, f"row {row} already has a slot"
        if not params_need_penalties(sampling_params):
            return
        self._ensure_capacity()
        slot = self.free_slots.pop()
        self.row_to_slot[row] = slot
        # Copy: the caller's array is a live view of the batch buffer.
        self._staged.append((slot, token_ids.astype(np.int64), prompt_len))

    def remove_row(self, row: int) -> None:
        slot = self.row_to_slot[row]
        if slot != -1:
            self.row_to_slot[row] = -1
            self.free_slots.append(int(slot))
            self._staged = [s for s in self._staged if s[0] != slot]

    def swap_rows(self, row1: int, row2: int) -> None:
        self.row_to_slot[[row1, row2]] = self.row_to_slot[[row2, row1]]

    def move_row(self, src: int, dst: int) -> None:
        assert self.row_to_slot[dst] == -1, f"row {dst} still has a slot"
        self.row_to_slot[dst] = self.row_to_slot[src]
        self.row_to_slot[src] = -1

    def make_slot_mapping(self, num_reqs: int) -> torch.Tensor:
        """Flush staged admissions and snapshot row -> slot for the device.

        Called when the batch composition changes (metadata refresh). Returns
        a fresh tensor so kernels enqueued with an older snapshot are
        unaffected by later changes.
        """
        self._flush_staged()
        return torch.from_numpy(self.row_to_slot[:num_reqs].copy()).to(
            self.device, non_blocking=True
        )

    def _flush_staged(self) -> None:
        if not self._staged:
            return
        assert self.output_bin_counts is not None
        assert self.prompt_bin_mask is not None
        slots, token_arrays, prompt_lens = zip(*self._staged)
        self._staged = []

        slots_t = torch.tensor(slots, dtype=torch.int32, device=self.device)
        # Zero on alloc: slots are recycled dirty.
        slots_long = slots_t.long()
        self.output_bin_counts.index_fill_(0, slots_long, 0)
        self.prompt_bin_mask.index_fill_(0, slots_long, 0)

        lens = [len(t) for t in token_arrays]
        starts = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=starts[1:])
        flat = torch.from_numpy(np.concatenate(token_arrays)).to(
            self.device, non_blocking=True
        )
        starts_t = torch.from_numpy(starts).to(self.device, non_blocking=True)
        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.int64, device=self.device)
        total_lens_t = torch.tensor(lens, dtype=torch.int64, device=self.device)

        if self.device.type == "cpu":
            self._bincount_native(slots_t, flat, starts_t, prompt_lens_t, total_lens_t)
        else:
            max_len = max(lens)
            BLOCK_SIZE = 1024
            _admission_bincount_kernel[(len(slots), cdiv(max_len, BLOCK_SIZE))](
                slots_t,
                flat,
                starts_t,
                prompt_lens_t,
                total_lens_t,
                self.prompt_bin_mask,
                self.prompt_bin_mask.stride(0),
                self.output_bin_counts,
                self.output_bin_counts.stride(0),
                self.vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    def _bincount_native(
        self,
        slots: torch.Tensor,
        flat_token_ids: torch.Tensor,
        starts: torch.Tensor,
        prompt_lens: torch.Tensor,
        total_lens: torch.Tensor,
    ) -> None:
        assert self.output_bin_counts is not None
        assert self.prompt_bin_mask is not None
        packed_vocab = self.prompt_bin_mask.shape[1]
        for i, slot in enumerate(slots.tolist()):
            start = int(starts[i])
            prompt_len = int(prompt_lens[i])
            total_len = int(total_lens[i])
            tokens = flat_token_ids[start : start + total_len]
            valid = (tokens >= 0) & (tokens < self.vocab_size)
            prompt_tokens = tokens[:prompt_len][valid[:prompt_len]]
            present = torch.zeros(
                packed_vocab * 32, dtype=torch.bool, device=self.device
            )
            present[prompt_tokens] = True
            bits = present.view(packed_vocab, 32).int() << torch.arange(
                32, dtype=torch.int32, device=self.device
            )
            self.prompt_bin_mask[slot] = bits.sum(dim=1, dtype=torch.int32)
            output_tokens = tokens[prompt_len:][valid[prompt_len:]]
            self.output_bin_counts[slot] = torch.bincount(
                output_tokens, minlength=self.vocab_size
            ).to(torch.int32)

    def commit(
        self,
        sampled_token_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        discard_mask: torch.Tensor | None,
    ) -> None:
        """Fold the step's accepted tokens into the statistics.

        ``sampled_token_ids`` is the sampler's device output,
        [num_reqs, num_sampled]; rejected/invalid positions hold negative or
        >= vocab_size placeholders and are skipped, matching
        ``RejectionSampler.parse_output``.
        """
        if self.no_penalties:
            return
        assert self.output_bin_counts is not None
        num_reqs, num_sampled = sampled_token_ids.shape
        if self.device.type == "cpu":
            self._commit_native(sampled_token_ids, slot_mapping, discard_mask)
            return
        _commit_kernel[(num_reqs,)](
            sampled_token_ids,
            sampled_token_ids.stride(0),
            slot_mapping,
            discard_mask,
            self.output_bin_counts,
            self.output_bin_counts.stride(0),
            self.vocab_size,
            num_sampled,
            HAS_DISCARD=discard_mask is not None,
        )

    def _commit_native(
        self,
        sampled_token_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        discard_mask: torch.Tensor | None,
    ) -> None:
        assert self.output_bin_counts is not None
        for req_idx, slot in enumerate(slot_mapping.tolist()):
            if slot < 0:
                continue
            if discard_mask is not None and bool(discard_mask[req_idx]):
                continue
            tokens = sampled_token_ids[req_idx]
            tokens = tokens[(tokens >= 0) & (tokens < self.vocab_size)]
            self.output_bin_counts[slot] += torch.bincount(
                tokens, minlength=self.vocab_size
            ).to(torch.int32)

    def apply(
        self,
        logits: torch.Tensor,
        slot_mapping: torch.Tensor,
        repetition_penalties: torch.Tensor,
        frequency_penalties: torch.Tensor,
        presence_penalties: torch.Tensor,
        row_to_req: torch.Tensor | None = None,
        prefix_lens: torch.Tensor | None = None,
        draft_token_ids: torch.Tensor | None = None,
        draft_starts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply penalties to ``logits`` in place.

        ``row_to_req`` maps logits rows to request indices (identity when
        None). ``prefix_lens``/``draft_starts`` describe, per row, how many of
        the current step's draft tokens (a slice of flat ``draft_token_ids``)
        to superimpose on the committed statistics: the row for draft
        position p sees drafts [0, p), a bonus row sees all of them.
        """
        num_rows, vocab_size = logits.shape
        assert vocab_size == self.vocab_size
        assert self.output_bin_counts is not None
        assert self.prompt_bin_mask is not None

        if row_to_req is not None:
            row_to_req_long = row_to_req.long()
            slot_of_row = slot_mapping[row_to_req_long]
            repetition_penalties = repetition_penalties[row_to_req_long]
            frequency_penalties = frequency_penalties[row_to_req_long]
            presence_penalties = presence_penalties[row_to_req_long]
        else:
            slot_of_row = slot_mapping

        has_prefix = prefix_lens is not None
        if has_prefix:
            assert draft_token_ids is not None and draft_starts is not None
        else:
            # Unused; any tensor keeps the kernel signature uniform.
            prefix_lens = slot_of_row
            draft_token_ids = slot_of_row
            draft_starts = slot_of_row

        if self.device.type == "cpu":
            return self._apply_native(
                logits,
                slot_of_row,
                repetition_penalties,
                frequency_penalties,
                presence_penalties,
                prefix_lens if has_prefix else None,
                draft_token_ids,
                draft_starts,
            )

        BLOCK_SIZE = 8192
        _apply_penalties_kernel[(num_rows, cdiv(vocab_size, BLOCK_SIZE))](
            logits,
            logits.stride(0),
            slot_of_row,
            repetition_penalties,
            frequency_penalties,
            presence_penalties,
            prefix_lens,
            draft_token_ids,
            draft_starts,
            self.prompt_bin_mask,
            self.prompt_bin_mask.stride(0),
            self.output_bin_counts,
            self.output_bin_counts.stride(0),
            vocab_size,
            HAS_PREFIX=has_prefix,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return logits

    def _apply_native(
        self,
        logits: torch.Tensor,
        slot_of_row: torch.Tensor,
        repetition_penalties: torch.Tensor,
        frequency_penalties: torch.Tensor,
        presence_penalties: torch.Tensor,
        prefix_lens: torch.Tensor | None,
        draft_token_ids: torch.Tensor,
        draft_starts: torch.Tensor,
    ) -> torch.Tensor:
        assert self.output_bin_counts is not None
        assert self.prompt_bin_mask is not None
        vocab_size = self.vocab_size
        for row, slot in enumerate(slot_of_row.tolist()):
            if slot < 0:
                continue
            counts = self.output_bin_counts[slot]
            if prefix_lens is not None and int(prefix_lens[row]) > 0:
                start = int(draft_starts[row])
                prefix = draft_token_ids[start : start + int(prefix_lens[row])]
                counts = counts + torch.bincount(
                    prefix.long(), minlength=vocab_size
                ).to(torch.int32)
            output_mask = counts > 0

            row_logits = logits[row]
            rep = float(repetition_penalties[row])
            if rep != 1.0:
                packed = self.prompt_bin_mask[slot]
                bits = (packed.unsqueeze(1) >> torch.arange(32, device=self.device)) & 1
                prompt_mask = bits.reshape(-1)[:vocab_size].bool()
                penalized = prompt_mask | output_mask
                scale = torch.where(penalized, torch.full_like(row_logits, rep), 1.0)
                row_logits *= torch.where(row_logits > 0, 1.0 / scale, scale)

            row_logits -= float(frequency_penalties[row]) * counts
            row_logits -= float(presence_penalties[row]) * output_mask
        return logits


@triton.jit
def _admission_bincount_kernel(
    slots_ptr,
    flat_token_ids_ptr,
    starts_ptr,
    prompt_lens_ptr,
    total_lens_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    item_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    total_len = tl.load(total_lens_ptr + item_idx)
    if block_idx * BLOCK_SIZE >= total_len:
        return

    slot = tl.load(slots_ptr + item_idx).to(tl.int64)
    start = tl.load(starts_ptr + item_idx)
    prompt_len = tl.load(prompt_lens_ptr + item_idx)

    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tokens = tl.load(flat_token_ids_ptr + start + offsets, mask=offsets < total_len)
    # Negative ids are async-scheduling placeholders; >= vocab_size cannot
    # occur in committed history but is guarded for symmetry with commit.
    valid = (offsets < total_len) & (tokens >= 0) & (tokens < vocab_size)

    prompt_valid = valid & (offsets < prompt_len)
    bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << (tokens % 32)
    tl.atomic_or(
        prompt_bin_mask_ptr + slot * prompt_bin_mask_stride + tokens // 32,
        bit,
        mask=prompt_valid,
    )

    output_valid = valid & (offsets >= prompt_len)
    tl.atomic_add(
        output_bin_counts_ptr + slot * output_bin_counts_stride + tokens,
        1,
        mask=output_valid,
    )


@triton.jit
def _commit_kernel(
    sampled_token_ids_ptr,
    sampled_token_ids_stride,
    slot_mapping_ptr,
    discard_mask_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    num_sampled,
    HAS_DISCARD: tl.constexpr,
):
    req_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + req_idx).to(tl.int64)
    if slot < 0:
        return
    # Nested on purpose: with HAS_DISCARD=False the mask pointer is null and
    # must not be traced.
    if HAS_DISCARD:  # noqa: SIM102
        if tl.load(discard_mask_ptr + req_idx) != 0:
            return
    for i in tl.range(num_sampled):
        token = tl.load(sampled_token_ids_ptr + req_idx * sampled_token_ids_stride + i)
        if token >= 0 and token < vocab_size:
            tl.atomic_add(
                output_bin_counts_ptr + slot * output_bin_counts_stride + token,
                1,
            )


@triton.jit
def _apply_penalties_kernel(
    logits_ptr,
    logits_stride,
    slot_of_row_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prefix_lens_ptr,
    draft_token_ids_ptr,
    draft_starts_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    HAS_PREFIX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    slot = tl.load(slot_of_row_ptr + row_idx).to(tl.int64)
    if slot < 0:
        # Row without penalties: early return before touching logits.
        return

    rep_penalty = tl.load(repetition_penalty_ptr + row_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + row_idx)
    pres_penalty = tl.load(presence_penalty_ptr + row_idx)

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + row_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    output_counts = tl.load(
        output_bin_counts_ptr + slot * output_bin_counts_stride + block,
        mask=mask,
        other=0,
    )
    if HAS_PREFIX:
        # Superimpose this step's draft prefix without committing it.
        prefix_len = tl.load(prefix_lens_ptr + row_idx)
        draft_start = tl.load(draft_starts_ptr + row_idx)
        for i in tl.range(prefix_len):
            draft_token = tl.load(draft_token_ids_ptr + draft_start + i)
            output_counts += (block == draft_token).to(tl.int32)
    output_mask = output_counts > 0

    if rep_penalty != 1.0:
        packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed = tl.load(
            prompt_bin_mask_ptr + slot * prompt_bin_mask_stride + packed_block,
            mask=packed_block < tl.cdiv(vocab_size, 32),
            other=0,
        )
        prompt_bits = (packed[:, None] >> tl.arange(0, 32)[None, :]) & 1
        prompt_mask = prompt_bits.to(tl.int1).reshape(BLOCK_SIZE)

        scale = tl.where(prompt_mask | output_mask, rep_penalty, 1.0)
        logits *= tl.where(logits > 0, 1.0 / scale, scale)

    # OpenAI-API definitions of frequency/presence penalties.
    logits -= freq_penalty * output_counts
    logits -= pres_penalty * output_mask
    tl.store(logits_ptr + row_idx * logits_stride + block, logits, mask=mask)
