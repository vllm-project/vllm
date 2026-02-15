# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor


class PenaltiesState:
    def __init__(self, max_num_reqs: int, vocab_size: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size
        self.device = device

        self.repetition_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.frequency_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.presence_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.use_penalty = np.zeros(max_num_reqs, dtype=bool)

        # Initialize repetition penalty manually because 0 is an invalid value for it.
        self.repetition_penalty.np.fill(1.0)
        self.repetition_penalty.copy_to_uva()

        # Statistics for penalties.
        self.prompt_bin_mask = torch.zeros(
            self.max_num_reqs,
            cdiv(self.vocab_size, 32),
            dtype=torch.int32,
            device=self.device,
        )
        # TODO(woosuk): This tensor is rarely used but can be very large, taking up
        # GBs of GPU memory. Optimize the memory usage.
        self.output_bin_counts = torch.zeros(
            self.max_num_reqs, self.vocab_size, dtype=torch.int32, device=self.device
        )

        self._penalties_reqs: list[int] = []

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        self.repetition_penalty.np[req_idx] = sampling_params.repetition_penalty
        self.frequency_penalty.np[req_idx] = sampling_params.frequency_penalty
        self.presence_penalty.np[req_idx] = sampling_params.presence_penalty

        do_penalty = use_penalty(sampling_params)
        self.use_penalty[req_idx] = do_penalty
        if do_penalty:
            self._penalties_reqs.append(req_idx)

    def apply_staged_writes(
        self,
        prefill_token_ids: torch.Tensor,
        prefill_lens: np.ndarray,
        prompt_lens: np.ndarray,
    ) -> None:
        # TODO(woosuk): Optimize this.
        for req_idx in self._penalties_reqs:
            bincount(
                prefill_token_ids[req_idx],
                int(prefill_lens[req_idx]),
                int(prompt_lens[req_idx]),
                self.prompt_bin_mask[req_idx],
                self.output_bin_counts[req_idx],
            )
        self._penalties_reqs.clear()

        self.repetition_penalty.copy_to_uva()
        self.frequency_penalty.copy_to_uva()
        self.presence_penalty.copy_to_uva()

    def apply_penalties(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        num_speculative_tokens: int,
    ) -> None:
        if not np.any(self.use_penalty[idx_mapping_np]):
            # No request uses penalties. Skip the kernel launch.
            return

        apply_penalties(
            logits,
            idx_mapping,
            input_ids,
            self.repetition_penalty.gpu,
            self.frequency_penalty.gpu,
            self.presence_penalty.gpu,
            self.prompt_bin_mask,
            self.output_bin_counts,
            num_speculative_tokens,
        )


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    token_ids_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty
    if not use_penalty:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    base_output_counts = tl.load(
        output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
        mask=mask,
        other=0,
    )

    # Derive local position within this request's logit group by scanning
    # backwards in expanded_idx_mapping. Since req_state_idx is unique per
    # active request, a change in value marks the request boundary.
    pos = 0
    for i in tl.static_range(MAX_SPEC_LEN):
        check_idx = token_idx - i - 1
        prev_req = tl.load(idx_mapping_ptr + check_idx, mask=check_idx >= 0, other=-1)
        pos += (prev_req == req_state_idx).to(tl.int32)
    start_idx = token_idx - pos
    draft_counts = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for prev_pos in tl.static_range(MAX_SPEC_LEN):
        if prev_pos < pos:
            prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
            token_match = block == prev_token
            draft_counts = draft_counts + token_match.to(tl.int32)

    # Total counts = base output counts + cumulative draft counts
    output_bin_counts = base_output_counts + draft_counts
    output_bin_mask = output_bin_counts > 0

    # Apply repetition penalties.
    if use_rep_penalty:
        packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed_mask = tl.load(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
            mask=packed_block < tl.cdiv(vocab_size, 32),
            other=0,
        )
        prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
        prompt_bin_mask = prompt_bin_mask.to(tl.int1)
        prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

        # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
        scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
        # If logits are positive, divide by penalty, otherwise multiply by penalty.
        logits *= tl.where(logits > 0, 1.0 / scale, scale)

    # Apply frequency penalties.
    logits -= freq_penalty * output_bin_counts
    # Apply presence penalties.
    logits -= pres_penalty * output_bin_mask
    # Store back to logits.
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_penalties(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    num_speculative_tokens: int,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _penalties_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        idx_mapping,
        token_ids,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_SPEC_LEN=num_speculative_tokens,
    )


@triton.jit(do_not_specialize=["prefill_len", "prompt_len"])
def _bincount_kernel(
    prefill_token_ids_ptr,
    prefill_len,
    prompt_len,
    prompt_bin_mask_ptr,
    output_bin_counts_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if block_idx * BLOCK_SIZE >= prefill_len:
        return

    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if block_idx * BLOCK_SIZE < prompt_len:
        mask = block < prompt_len
        prefill_tokens = tl.load(prefill_token_ids_ptr + block, mask=mask)
        idx = prefill_tokens // 32
        bit_idx = prefill_tokens % 32
        bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_idx
        tl.atomic_or(prompt_bin_mask_ptr + idx, bit, mask=mask)
    if (block_idx + 1) * BLOCK_SIZE >= prompt_len:
        mask = block < prefill_len
        mask &= block >= prompt_len
        prefill_tokens = tl.load(prefill_token_ids_ptr + block, mask=mask)
        tl.atomic_add(output_bin_counts_ptr + prefill_tokens, 1, mask=mask)


def bincount(
    prefill_token_ids: torch.Tensor,
    prefill_len: int,
    prompt_len: int,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    prompt_bin_mask.zero_()
    output_bin_counts.zero_()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(prefill_len, BLOCK_SIZE)
    _bincount_kernel[(num_blocks,)](
        prefill_token_ids,
        prefill_len,
        prompt_len,
        prompt_bin_mask,
        output_bin_counts,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def use_penalty(sampling_params: SamplingParams) -> bool:
    return (
        sampling_params.repetition_penalty != 1.0
        or sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
    )
