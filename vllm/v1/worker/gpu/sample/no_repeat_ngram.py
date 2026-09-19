# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched GPU exact no-repeat n-gram sampling state."""

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState


class NoRepeatNGramState:
    """Per-request configuration for output-only no-repeat n-gram masking.

    Token history already lives in :class:`RequestState`; retaining a second
    Python or GPU index costs more than scanning it in parallel for the long
    n-grams used by generative OCR models. The only additional persistent
    state is therefore one integer per request slot.
    """

    def __init__(self, req_states: RequestState, num_speculative_tokens: int):
        self.req_states = req_states
        self.max_model_len = req_states.max_model_len
        self.num_speculative_tokens = num_speculative_tokens
        self.ngram_sizes = UvaBackedTensor(req_states.max_num_reqs, dtype=torch.int32)
        self.ngram_sizes.np.fill(0)
        self._ever_enabled = False

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        extra_args = sampling_params.extra_args or {}
        raw_size = extra_args.get("no_repeat_ngram_size", 0)
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise ValueError("no_repeat_ngram_size must be an integer")
        ngram_size = raw_size
        if ngram_size < 0:
            raise ValueError("no_repeat_ngram_size cannot be negative")
        if ngram_size > self.max_model_len:
            raise ValueError(
                "no_repeat_ngram_size cannot exceed the configured max model length"
            )
        if ngram_size and self.num_speculative_tokens != 1:
            raise ValueError(
                "no_repeat_ngram_size does not yet support speculative decoding"
            )
        # Always overwrite reused slots, including requests with the feature off.
        self.ngram_sizes.np[req_idx] = ngram_size
        self._ever_enabled |= ngram_size > 0

    def apply_staged_writes(self) -> None:
        if self._ever_enabled:
            self.ngram_sizes.copy_to_uva()

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if (
            not self._ever_enabled
            or not (self.ngram_sizes.np[idx_mapping_np] >= 1).any()
        ):
            return
        apply_no_repeat_ngram(
            logits,
            expanded_idx_mapping,
            self.req_states.all_token_ids.gpu,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            self.ngram_sizes.gpu,
        )


@triton.jit
def _no_repeat_ngram_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    ngram_sizes_ptr,
    vocab_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    req_idx = tl.load(expanded_idx_mapping_ptr + row)
    ngram_size = tl.load(ngram_sizes_ptr + req_idx)
    prompt_len = tl.load(prompt_len_ptr + req_idx)
    output_len = tl.load(total_len_ptr + req_idx) - prompt_len
    if ngram_size < 1 or output_len < ngram_size:
        return

    starts = tl.program_id(1) * block_size + tl.arange(0, block_size)
    valid = starts <= output_len - ngram_size
    output = all_token_ids_ptr + req_idx * all_token_ids_stride + prompt_len
    matches = valid
    if ngram_size > 1:
        current_prefix = output_len - ngram_size + 1
        matches &= tl.load(output + starts, mask=valid, other=-1) == tl.load(
            output + current_prefix
        )
        for offset in range(1, ngram_size - 1):
            matches &= tl.load(
                output + starts + offset, mask=matches, other=-1
            ) == tl.load(output + current_prefix + offset)

    banned = tl.load(output + starts + ngram_size - 1, mask=matches, other=0)
    tl.atomic_xchg(
        logits_ptr + row * logits_stride + banned,
        -float("inf"),
        mask=matches & (banned >= 0) & (banned < vocab_size),
        sem="relaxed",
    )


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    ngram_sizes: torch.Tensor,
) -> None:
    _no_repeat_ngram_kernel[
        (logits.shape[0], triton.cdiv(all_token_ids.shape[1], 256))
    ](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_lens,
        total_lens,
        ngram_sizes,
        logits.shape[1],
        256,
    )
