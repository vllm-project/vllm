# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState

MAX_BAD_WORDS_TOTAL_TOKENS = 1024  # Max total tokens for all bad words per request
MAX_NUM_BAD_WORDS = 128  # Max number of bad words per request


class BadWordsState:
    def __init__(self, req_states: RequestState):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device

        # flattened bad word tokens: [max_num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS]
        self.bad_word_token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS),
            dtype=torch.int32,
            device=self.device,
        )
        # cumulative offsets of bad words: [max_num_reqs, MAX_NUM_BAD_WORDS + 1]
        self.bad_word_offsets = StagedWriteTensor(
            (self.max_num_reqs, MAX_NUM_BAD_WORDS + 1),
            dtype=torch.int32,
            device=self.device,
        )
        # number of bad words per request
        self.num_bad_words = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        bad_words_token_ids = sampling_params.bad_words_token_ids
        if not bad_words_token_ids:
            self.num_bad_words.np[req_idx] = 0
            return

        num_bad_words = len(bad_words_token_ids)
        if num_bad_words > MAX_NUM_BAD_WORDS:
            raise ValueError(
                f"Too many bad words: {num_bad_words}. "
                f"The max number is {MAX_NUM_BAD_WORDS}."
            )

        # Flatten bad words and compute offsets
        flattened_tokens: list[int] = []
        offsets: list[int] = [0]
        for bad_word in bad_words_token_ids:
            flattened_tokens.extend(bad_word)
            offsets.append(len(flattened_tokens))

        if len(flattened_tokens) > MAX_BAD_WORDS_TOTAL_TOKENS:
            raise ValueError(
                f"Too many total bad word tokens: {len(flattened_tokens)}. "
                f"The max is {MAX_BAD_WORDS_TOTAL_TOKENS}."
            )

        # Stage writes
        self.bad_word_token_ids.stage_write(req_idx, 0, flattened_tokens)
        self.bad_word_offsets.stage_write(req_idx, 0, offsets)
        self.num_bad_words.np[req_idx] = num_bad_words

    def apply_staged_writes(self) -> None:
        self.num_bad_words.copy_to_uva()
        self.bad_word_token_ids.apply_write()
        self.bad_word_offsets.apply_write()

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        max_num_bad_words = int(self.num_bad_words.np[idx_mapping_np].max())
        if max_num_bad_words == 0:
            # No request uses bad words. Skip the kernel launch.
            return

        apply_bad_words(
            logits,
            idx_mapping,
            self.bad_word_token_ids.gpu,
            self.bad_word_offsets.gpu,
            self.num_bad_words.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            expanded_local_pos,
            max_num_bad_words,
        )


@triton.jit
def _bad_words_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    bad_word_token_ids_ptr,
    bad_word_token_ids_stride,
    bad_word_offsets_ptr,
    bad_word_offsets_stride,
    num_bad_words_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
):
    logit_idx = tl.program_id(0)
    bw_idx = tl.program_id(1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
    num_bad_words = tl.load(num_bad_words_ptr + req_state_idx)

    if bw_idx >= num_bad_words:
        return

    pos = tl.load(expanded_local_pos_ptr + logit_idx)
    cur_req_first_pos = logit_idx - pos

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    total_len = tl.load(total_len_ptr + req_state_idx)
    output_len = total_len - prompt_len
    effective_len = output_len + pos

    bd_offsets_base = bad_word_offsets_ptr + req_state_idx * bad_word_offsets_stride
    bd_tokens_base = bad_word_token_ids_ptr + req_state_idx * bad_word_token_ids_stride
    output_base = all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len

    start = tl.load(bd_offsets_base + bw_idx)
    end = tl.load(bd_offsets_base + bw_idx + 1)
    bad_word_len = end - start
    prefix_len = bad_word_len - 1

    if prefix_len > effective_len:
        return

    last_token = tl.load(bd_tokens_base + end - 1)
    match = 1
    for i in range(prefix_len):
        expected = tl.load(bd_tokens_base + start + i)
        actual_pos = effective_len - prefix_len + i

        from_spec_input = actual_pos >= output_len
        if from_spec_input:
            spec_offset = actual_pos - output_len
            actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset)
        else:
            actual = tl.load(output_base + actual_pos)

        match = match & (expected == actual)

    if match:
        tl.store(logits_ptr + logit_idx * logits_stride + last_token, -float("inf"))


def apply_bad_words(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    bad_word_token_ids: torch.Tensor,
    bad_word_offsets: torch.Tensor,
    num_bad_words: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    max_num_bad_words: int,
) -> None:
    total_num_tokens = logits.shape[0]
    _bad_words_kernel[(total_num_tokens, max_num_bad_words)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        bad_word_token_ids,
        bad_word_token_ids.stride(0),
        bad_word_offsets,
        bad_word_offsets.stride(0),
        num_bad_words,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        total_len,
        input_ids,
        expanded_local_pos,
    )
