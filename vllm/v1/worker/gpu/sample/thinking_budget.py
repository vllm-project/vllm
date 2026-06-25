# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig


class ThinkingBudgetState:
    """Model Runner V2 state for per-request thinking token budgets."""

    def __init__(
        self,
        req_states: RequestState,
        reasoning_config: "ReasoningConfig | None",
    ):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device

        self.thinking_token_budget = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.thinking_token_budget.np.fill(-1)
        self.thinking_token_budget.copy_to_uva()
        self.use_thinking_budget = np.zeros(self.max_num_reqs, dtype=bool)

        start_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_start_token_ids or []
        )
        end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_end_token_ids or []
        )
        self.enabled = bool(start_ids and end_ids)
        self.reasoning_start_token_ids = torch.tensor(
            start_ids or [0], dtype=torch.int32, device=self.device
        )
        self.reasoning_end_token_ids = torch.tensor(
            end_ids or [0], dtype=torch.int32, device=self.device
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        budget = sampling_params.thinking_token_budget
        if budget is None or not self.enabled:
            self.thinking_token_budget.np[req_idx] = -1
            self.use_thinking_budget[req_idx] = False
            return

        self.thinking_token_budget.np[req_idx] = budget
        self.use_thinking_budget[req_idx] = True

    def remove_request(self, req_idx: int) -> None:
        self.thinking_token_budget.np[req_idx] = -1
        self.use_thinking_budget[req_idx] = False

    def apply_staged_writes(self) -> None:
        self.thinking_token_budget.copy_to_uva()

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        if not self.enabled or not np.any(self.use_thinking_budget[idx_mapping_np]):
            return

        apply_thinking_budget(
            logits,
            expanded_idx_mapping,
            self.thinking_token_budget.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            expanded_local_pos,
            self.reasoning_start_token_ids,
            self.reasoning_end_token_ids,
        )


@triton.jit
def _load_effective_token(
    all_token_ids_ptr,
    all_token_ids_stride,
    input_ids_ptr,
    cur_req_first_pos,
    req_state_idx,
    total_len,
    pos,
):
    if pos < total_len:
        return tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos)
    # In decode/spec-decode, input_ids at local position 0 is the already
    # committed last sampled token. Effective draft-prefix positions start at
    # local position 1.
    input_pos = cur_req_first_pos + pos - total_len + 1
    return tl.load(input_ids_ptr + input_pos)


@triton.jit
def _thinking_budget_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    reasoning_start_token_ids_ptr,
    reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - local_pos
    total_len = tl.load(total_len_ptr + req_state_idx)
    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    effective_len = total_len + local_pos

    last_start = -1
    last_end = -1
    for i in tl.range(0, effective_len):
        if i + START_LEN <= effective_len:
            start_match = True
            for j in tl.static_range(0, START_LEN):
                expected = tl.load(reasoning_start_token_ids_ptr + j)
                actual = _load_effective_token(
                    all_token_ids_ptr,
                    all_token_ids_stride,
                    input_ids_ptr,
                    cur_req_first_pos,
                    req_state_idx,
                    total_len,
                    i + j,
                )
                start_match = start_match & (actual == expected)
            if start_match:
                last_start = i

        if i + END_LEN <= effective_len:
            end_match = True
            for j in tl.static_range(0, END_LEN):
                expected = tl.load(reasoning_end_token_ids_ptr + j)
                actual = _load_effective_token(
                    all_token_ids_ptr,
                    all_token_ids_stride,
                    input_ids_ptr,
                    cur_req_first_pos,
                    req_state_idx,
                    total_len,
                    i + j,
                )
                end_match = end_match & (actual == expected)
            if end_match:
                last_end = i

    if last_start < 0 or last_start <= last_end:
        return

    reasoning_start = last_start + START_LEN
    # If the request resumes from a prompt that already contains generated
    # reasoning content, count it against the remaining budget just like the
    # original Model Runner V1 path.
    num_reasoning_tokens = effective_len - reasoning_start
    if num_reasoning_tokens < budget:
        return

    end_prefix_len = 0
    max_prefix_len = END_LEN - 1
    if effective_len - prompt_len < max_prefix_len:
        max_prefix_len = effective_len - prompt_len

    for prefix_len in tl.static_range(1, END_LEN):
        if prefix_len <= max_prefix_len:
            prefix_match = True
            suffix_start = effective_len - prefix_len
            for j in tl.static_range(0, END_LEN):
                if j < prefix_len:
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    actual = _load_effective_token(
                        all_token_ids_ptr,
                        all_token_ids_stride,
                        input_ids_ptr,
                        cur_req_first_pos,
                        req_state_idx,
                        total_len,
                        suffix_start + j,
                    )
                    prefix_match = prefix_match & (actual == expected)
            if prefix_match:
                end_prefix_len = prefix_len

    force_token_id = tl.load(reasoning_end_token_ids_ptr + end_prefix_len)
    for i in tl.range(0, vocab_size, LOGITS_BLOCK_SIZE):
        offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
        tl.store(
            logits_ptr + token_idx * logits_stride + offset,
            -float("inf"),
            mask=offset < vocab_size,
        )
    tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)


def apply_thinking_budget(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    reasoning_end_token_ids: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    _thinking_budget_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        total_len,
        input_ids,
        expanded_local_pos,
        reasoning_start_token_ids,
        reasoning_end_token_ids,
        START_LEN=reasoning_start_token_ids.shape[0],
        END_LEN=reasoning_end_token_ids.shape[0],
        LOGITS_BLOCK_SIZE=8192,
    )
