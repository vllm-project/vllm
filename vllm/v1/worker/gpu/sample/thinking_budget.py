# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.thinking_budget_state import (
    ThinkingBudgetStateHolder,
    maybe_create_thinking_budget_state_holder,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class ThinkingBudgetState:
    """V2 adapter for the V1 thinking-token-budget state machine."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        max_num_reqs: int,
        num_spec_tokens: int,
        device: torch.device,
        pin_memory: bool,
        req_states: RequestState,
    ) -> None:
        self.req_states = req_states
        self.holder = maybe_create_thinking_budget_state_holder(
            vllm_config.reasoning_config,
            max_num_reqs,
            num_spec_tokens,
            device,
            pin_memory,
        )
        self.request_output_token_ids: dict[int, list[int]] = {}

    @property
    def is_enabled(self) -> bool:
        return self.holder is not None and self.holder.is_enabled

    def add_request(
        self,
        req_idx: int,
        prompt_token_ids: list[int] | None,
        output_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> None:
        if not self.is_enabled:
            return
        self.remove_request(req_idx)

        thinking_token_budget = sampling_params.thinking_token_budget
        if thinking_token_budget is None:
            return

        assert self.holder is not None
        self.request_output_token_ids[req_idx] = output_token_ids
        self.holder._state[req_idx] = self.holder._init_state_entry(
            prompt_token_ids,
            thinking_token_budget,
        )
        self.holder._state[req_idx]["output_tok_ids"] = output_token_ids
        self.holder._state[req_idx]["spec_token_ids"] = []

    def remove_request(self, req_idx: int) -> None:
        if self.holder is not None:
            self.holder._state.pop(req_idx, None)
        self.request_output_token_ids.pop(req_idx, None)

    def has_requests(self, idx_mapping_np: np.ndarray) -> bool:
        if self.holder is None or not self.holder.has_tracked_requests():
            return False
        return any(
            int(req_idx) in self.request_output_token_ids for req_idx in idx_mapping_np
        )

    def apply(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        predict_bonus_token: bool = False,
    ) -> torch.Tensor:
        if not self.has_requests(input_batch.idx_mapping_np):
            return logits

        holder = self._make_batch_holder(input_batch, predict_bonus_token)
        if not holder.has_tracked_requests():
            return logits
        return self._apply_forcing_to_logits(
            logits, input_batch, holder, predict_bonus_token
        )

    def update_after_sample(
        self,
        input_batch: InputBatch,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> None:
        if not self.has_requests(input_batch.idx_mapping_np):
            return

        sampled_token_ids_cpu = sampled_token_ids.cpu().tolist()
        num_sampled_cpu = num_sampled.cpu().tolist()
        for batch_idx, req_idx_np in enumerate(input_batch.idx_mapping_np):
            req_idx = int(req_idx_np)
            output_token_ids = self.request_output_token_ids.get(req_idx)
            if output_token_ids is None:
                continue
            num_tokens = int(num_sampled_cpu[batch_idx])
            if num_tokens <= 0:
                continue
            output_token_ids.extend(sampled_token_ids_cpu[batch_idx][:num_tokens])

    def _make_batch_holder(
        self,
        input_batch: InputBatch,
        predict_bonus_token: bool,
    ) -> ThinkingBudgetStateHolder:
        assert self.holder is not None
        holder = ThinkingBudgetStateHolder(
            reasoning_config=None,
            max_num_seqs=input_batch.num_reqs,
            num_spec_tokens=0 if predict_bonus_token else self.holder.num_spec_tokens,
            device=self.holder.device,
            is_pin_memory=False,
        )
        holder.is_enabled = self.holder.is_enabled
        holder.think_start_token_ids = self.holder.think_start_token_ids
        holder.think_end_token_ids = self.holder.think_end_token_ids
        holder.in_spec_mode = (
            input_batch.num_draft_tokens > 0 and not predict_bonus_token
        )
        holder.num_spec_tokens = (
            0 if predict_bonus_token else self.holder.num_spec_tokens
        )
        if holder.in_spec_mode:
            holder._mask_capacity = input_batch.num_reqs * (holder.num_spec_tokens + 1)
        else:
            holder._mask_capacity = input_batch.num_reqs

        output_token_ids: list[list[int]] = []
        spec_token_ids = self._get_spec_token_ids(input_batch)
        active_batch_indices: list[int] = []
        for batch_idx, req_idx_np in enumerate(input_batch.idx_mapping_np):
            req_idx = int(req_idx_np)
            state = self.holder._state.get(req_idx)
            output_tokens = self.request_output_token_ids.get(req_idx)
            if state is None or output_tokens is None:
                output_token_ids.append([])
                continue

            active_batch_indices.append(batch_idx)
            holder._state[batch_idx] = deepcopy(state)
            output_token_ids.append(output_tokens)

        if not active_batch_indices:
            return holder

        holder.update_state(output_token_ids, spec_token_ids, repeat_indices=None)
        self._sync_batch_holder(holder, input_batch)
        return holder

    def _sync_batch_holder(
        self,
        holder: ThinkingBudgetStateHolder,
        input_batch: InputBatch,
    ) -> None:
        assert self.holder is not None
        for batch_idx, state in holder._state.items():
            req_idx = int(input_batch.idx_mapping_np[batch_idx])
            self.holder._state[req_idx] = deepcopy(state)

    def _apply_forcing_to_logits(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        holder: ThinkingBudgetStateHolder,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        active_indices_cpu: list[int] = []
        force_tokens_cpu: list[int] = []

        for batch_idx in sorted(holder._state.keys()):
            if batch_idx >= input_batch.num_reqs:
                continue
            state = holder._state[batch_idx]
            if not state.get("in_end", False):
                continue

            force_index = state.get("force_index", [])
            if not force_index:
                continue

            if predict_bonus_token:
                spec_len = len(state.get("spec_token_ids", []))
                if force_index[0] < spec_len:
                    continue
                force_index = [0]

            end_count = state.get("end_count", 0)
            if end_count >= len(holder.think_end_token_ids):
                continue

            row_start = int(input_batch.cu_num_logits_np[batch_idx])
            row_end = int(input_batch.cu_num_logits_np[batch_idx + 1])
            if predict_bonus_token:
                row_end = row_start + 1

            for force_idx in force_index:
                mask_idx = row_start + int(force_idx)
                if row_start <= mask_idx < row_end and mask_idx < logits.shape[0]:
                    active_indices_cpu.append(mask_idx)
                    force_tokens_cpu.append(holder.think_end_token_ids[end_count])

        if not active_indices_cpu:
            return logits

        device = logits.device
        active_indices = async_tensor_h2d(
            active_indices_cpu, dtype=torch.long, device=device
        )
        force_tokens = async_tensor_h2d(
            force_tokens_cpu, dtype=torch.long, device=device
        )
        fill = logits.new_full((len(active_indices_cpu),), 1e9)
        logits.index_put_((active_indices, force_tokens), fill)
        return logits

    def _get_spec_token_ids(self, input_batch: InputBatch) -> list[list[int]]:
        if input_batch.num_draft_tokens == 0:
            return [[] for _ in range(input_batch.num_reqs)]

        spec_token_ids: list[list[int]] = []
        for batch_idx, req_idx_np in enumerate(input_batch.idx_mapping_np):
            num_draft_tokens = int(input_batch.num_draft_tokens_per_req[batch_idx])
            if num_draft_tokens == 0:
                spec_token_ids.append([])
                continue
            req_idx = int(req_idx_np)
            spec_token_ids.append(
                self.req_states.draft_tokens[req_idx, :num_draft_tokens]
                .detach()
                .cpu()
                .tolist()
            )
        return spec_token_ids
