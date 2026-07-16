# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
    from vllm.v1.worker.gpu.states import RequestState


class AdaptiveDraftingManager(ABC):
    @abstractmethod
    def plan_batch(self, scheduler_output: SchedulerOutput) -> tuple[int, int]:
        """Return the target token count and aggregate draft-token budget."""
        raise NotImplementedError

    @abstractmethod
    def prepare_verification_layout(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: Sequence[str],
        idx_mapping: torch.Tensor,
        draft_token_budget: int,
        query_start_loc: torch.Tensor,
        num_reqs_padded: int,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Allocate the draft budget and build query and logit layouts."""
        raise NotImplementedError


class DynamicSDDraftingManager(AdaptiveDraftingManager):
    def __init__(
        self,
        num_speculative_tokens_per_batch_size: object,
        max_num_reqs: int,
        max_num_spec_tokens: int,
        device: torch.device,
        speculator: BaseSpeculator,
        req_states: RequestState,
    ) -> None:
        self._speculator = speculator
        self._req_states = req_states
        self._num_spec_tokens_by_batch_size = build_dynamic_sd_schedule_lookup(
            num_speculative_tokens_per_batch_size,
            vllm_max_batch_size=max_num_reqs,
            vllm_num_speculative_tokens=max_num_spec_tokens,
        )
        self._draft_token_caps = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._base_query_counts = torch.empty_like(self._draft_token_caps)
        self._cu_num_logits = torch.empty(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )

    def plan_batch(self, scheduler_output: SchedulerOutput) -> tuple[int, int]:
        req_ids = tuple(scheduler_output.num_scheduled_tokens)
        if not req_ids:
            return 0, 0

        draft_caps, base_query_counts, is_prefill = self._get_request_layout(
            scheduler_output, req_ids
        )
        target_k = self._num_spec_tokens_by_batch_size[
            -1 if is_prefill.any() else len(req_ids)
        ]
        num_decode_reqs = len(req_ids) - int(is_prefill.sum())
        draft_token_budget = min(
            int(draft_caps.sum()),
            target_k * num_decode_reqs,
        )
        return int(base_query_counts.sum()) + draft_token_budget, draft_token_budget

    def prepare_verification_layout(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: Sequence[str],
        idx_mapping: torch.Tensor,
        draft_token_budget: int,
        query_start_loc: torch.Tensor,
        num_reqs_padded: int,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        draft_caps, base_query_counts, is_prefill = self._get_request_layout(
            scheduler_output, req_ids
        )
        num_reqs = len(req_ids)
        draft_caps_gpu = async_copy_to_gpu(
            draft_caps, out=self._draft_token_caps[:num_reqs]
        )
        base_query_counts_gpu = async_copy_to_gpu(
            base_query_counts, out=self._base_query_counts[:num_reqs]
        )
        num_draft_tokens_per_req = self._speculator.allocate_draft_token_budget(
            idx_mapping,
            draft_token_budget,
            draft_caps_gpu,
        )

        query_start_loc[:1].zero_()
        torch.cumsum(
            base_query_counts_gpu + num_draft_tokens_per_req,
            dim=0,
            out=query_start_loc[1 : num_reqs + 1],
        )
        num_tokens = int(base_query_counts.sum()) + draft_token_budget
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        cu_num_logits = self._cu_num_logits[: num_reqs + 1]
        cu_num_logits[:1].zero_()
        torch.cumsum(
            num_draft_tokens_per_req + 1,
            dim=0,
            out=cu_num_logits[1:],
        )

        if is_prefill.any():
            # Prompt logprobs use host-side query offsets, so mixed prefill
            # batches need the exact device allocation reflected on the host.
            if draft_token_budget == int(draft_caps.sum()):
                actual_draft_counts = draft_caps
            else:
                actual_draft_counts = (
                    num_draft_tokens_per_req.detach().to("cpu").numpy()
                )
            query_counts = base_query_counts + actual_draft_counts
        else:
            query_counts = np.fromiter(
                map(scheduler_output.num_scheduled_tokens.get, req_ids),
                dtype=np.int32,
                count=num_reqs,
            )

        query_start_loc_np = np.empty(num_reqs_padded + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(query_counts, out=query_start_loc_np[1 : num_reqs + 1])
        query_start_loc_np[num_reqs + 1 :] = query_start_loc_np[num_reqs]
        return num_draft_tokens_per_req, cu_num_logits, query_start_loc_np

    def _get_request_layout(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_reqs = len(req_ids)
        draft_caps = np.empty(num_reqs, dtype=np.int32)
        base_query_counts = np.empty(num_reqs, dtype=np.int32)
        is_prefill = np.empty(num_reqs, dtype=np.bool_)
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        scheduled_tokens = scheduler_output.num_scheduled_tokens

        for index, req_id in enumerate(req_ids):
            token_ids = draft_tokens.get(req_id, ())
            base_query_counts[index] = scheduled_tokens[req_id] - len(token_ids)
            if base_query_counts[index] < 0:
                raise ValueError(
                    f"Request {req_id} has more draft tokens than scheduled tokens."
                )
            is_prefill[index] = self._is_prefill(req_id)
            draft_caps[index] = 0 if is_prefill[index] else len(token_ids)
        return draft_caps, base_query_counts, is_prefill

    def _is_prefill(self, req_id: str) -> bool:
        req_index = self._req_states.req_id_to_index.get(req_id)
        return req_index is not None and (
            self._req_states.num_computed_prefill_tokens[req_index]
            < self._req_states.prefill_len.np[req_index]
        )
