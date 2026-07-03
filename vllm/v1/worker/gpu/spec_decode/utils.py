# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


def get_effective_scheduled_token_counts(
    scheduler_output: SchedulerOutput,
    req_id_to_index: dict[str, int],
    draft_token_capacity_np: np.ndarray,
) -> tuple[int, int]:
    num_tokens_per_req = scheduler_output.num_scheduled_tokens
    req_ids = tuple(num_tokens_per_req)
    num_reqs = len(req_ids)
    num_scheduled_tokens = np.fromiter(
        num_tokens_per_req.values(), dtype=np.int32, count=num_reqs
    )
    draft_tokens = scheduler_output.scheduled_spec_decode_tokens
    if draft_tokens:
        scheduled_draft_tokens_per_req = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        idx_mapping_np = np.fromiter(
            map(req_id_to_index.get, req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_scheduled_tokens -= scheduled_draft_tokens_per_req - np.minimum(
            scheduled_draft_tokens_per_req,
            draft_token_capacity_np[idx_mapping_np],
        )
    return int(num_scheduled_tokens.sum()), int(num_scheduled_tokens.max())


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        self.copy_event = torch.Event()

        self.req_ids: list[str] = []
        self.idx_mapping_np: np.ndarray | None = None
        self.draft_tokens_np: np.ndarray | None = None
        self.draft_token_capacity_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        self.copy_event_pending = False

    def _sync_copy(self) -> None:
        if self.copy_event_pending:
            self.copy_event.synchronize()
            self.copy_event_pending = False

    def _get_draft_token_capacities(self) -> np.ndarray | None:
        if self.draft_token_capacity_np is None:
            return None
        return np.clip(self.draft_token_capacity_np, 0, self.num_draft_tokens)

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        draft_token_capacity: torch.Tensor | None = None,
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.idx_mapping_np = input_batch.idx_mapping_np
        self.num_draft_tokens = draft_tokens.shape[1]
        self.draft_tokens_np = None
        self.draft_token_capacity_np = None
        self.copy_event_pending = False
        if not input_batch.has_structured_output_reqs and draft_token_capacity is None:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            return

        # Transfer only the scheduler metadata needed by the current batch. Draft
        # token ids are needed for structured output validation; DSpark capacity is
        # a small per-request int vector used by the CPU scheduler on later steps.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            if input_batch.has_structured_output_reqs:
                self.draft_tokens_np = async_copy_to_np(draft_tokens)
                # draft_tokens is a temporary allocation on the main stream and read
                # here on copy_stream; without record_stream, the caching allocator
                # may reuse its memory before the async copy executes.
                draft_tokens.record_stream(self.copy_stream)
            if draft_token_capacity is not None:
                self.draft_token_capacity_np = async_copy_to_np(draft_token_capacity)
                draft_token_capacity.record_stream(self.copy_stream)
            self.copy_event.record()
            self.copy_event_pending = True

    def sync_draft_token_capacities(
        self,
        draft_token_capacity_np: np.ndarray,
        req_id_to_index: dict[str, int],
    ) -> None:
        if self.draft_token_capacity_np is None:
            return
        self._sync_copy()
        draft_token_capacities = self._get_draft_token_capacities()
        assert draft_token_capacities is not None
        assert self.idx_mapping_np is not None
        active = np.isin(self.req_ids, tuple(req_id_to_index))
        draft_token_capacity_np[self.idx_mapping_np[active]] = draft_token_capacities[
            active
        ]

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            self._sync_copy()
            draft_token_capacities = self._get_draft_token_capacities()
            draft_token_ids = self.draft_tokens_np.tolist()
            if draft_token_capacities is not None:
                draft_token_ids = [
                    token_ids[:capacity]
                    for token_ids, capacity in zip(
                        draft_token_ids, draft_token_capacities
                    )
                ]
        else:
            # No token-id copy was needed. Keep placeholder lengths unchanged so
            # capacity-only batches do not block the scheduler on a D2H copy.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
