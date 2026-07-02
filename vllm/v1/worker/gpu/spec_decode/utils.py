# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        self.copy_event = torch.Event()

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.lengths_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        self.lengths_np = None
        needs_copy = input_batch.has_structured_output_reqs or lengths is not None
        if not needs_copy:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs (and for per-request draft
        # lengths), transfer back to the scheduler asynchronously.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            if input_batch.has_structured_output_reqs:
                self.draft_tokens_np = async_copy_to_np(draft_tokens)
                # draft_tokens is a temporary allocation on the main stream and
                # read here on copy_stream; without record_stream, the caching
                # allocator may reuse its memory before the async copy executes.
                draft_tokens.record_stream(self.copy_stream)
            else:
                self.draft_tokens_np = None
            if lengths is not None:
                self.lengths_np = async_copy_to_np(lengths)
                lengths.record_stream(self.copy_stream)
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        lengths = None
        if self.lengths_np is not None or self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            if self.lengths_np is not None:
                lengths = [int(n) for n in self.lengths_np]
        if self.draft_tokens_np is not None:
            rows = self.draft_tokens_np.tolist()
            if lengths is not None:
                draft_token_ids = [row[:n] for row, n in zip(rows, lengths)]
            else:
                draft_token_ids = rows
        elif lengths is not None:
            # Per-request draft counts; ids stay GPU-side (-1 placeholders).
            draft_token_ids = [[-1] * n for n in lengths]
        else:
            # This case only happens when async scheduling is disabled.
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
