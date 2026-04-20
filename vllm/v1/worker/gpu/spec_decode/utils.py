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
        self.copy_event = torch.cuda.Event()

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        # Per-request valid-draft-count, populated by speculators that
        # return variable-length drafts (e.g. NgramGPUSpeculator). ``None``
        # preserves legacy fixed-length semantics (Eagle / Medusa).
        self.num_valid_draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        num_valid_draft_tokens: torch.Tensor | None = None,
    ) -> None:
        """Stage the per-step draft tokens for async D2H copy.

        Args:
            input_batch: The just-executed step's input batch (used for
                ``req_ids`` and the structured-outputs flag).
            draft_tokens: ``[num_reqs, num_speculative_steps]`` int64 on
                device. Must be a final tensor (no further GPU mutation
                after this call) so the async copy is safe.
            num_valid_draft_tokens: Optional ``[num_reqs]`` int32 on device.
                Required for variable-length speculators so the scheduler
                can trim per-request drafts before the next scheduling
                round. When ``None`` (Eagle-style fixed-length drafts),
                all draft positions are treated as valid.
        """
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]

        needs_draft_copy = input_batch.has_structured_output_reqs
        needs_valid_copy = num_valid_draft_tokens is not None

        if not needs_draft_copy and not needs_valid_copy:
            # No downstream consumer for this step's drafts.
            self.draft_tokens_np = None
            self.num_valid_draft_tokens_np = None
            return

        # Kick off all D2H copies on a dedicated side stream so that they
        # overlap with draft-scatter and any subsequent main-stream work.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            if needs_draft_copy:
                self.draft_tokens_np = async_copy_to_np(draft_tokens)
            else:
                self.draft_tokens_np = None
            if needs_valid_copy:
                # ``async_copy_to_np`` already issues a non-blocking copy
                # and returns a NumPy view over the resulting pinned host
                # buffer. We synchronise on ``copy_event`` in
                # ``get_draft_tokens`` before materialising the list.
                self.num_valid_draft_tokens_np = async_copy_to_np(
                    num_valid_draft_tokens
                )
            else:
                self.num_valid_draft_tokens_np = None
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        # If either side-band was copied, we must sync the event exactly
        # once before reading the pinned buffers.
        if (
            self.draft_tokens_np is not None
            or self.num_valid_draft_tokens_np is not None
        ):
            self.copy_event.synchronize()

        if self.draft_tokens_np is not None:
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This branch is taken when async scheduling is disabled AND
            # there are no structured-output requests AND the speculator
            # did not emit per-request valid counts. The scheduler treats
            # an empty/``-1`` placeholder as "drafts were not echoed
            # back" — compatible with legacy Eagle behaviour.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]

        num_valid_list: list[int] | None = None
        if self.num_valid_draft_tokens_np is not None:
            # ``.tolist()`` on an int32 ndarray returns ``list[int]``.
            num_valid_list = self.num_valid_draft_tokens_np.tolist()

        return DraftTokenIds(
            req_ids=self.req_ids,
            draft_token_ids=draft_token_ids,
            num_valid_draft_tokens=num_valid_list,
        )
