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

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.req_ids = []
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        self.req_ids = input_batch.req_ids
        current_stream = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(current_stream)
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            self.copy_event.record()

    def take_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is None:
            return None

        self.copy_event.synchronize()
        return DraftTokenIds(self.req_ids, self.draft_tokens_np.tolist())
