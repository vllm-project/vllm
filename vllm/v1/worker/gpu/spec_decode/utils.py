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
        self.num_valid_draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        num_valid_draft_tokens: torch.Tensor | None = None,
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]

        needs_draft_copy = input_batch.has_structured_output_reqs

        if not needs_draft_copy and num_valid_draft_tokens is None:
            self.draft_tokens_np = None
            self.num_valid_draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            # draft_tokens / num_valid_draft_tokens are temporary allocations on
            # the main stream and read here on copy_stream; without record_stream,
            # the caching allocator may reuse their memory before the async copy
            # executes.
            if needs_draft_copy:
                self.draft_tokens_np = async_copy_to_np(draft_tokens)
                draft_tokens.record_stream(self.copy_stream)
            else:
                self.draft_tokens_np = None
            if num_valid_draft_tokens is not None:
                self.num_valid_draft_tokens_np = async_copy_to_np(
                    num_valid_draft_tokens
                )
                num_valid_draft_tokens.record_stream(self.copy_stream)
            else:
                self.num_valid_draft_tokens_np = None
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if (
            self.draft_tokens_np is not None
            or self.num_valid_draft_tokens_np is not None
        ):
            self.copy_event.synchronize()

        if self.draft_tokens_np is not None:
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]

        num_valid_list: list[int] | None = None
        if self.num_valid_draft_tokens_np is not None:
            num_valid_list = self.num_valid_draft_tokens_np.tolist()

        return DraftTokenIds(
            req_ids=self.req_ids,
            draft_token_ids=draft_token_ids,
            num_valid_draft_tokens=num_valid_list,
        )


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
