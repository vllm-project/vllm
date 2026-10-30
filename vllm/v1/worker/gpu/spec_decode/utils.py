# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple, cast

import numpy as np
import torch

from vllm.config import LoadConfig, replace
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


def get_drafter_hidden_states(
    model: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Hidden states fed to the drafter.

    Targets such as DeepSeek V4 expose the pre-hc_head residual through
    get_mtp_target_hidden_states(). The buffer is sized at
    max_num_batched_tokens and only allocated for drafters that consume
    target hidden states, so None means "use the regular hidden states".
    """
    get_target_hidden_states = getattr(model, "get_mtp_target_hidden_states", None)
    if get_target_hidden_states is None:
        return hidden_states
    target_hidden_states = get_target_hidden_states()
    if target_hidden_states is None:
        return hidden_states
    return target_hidden_states[: hidden_states.shape[0]]


def get_pp_safe_draft_load_config(load_config: LoadConfig) -> LoadConfig:
    """Avoid collectives that include PP ranks without a draft model."""
    if get_pp_group().world_size > 1 and load_config.load_format == "fastsafetensors":
        logger.warning_once(
            "fastsafetensors cannot load a draft model instantiated on only "
            "one pipeline stage; falling back to the standard safetensors "
            "loader for the draft model. The target model still uses "
            "fastsafetensors."
        )
        return replace(load_config, load_format="auto")
    return load_config


class CopyBarrier(NamedTuple):
    event: torch.cuda.Event
    blocked_reqs: set[str]


class DraftTokens(NamedTuple):
    token_ids: np.ndarray | list[int]
    copy_barrier: CopyBarrier | None


class DraftTokensHandler:
    def __init__(self, max_concurrent_batches: int, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self._copy_barriers: list[CopyBarrier] = [
            CopyBarrier(event=torch.cuda.Event(blocking=True), blocked_reqs=set())
            for _ in range(max_concurrent_batches)
        ]
        self._next_copy_barrier = 0

        # Latest drafts per live request: a row view into an async-copied
        # pinned array (synced in get_draft_tokens), or list of draft token ids
        # when copy is complete.
        self._draft_tokens_by_req: dict[str, DraftTokens] = {}
        # Requests whose drafts the scheduler will fetch for the latest step's
        # deferred grammar bitmask; None means all requests.
        self._deferred_req_ids: list[str] | None = None

    def begin_step(self, scheduler_output: SchedulerOutput) -> None:
        # get_draft_tokens is only called for the most recently executed
        # scheduler output.
        self._deferred_req_ids = (
            list(scheduler_output.scheduled_spec_decode_tokens)
            if scheduler_output.pending_structured_output_tokens
            else None
        )

    def discard(self, req_id: str) -> None:
        draft_tokens = self._draft_tokens_by_req.pop(req_id, None)
        if draft_tokens is not None and draft_tokens.copy_barrier is not None:
            draft_tokens.copy_barrier.blocked_reqs.discard(req_id)

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        for req_id in input_batch.req_ids:
            self.discard(req_id)

        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch. Record draft token widths only.
            num_draft_tokens = draft_tokens.shape[1]
            for req_id in input_batch.req_ids:
                self._draft_tokens_by_req[req_id] = DraftTokens(
                    token_ids=[-1] * num_draft_tokens,
                    copy_barrier=None,
                )
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        copy_barrier = self._copy_barriers[self._next_copy_barrier]
        self._next_copy_barrier = (self._next_copy_barrier + 1) % len(
            self._copy_barriers
        )
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            event, blocked_reqs = copy_barrier
            event.record()
            blocked_reqs.update(input_batch.req_ids)

        for req_id, row in zip(input_batch.req_ids, draft_tokens_np):
            self._draft_tokens_by_req[req_id] = DraftTokens(
                token_ids=row, copy_barrier=copy_barrier
            )

    def get_draft_tokens(self) -> DraftTokenIds | None:
        """Return the latest drafts needed for the current step's deferred
        grammar bitmask, or for all live requests if sampling isn't deferred.
        Requests without retained drafts are omitted."""
        returned_req_ids: list[str] = []
        draft_token_ids: list[list[int]] = []
        request_ids = (
            self._deferred_req_ids
            if self._deferred_req_ids is not None
            else list(self._draft_tokens_by_req)
        )
        for req_id in request_ids:
            draft_tokens = self._draft_tokens_by_req[req_id]
            if draft_tokens.copy_barrier is None:
                # Copy is already complete.
                draft_token_ids.append(cast(list[int], draft_tokens.token_ids))
            else:
                # Wait for the copy to complete and update all blocked requests.
                event, blocked_reqs = draft_tokens.copy_barrier
                event.synchronize()
                for blocked_req_id in blocked_reqs:
                    token_ids, _ = self._draft_tokens_by_req[blocked_req_id]
                    token_ids = cast(np.ndarray, token_ids).tolist()
                    self._draft_tokens_by_req[blocked_req_id] = DraftTokens(
                        token_ids=token_ids, copy_barrier=None
                    )
                    if blocked_req_id == req_id:
                        draft_token_ids.append(token_ids)
                blocked_reqs.clear()
            returned_req_ids.append(req_id)

        return DraftTokenIds(returned_req_ids, draft_token_ids)


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
