# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.config import LoadConfig, replace
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
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


class DraftTokensHandler:
    # Upper bound on retained per-request drafts; normally pruned by discard()
    # as requests finish. Eviction is LRU (dict insertion order).
    _MAX_RETAINED = 4096
    # Fold pending async copies after this many unconsumed batches.
    _MAX_PENDING = 32

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        # Under PP a request is rescheduled pp_size steps after it was
        # sampled and other microbatches overwrite a single "latest batch"
        # slot in between, so a grammar fetch would return another batch's
        # drafts and this request's would stay placeholders. Keep the
        # latest drafts per request instead.
        self._latest: dict[str, list[int]] = {}
        # Async copies not yet folded into _latest (oldest first).
        self._pending: list[tuple[list[str], np.ndarray]] = []

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()
        self._pending.append((self.req_ids, self.draft_tokens_np))
        if len(self._pending) > self._MAX_PENDING:
            # Nobody is collecting (e.g. structured batches without a grammar
            # step in between); fold so this cannot grow unbounded.
            self._fold_pending()

    def _fold_pending(self) -> None:
        if not self._pending:
            return
        # One event on one stream: waiting for the latest record also covers
        # every earlier copy queued on copy_stream.
        self.copy_event.synchronize()
        for req_ids, drafts_np in self._pending:
            for req_id, row in zip(req_ids, drafts_np.tolist()):
                # Re-insert so dict order is least-recently-updated first.
                self._latest.pop(req_id, None)
                self._latest[req_id] = row
        self._pending.clear()
        while len(self._latest) > self._MAX_RETAINED:
            self._latest.pop(next(iter(self._latest)))

    def discard(self, req_id: str) -> None:
        """Forget a request's drafts (finished, preempted, or re-added).

        Also called for every NEW request (add_requests removes any prior
        state first), i.e. before the request has produced drafts, so this
        must only drop what exists NOW: fold the pending batches first,
        then pop.
        """
        self._fold_pending()
        self._latest.pop(req_id, None)

    def get_draft_tokens(self) -> DraftTokenIds | None:
        self._fold_pending()
        req_ids = list(self._latest.keys())
        draft_token_ids = [list(row) for row in self._latest.values()]
        if self.draft_tokens_np is None:
            # Latest batch had no structured-output requests: no validation
            # needed for it, hand back placeholders for its requests that
            # have no retained drafts (the only output when async scheduling
            # is disabled and nothing is structured).
            for req_id in self.req_ids:
                if req_id not in self._latest:
                    req_ids.append(req_id)
                    draft_token_ids.append([-1] * self.num_draft_tokens)
        return DraftTokenIds(req_ids, draft_token_ids)


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
