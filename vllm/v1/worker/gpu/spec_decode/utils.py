# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import LoadConfig, replace
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

# Keep enough consumption-time snapshots to cover async PP in-flight
# batches (`max_concurrent_batches` is `pp_size + 1`) with headroom.
_DEFAULT_MAX_SNAPSHOTS = 8


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


@dataclass
class _DraftSnapshot:
    step_id: int | None
    req_ids: list[str]
    num_draft_tokens: int
    draft_tokens_np: np.ndarray | None
    copy_event: torch.cuda.Event | None

    def to_draft_token_ids(self) -> DraftTokenIds:
        if self.draft_tokens_np is not None:
            if self.copy_event is not None:
                self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


class DraftTokensHandler:
    def __init__(
        self,
        device: torch.device | None = None,
        max_snapshots: int = _DEFAULT_MAX_SNAPSHOTS,
    ):
        self.device = device
        self.copy_stream: torch.cuda.Stream | None = None
        if device is not None and device.type == "cuda":
            self.copy_stream = torch.cuda.Stream(device)

        self._max_snapshots = max(1, max_snapshots)
        # Consumption-time snapshots tagged with scheduler step identity.
        self._consumed: deque[_DraftSnapshot] = deque(maxlen=self._max_snapshots)
        # Latest proposal-time drafts for the non-async post_step path.
        self._proposed: _DraftSnapshot | None = None

    def _copy_drafts(
        self, draft_tokens: torch.Tensor
    ) -> tuple[np.ndarray, torch.cuda.Event | None]:
        if self.copy_stream is None:
            return draft_tokens.detach().cpu().numpy(), None

        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        copy_event = torch.cuda.Event(blocking=True)
        with torch.cuda.stream(self.copy_stream):
            draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and
            # read here on copy_stream; without record_stream, the caching
            # allocator may reuse its memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            copy_event.record()
        return draft_tokens_np, copy_event

    def snapshot_consumed_drafts(
        self,
        step_id: int,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
    ) -> None:
        """Record drafts that prepare_inputs fed into input_ids for this step.

        Tagged with the scheduler step so the engine can retrieve the matching
        snapshot later by equality. Batches without structured-output requests
        are not stored, so they cannot evict a useful snapshot.
        """
        if not input_batch.has_structured_output_reqs:
            return

        draft_tokens_np, copy_event = self._copy_drafts(draft_tokens)
        self._consumed.append(
            _DraftSnapshot(
                step_id=step_id,
                req_ids=list(input_batch.req_ids),
                num_draft_tokens=draft_tokens.shape[1],
                draft_tokens_np=draft_tokens_np,
                copy_event=copy_event,
            )
        )

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        req_ids = list(input_batch.req_ids)
        num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self._proposed = _DraftSnapshot(
                step_id=None,
                req_ids=req_ids,
                num_draft_tokens=num_draft_tokens,
                draft_tokens_np=None,
                copy_event=None,
            )
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        draft_tokens_np, copy_event = self._copy_drafts(draft_tokens)
        self._proposed = _DraftSnapshot(
            step_id=None,
            req_ids=req_ids,
            num_draft_tokens=num_draft_tokens,
            draft_tokens_np=draft_tokens_np,
            copy_event=copy_event,
        )

    def get_draft_tokens(self, step_id: int | None = None) -> DraftTokenIds | None:
        """Return draft tokens for grammar validation.

        If ``step_id`` is set, look up the consumption-time snapshot for that
        scheduler step by equality. A miss returns None so placeholders stay
        -1 and fail-closed invalidation can pin acceptance via is_valid_draft.
        If ``step_id`` is None, return the latest proposal-time drafts (the
        non-async post_step path).
        """
        if step_id is not None:
            for snapshot in reversed(self._consumed):
                if snapshot.step_id == step_id:
                    return snapshot.to_draft_token_ids()
            logger.debug(
                "Draft-token snapshot miss for scheduler step %s",
                step_id,
            )
            return None

        if self._proposed is None:
            return DraftTokenIds([], [])
        return self._proposed.to_draft_token_ids()


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
