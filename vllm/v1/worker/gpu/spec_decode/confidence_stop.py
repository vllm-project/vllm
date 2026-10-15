# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import numpy as np
import torch

from vllm.utils.gpu_sync_debug import gpu_sync_allowed


class DraftConfidenceStop:
    """Ends a single request's draft round once a draft's top-1 probability
    falls below the threshold; the first draft is always kept."""

    def __init__(
        self,
        threshold: float,
        fallback_depth: int,
        max_num_reqs: int,
        num_steps: int,
        device: torch.device,
    ):
        self.fallback_depth = fallback_depth
        self.num_steps = num_steps
        self.log_threshold = math.log(threshold)
        self.alive = torch.ones(1, dtype=torch.bool, device=device)
        self._alive_cpu = torch.ones(1, dtype=torch.bool, pin_memory=True)
        self._alive_ready = torch.cuda.Event()
        # Verifiable drafts per request-state slot, from its last draft round.
        self._num_drafts_np = np.full(max_num_reqs, num_steps, dtype=np.int32)
        self._slot: int | None = None
        # A round that ran every step; its last step is read when next needed.
        self._pending_slot: int | None = None
        self._steps_launched = 0

    def update(self, logits: torch.Tensor, draft_step: torch.Tensor) -> None:
        """Fold the first row's draft confidence into the chain (graph-safe)."""
        row = logits[:1].float()
        confident = (row.amax(-1) - row.logsumexp(-1)) >= self.log_threshold
        self.alive.copy_(confident & (self.alive | (draft_step == 0)))

    def begin_round(self, slot: int) -> None:
        self._resolve_pending()
        self._slot = slot
        self._steps_launched = 0

    def fixed_round(self, slots: np.ndarray, num_drafts: int) -> None:
        self._resolve_pending()
        self._num_drafts_np[slots] = num_drafts
        self._slot = None

    def step_launched(self) -> None:
        self._steps_launched += 1
        self._alive_cpu.copy_(self.alive, non_blocking=True)
        self._alive_ready.record()

    def _read_alive(self) -> bool:
        with gpu_sync_allowed():
            self._alive_ready.synchronize()
        return bool(self._alive_cpu[0])

    def should_continue(self) -> bool:
        """Whether the chain survived the last launched step. Waits for it."""
        if self._read_alive():
            return True
        assert self._slot is not None
        self._num_drafts_np[self._slot] = max(1, self._steps_launched - 1)
        self._slot = None
        return False

    def end_round(self) -> None:
        """Close a round that launched every step, without waiting for it."""
        self._pending_slot, self._slot = self._slot, None

    def _resolve_pending(self) -> None:
        if self._pending_slot is not None:
            survived = self._read_alive()
            self._num_drafts_np[self._pending_slot] = self.num_steps - (not survived)
            self._pending_slot = None

    def num_verifiable_drafts(self, slots: np.ndarray) -> np.ndarray:
        """Drafts each request may verify, from its slot's last draft round."""
        self._resolve_pending()
        return self._num_drafts_np[slots]
