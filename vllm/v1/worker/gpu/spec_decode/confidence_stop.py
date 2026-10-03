# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confidence-based early exit for autoregressive drafting."""

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_max_and_sumexp,
)


@triton.jit
def _draft_confidence_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    idx_mapping_ptr,
    draft_step_ptr,
    threshold_ptr,
    alive_ptr,
    num_alive_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    step = tl.load(draft_step_ptr)
    # Cudagraph-padded rows carry -1 and never keep the batch drafting.
    alive = req_state_idx >= 0
    if step > 0:
        alive = alive & (tl.load(alive_ptr + row) != 0)
    if alive:
        # The top-1 probability is exp(m - m) / sum(exp(x - m)) = 1 / sumexp.
        running_max = float("-inf")
        running_sumexp = 0.0
        row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
        for i in range(0, vocab_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            logits = tl.load(
                row_ptr + offsets, mask=offsets < vocab_size, other=float("-inf")
            ).to(tl.float32)
            block_max, block_sumexp = _compute_max_and_sumexp(logits)
            new_max = tl.maximum(running_max, block_max)
            running_sumexp = tl.where(
                running_max > float("-inf"),
                running_sumexp * tl.exp(running_max - new_max),
                0.0,
            ) + tl.where(
                block_max > float("-inf"),
                block_sumexp * tl.exp(block_max - new_max),
                0.0,
            )
            running_max = new_max
        top_prob = tl.where(running_sumexp > 0.0, 1.0 / running_sumexp, 0.0)
        alive = top_prob >= tl.load(threshold_ptr)
    tl.store(alive_ptr + row, alive.to(tl.int32))
    if alive:
        tl.atomic_add(num_alive_ptr, 1)


class DraftConfidenceStop:
    """Ends a draft round once no request's chain is confident enough to go on.

    A request's chain continues while every draft so far has a drafter top-1
    probability of at least the threshold; its first draft is kept regardless.
    The device keeps per-row chain state and the number of live chains, which
    the CPU reads after each draft step to decide whether to launch the next.
    Since all rows run every launched step, every request in the round ends up
    with the same number of verifiable drafts:

    * no chain alive after step j (0-based): j drafts, but at least one, since
      step j's drafts were all below the threshold;
    * all steps launched: num_steps drafts if a chain survived the last step,
      else num_steps - 1.

    The decision only reads the drafter's distributions, never the tokens drawn
    from them, so stopping does not bias rejection sampling.
    """

    def __init__(
        self,
        threshold: float,
        max_num_reqs: int,
        num_steps: int,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.threshold = torch.tensor([threshold], dtype=torch.float32, device=device)
        self.alive = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.num_alive = torch.zeros(1, dtype=torch.int32, device=device)
        self._num_alive_cpu = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._num_alive_ready = torch.cuda.Event()

        # Verifiable drafts per request-state slot, from the slot's last round.
        self._num_drafts_np = np.full(max_num_reqs, num_steps, dtype=np.int32)
        # Slots of a round whose last step's outcome is still on the device.
        self._pending_slots: np.ndarray | None = None
        self._round_slots: np.ndarray | None = None
        self._steps_launched = 0

    def update(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_step: torch.Tensor,
    ) -> None:
        """Fold this step's drafter distributions into the chain state.

        Graph-capturable: everything it touches is a persistent buffer.
        """
        num_rows, vocab_size = logits.shape
        self.num_alive.zero_()
        _draft_confidence_kernel[(num_rows,)](
            logits,
            logits.stride(0),
            vocab_size,
            idx_mapping,
            draft_step,
            self.threshold,
            self.alive,
            self.num_alive,
            BLOCK_SIZE=8192,
        )

    def begin_round(self, idx_mapping_np: np.ndarray) -> None:
        self._resolve_pending()
        self._round_slots = idx_mapping_np
        self._steps_launched = 0

    def full_round(self, idx_mapping_np: np.ndarray) -> None:
        """Record a round that drafts every step without checking confidence."""
        self._resolve_pending()
        self._num_drafts_np[idx_mapping_np] = self.num_steps
        self._round_slots = None

    def step_launched(self) -> None:
        """Start copying the number of live chains after the step just launched."""
        self._steps_launched += 1
        self._num_alive_cpu.copy_(self.num_alive, non_blocking=True)
        self._num_alive_ready.record()

    def _read_num_alive(self) -> int:
        with gpu_sync_allowed():
            self._num_alive_ready.synchronize()
        return int(self._num_alive_cpu[0])

    def should_continue(self) -> bool:
        """Whether any chain survived the last launched step. Waits for it."""
        if self._read_num_alive() > 0:
            return True
        self._end_round(max(1, self._steps_launched - 1))
        return False

    def end_round(self) -> None:
        """Close a round that launched every step without stopping early.

        Its last step's outcome is resolved when next needed, so the CPU does
        not wait for that step here.
        """
        if self._round_slots is None:
            return
        self._pending_slots = self._round_slots
        self._round_slots = None

    def _end_round(self, num_drafts: int) -> None:
        assert self._round_slots is not None
        self._num_drafts_np[self._round_slots] = num_drafts
        self._round_slots = None

    def _resolve_pending(self) -> None:
        if self._pending_slots is None:
            return
        num_drafts = self.num_steps
        if self._read_num_alive() == 0:
            num_drafts -= 1
        self._num_drafts_np[self._pending_slots] = num_drafts
        self._pending_slots = None

    def num_verifiable_drafts(self, idx_mapping_np: np.ndarray) -> np.ndarray:
        """Drafts each request may verify, from its slot's last draft round."""
        self._resolve_pending()
        return self._num_drafts_np[idx_mapping_np]
