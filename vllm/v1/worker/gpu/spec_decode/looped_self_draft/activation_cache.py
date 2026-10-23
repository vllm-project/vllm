# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short-lived draft activations, indexed by vLLM request-state slot.

Only the immediately preceding proposal is reusable. Keeping request IDs on the
CPU and states on the GPU avoids synchronizing acceptance decisions to the CPU.
Graph writes skip negative request indices used for padding and dummy runs.
Rejection is handled by the next proposal's inputs,
which already start at the corrected position in the shared KV cache.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm.forward_context import get_forward_context
from vllm.triton_utils import tl, triton
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.looped_self_draft.speculator import (
        LoopedSelfDraftSpeculator,
    )


class LoopedSelfDraftActivationCache:
    def __init__(self, speculator: "LoopedSelfDraftSpeculator"):
        self.num_speculative_tokens = speculator.num_speculative_steps
        self.hidden_size = speculator.hidden_size
        self.idx_mapping = speculator.idx_mapping
        self.draft_step = speculator.current_draft_step
        shape = (speculator.max_num_reqs, self.num_speculative_tokens, self.hidden_size)
        kwargs = dict(device=speculator.device, dtype=speculator.dtype)
        self.input_hidden_states = torch.zeros(shape, **kwargs)
        self.hidden_states = torch.zeros(shape, **kwargs)
        self.verify_indices = torch.zeros(
            speculator.max_num_reqs, device=speculator.device, dtype=torch.int64
        )
        self.requests: set[str] = set()
        self.allowed = False
        self.verify = False
        self.prefix_slots: dict[str, torch.Tensor] = {}

    def invalidate(self) -> None:
        self.requests.clear()
        self.allowed = self.verify = False

    def can_reuse(self, scheduler_output: SchedulerOutput) -> bool:
        """Conservative gate, before graph dispatch, including async scheduling."""
        scheduled = scheduler_output.num_scheduled_tokens
        drafts = scheduler_output.scheduled_spec_decode_tokens
        invalid = set(scheduler_output.finished_req_ids)
        invalid.update(scheduler_output.preempted_req_ids or ())
        invalid.update(req.req_id for req in scheduler_output.scheduled_new_reqs)
        invalid.update(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
        return bool(scheduled) and all(
            req_id in self.requests
            and req_id not in invalid
            and count == self.num_speculative_tokens + 1
            and len(drafts.get(req_id, ())) == self.num_speculative_tokens
            for req_id, count in scheduled.items()
        )

    def begin_proposal(self, input_batch: InputBatch, dummy_run: bool) -> None:
        self.requests = set() if dummy_run else set(input_batch.req_ids)

    def record(
        self, input_hidden_states: torch.Tensor, hidden_states: torch.Tensor
    ) -> None:
        _record_draft_activations_kernel[(input_hidden_states.shape[0],)](
            input_hidden_states,
            hidden_states,
            self.input_hidden_states,
            self.hidden_states,
            self.idx_mapping,
            self.draft_step,
            self.hidden_size,
            self.num_speculative_tokens,
            BLOCK_SIZE=triton.next_power_of_2(self.hidden_size),
        )

    def merge(
        self, tail_input: torch.Tensor, tail_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = tail_input.shape[0] * (self.num_speculative_tokens + 1)
        input_hidden_states = torch.empty(
            (rows, self.hidden_size), device=tail_input.device, dtype=tail_input.dtype
        )
        hidden_states = torch.empty_like(input_hidden_states)
        _merge_draft_activations_kernel[(rows,)](
            self.input_hidden_states,
            self.hidden_states,
            tail_input,
            tail_state,
            input_hidden_states,
            hidden_states,
            self.verify_indices,
            self.hidden_size,
            self.num_speculative_tokens,
            BLOCK_SIZE=triton.next_power_of_2(self.hidden_size),
        )
        return input_hidden_states, hidden_states

    @contextmanager
    def prefix_context(self):
        """Attention KV stores need compact slots as well as compact metadata."""
        ctx = get_forward_context()
        old_slots, old_padding = ctx.slot_mapping, ctx.is_padding
        assert isinstance(old_slots, dict)
        ctx.slot_mapping = {**old_slots, **self.prefix_slots}
        if old_padding is not None:
            ctx.is_padding = old_padding[
                self.num_speculative_tokens :: self.num_speculative_tokens + 1
            ]
        try:
            yield
        finally:
            ctx.slot_mapping, ctx.is_padding = old_slots, old_padding


@triton.jit
def _record_draft_activations_kernel(
    input_hidden_states_ptr,
    hidden_states_ptr,
    saved_input_hidden_states_ptr,
    saved_hidden_states_ptr,
    idx_mapping_ptr,
    draft_step_ptr,
    HIDDEN_SIZE: tl.constexpr,
    NUM_DRAFTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + token_idx)
    if req_state_idx >= 0:
        step = tl.load(draft_step_ptr)
        hidden_offsets = tl.arange(0, BLOCK_SIZE)
        src_offsets = token_idx * HIDDEN_SIZE + hidden_offsets
        dst_offsets = (req_state_idx * NUM_DRAFTS + step) * HIDDEN_SIZE + hidden_offsets
        tl.store(
            saved_input_hidden_states_ptr + dst_offsets,
            tl.load(
                input_hidden_states_ptr + src_offsets, hidden_offsets < HIDDEN_SIZE, 0
            ),
            hidden_offsets < HIDDEN_SIZE,
        )
        tl.store(
            saved_hidden_states_ptr + dst_offsets,
            tl.load(hidden_states_ptr + src_offsets, hidden_offsets < HIDDEN_SIZE, 0),
            hidden_offsets < HIDDEN_SIZE,
        )


@triton.jit
def _merge_draft_activations_kernel(
    saved_input_hidden_states_ptr,
    saved_hidden_states_ptr,
    tail_input_hidden_states_ptr,
    tail_hidden_states_ptr,
    input_hidden_states_ptr,
    hidden_states_ptr,
    idx_mapping_ptr,
    HIDDEN_SIZE: tl.constexpr,
    NUM_DRAFTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_idx, step = token_idx // (NUM_DRAFTS + 1), token_idx % (NUM_DRAFTS + 1)
    hidden_offsets = tl.arange(0, BLOCK_SIZE)
    if step == NUM_DRAFTS:
        input_hidden_states = tl.load(
            tail_input_hidden_states_ptr + req_idx * HIDDEN_SIZE + hidden_offsets,
            hidden_offsets < HIDDEN_SIZE,
            0,
        )
        hidden_states = tl.load(
            tail_hidden_states_ptr + req_idx * HIDDEN_SIZE + hidden_offsets,
            hidden_offsets < HIDDEN_SIZE,
            0,
        )
    else:
        req_state_idx = tl.load(idx_mapping_ptr + req_idx)
        src_offsets = (req_state_idx * NUM_DRAFTS + step) * HIDDEN_SIZE + hidden_offsets
        input_hidden_states = tl.load(
            saved_input_hidden_states_ptr + src_offsets, hidden_offsets < HIDDEN_SIZE, 0
        )
        hidden_states = tl.load(
            saved_hidden_states_ptr + src_offsets, hidden_offsets < HIDDEN_SIZE, 0
        )
    tl.store(
        input_hidden_states_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        input_hidden_states,
        hidden_offsets < HIDDEN_SIZE,
    )
    tl.store(
        hidden_states_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        hidden_states,
        hidden_offsets < HIDDEN_SIZE,
    )
