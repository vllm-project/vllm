# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]


def compute_need_sampled_mask(
    input_batch: InputBatch, req_states: RequestState
) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    with outputs that might be needed in a subsequent (decode) step.
    Returns None if no sampled outputs are needed in the requests' next step."""

    idx_np = input_batch.idx_mapping_np
    old_computed = req_states.num_computed_tokens_np[idx_np]
    prefill_len = req_states.prefill_len.np[idx_np]
    max_seq_len = req_states.max_seq_len[idx_np]
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # Exclude requests that we know are finished.
    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
    need_sampled_mask = produces_sample & not_finishing
    return need_sampled_mask if need_sampled_mask.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_step_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, slots[k] holds step (curr_step - pp_size + k)'s
        # pending postprocess after that step's `receive`. None means no
        # postprocess is pending for that slot (e.g. broadcast was skipped).
        self.slots: list[PendingRecv | None] = (
            [] if self.is_last_rank else [None] * get_pp_group().world_size
        )
        self.slot_index = -1

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_step_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
        """Advance to this step's slot and wait for its recv event, then
        filter out entries whose request was freed since `receive`.
        """
        if not self.slots:
            return None
        self.slot_index = (self.slot_index + 1) % len(self.slots)
        slot = self.slots[self.slot_index]
        if slot is None:
            return None
        self.main_stream.wait_event(slot.event)
        self.slots[self.slot_index] = None

        # Skip indices whose request has been freed (and possibly reassigned)
        # since this `PendingRecv` was created.
        alive_mask = self.req_idx_gen_np[slot.idx_mapping_np] == slot.gen_at_receive_np
        if alive_mask.all():
            return dict(
                sampled_tokens=slot.sampled_tokens,
                num_sampled=slot.num_sampled,
                num_rejected=slot.num_rejected,
                idx_mapping=slot.idx_mapping,
            )
        if not alive_mask.any():
            return None

        alive_indices_np = np.flatnonzero(alive_mask)
        alive_indices = async_copy_to_gpu(alive_indices_np, device=self.device)
        return dict(
            sampled_tokens=slot.sampled_tokens[alive_indices],
            num_sampled=slot.num_sampled[alive_indices],
            num_rejected=slot.num_rejected[alive_indices],
            idx_mapping=slot.idx_mapping[alive_indices],
        )

    def receive(self, input_batch: InputBatch, req_states: RequestState) -> bool:
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch, req_states)
        if need_sampled_mask is None:
            self.slots[self.slot_index] = None
            return False

        # All requests decode next iff every request in the batch needs its
        # sampled output (i.e. no non-final prefill chunks or finishing reqs).
        all_decode_next = bool(need_sampled_mask.all())
        if all_decode_next:
            idx_mapping = input_batch.idx_mapping
            idx_mapping_np = input_batch.idx_mapping_np
        else:
            idx_mapping_np = input_batch.idx_mapping_np[need_sampled_mask]
            idx_mapping = torch.from_numpy(idx_mapping_np).to(
                self.device, non_blocking=True
            )

        # Snapshot the per-slot generation counter so a later free of any
        # of these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[idx_mapping_np]

        # The sender always broadcasts the full batch; receive into full-size
        # buffers and slice down to the subset that needs a deferred postprocess.
        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            if not all_decode_next:
                subset_np = np.flatnonzero(need_sampled_mask)
                subset = async_copy_to_gpu(subset_np, device=self.device)
                sampled_tokens = sampled_tokens[subset]
                combined = combined[:, subset]
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            event = self.broadcast_stream.record_event()
        self.slots[self.slot_index] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            idx_mapping,
            idx_mapping_np,
            gen_at_receive_np,
        )
        return all_decode_next

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch, req_states) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                sampled_token_ids.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
