# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel sampled-token synchronization for the CPU runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.worker.cpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    sampled_tokens: torch.Tensor  # [num_reqs, 1]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    with outputs that might be needed in a subsequent (decode) step.
    Returns None if no sampled outputs are needed in the requests' next step."""
    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    need_sampled_mask = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    if input_batch.max_seq_len_np is not None:
        # Also exclude final prefill chunks whose single sampled token reaches
        # the request's length cap.
        finished_prefill = prefill_len + 1 >= input_batch.max_seq_len_np
        finished_prefill &= input_batch.is_prefilling_np
        need_sampled_mask &= ~finished_prefill
    return need_sampled_mask if need_sampled_mask.any() else None


class PPHandler:
    """Synchronize sampled tokens from the last pipeline rank on CPU."""

    def __init__(self, max_num_reqs: int, device: torch.device):
        pp_group = get_pp_group()
        self.is_last_rank = pp_group.is_last_rank
        self.last_rank = pp_group.last_rank
        self.device = device

        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * pp_group.world_size)
        )
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)
        self.broadcast_group = pp_group.make_sibling_device_group(
            group_desc="pp_broadcast"
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
        if not self.queue:
            return None

        slot = self.queue.popleft()
        self.queue.append(None)
        if slot is None:
            return None

        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                return None
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = torch.from_numpy(idx_mapping_np).to(device=self.device)

        return {
            "sampled_tokens": slot.sampled_tokens,
            "num_sampled": slot.num_sampled,
            "num_rejected": slot.num_rejected,
            "idx_mapping": idx_mapping,
        }

    def receive(self, input_batch: InputBatch) -> bool:
        """Receive this step's sampled-token data on a non-last rank."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        sampled_tokens = torch.empty(num_reqs, 1, dtype=torch.int64, device=self.device)
        combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(
            sampled_tokens, src=self.last_rank, group=self.broadcast_group
        )
        torch.distributed.broadcast(
            combined, src=self.last_rank, group=self.broadcast_group
        )
        num_sampled, num_rejected = combined.unbind(dim=0)

        self.queue[-1] = PendingRecv(
            sampled_tokens=sampled_tokens,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            idx_mapping=input_batch.idx_mapping,
            idx_mapping_np=input_batch.idx_mapping_np,
            need_sampled_mask=need_sampled_mask,
            gen_at_receive_np=gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        """Broadcast this step's sampled-token data from the last rank."""
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return

        assert sampled_token_ids.dtype == torch.int64
        assert sampled_token_ids.shape[-1] == 1
        torch.distributed.broadcast(
            sampled_token_ids.contiguous(),
            src=self.last_rank,
            group=self.broadcast_group,
        )
        combined = torch.stack((num_sampled, num_rejected), dim=0)
        torch.distributed.broadcast(
            combined, src=self.last_rank, group=self.broadcast_group
        )
