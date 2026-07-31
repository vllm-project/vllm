# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    # Draft proposals for the step this slot feeds, when spec decoding is on.
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    with outputs that might be needed in a subsequent (decode) step.
    Returns None if no sampled outputs are needed in the requests' next step."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    max_seq_len = input_batch.max_seq_len_np
    assert max_seq_len is not None  # always populated under PP
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # Exclude requests that we know are finished.
    # The scheduler advances num_computed_tokens by the full scheduled width
    # (bonus + drafts) up front and only rolls the rejected part back in
    # update_from_output, which under PP runs after the next batch has already
    # been scheduled. Comparing the inflated count against max_seq_len would
    # mark a request as finishing up to num_draft tokens early, after which the
    # last rank stops broadcasting and the other ranks' last_sampled_tokens
    # freeze, silently repeating a stale token. Discount the drafts: a
    # redundant broadcast costs one small collective, a skipped one corrupts.
    finish_computed = old_computed
    if input_batch.num_draft_tokens_per_req is not None:
        finish_computed = old_computed - input_batch.num_draft_tokens_per_req
    not_finishing = np.maximum(finish_computed, prefill_len) + 1 < max_seq_len
    need_sampled_mask = produces_sample & not_finishing
    return need_sampled_mask if need_sampled_mask.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

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
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

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

    def get_prev_sampled_outputs(
        self,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        outputs = dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )
        if slot.draft_tokens is not None:
            # Scatter with the unfiltered mapping. A row whose request has since
            # been freed writes to a slot nobody reads, and add_requests zeroes
            # the row before any reuse, whereas the -1 sentinels in the filtered
            # `idx_mapping` would alias the last row.
            outputs["draft_update"] = (slot.draft_tokens, slot.idx_mapping)
        return outputs

    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        """Publish this step's proposals to the other PP ranks.

        Only the last rank runs the drafter, but the first rank owns
        embed_tokens and therefore builds the input embeddings for the whole
        pipeline. Without this the non-last ranks embed the scheduler's
        PLACEHOLDER_TOKEN_ID (-1) in the draft slots, and the last rank ends up
        verifying real proposals against logits computed from placeholders.
        """
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens.contiguous()
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )
            send.record_stream(self.broadcast_stream)

    def receive_drafts(self, input_batch: InputBatch) -> None:
        """Counterpart of `broadcast_drafts`.

        Attaches to the slot `receive` just pushed so the proposals are
        consumed at the same step as the sampled tokens they belong with. Both
        sides gate on the same mask, so the wire order always matches.
        """
        assert not self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            draft_tokens = torch.empty(
                num_reqs,
                self.num_speculative_steps,
                dtype=torch.int64,
                device=self.device,
            )
            torch.distributed.broadcast(
                draft_tokens, src=self.last_rank, group=self.broadcast_group
            )
            # Strictly later on broadcast_stream than the event `receive`
            # recorded, so it also covers the sampled tensors. Replacing the
            # slot's event keeps the consumer's single wait_event correct;
            # waiting on the older event would not order this broadcast.
            event = self.broadcast_stream.record_event()
            draft_tokens.record_stream(self.main_stream)
        slot = self.queue[-1]
        if slot is not None:
            slot.draft_tokens = draft_tokens
            slot.event = event

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

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
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            # receive() unconditionally allocates max_sample_len columns, but
            # the non-spec sampler path (num_draft_tokens == 0) returns width 1,
            # so an unpadded broadcast leaves the peer waiting on a larger count
            # than the root sends. NCCL does not diagnose the mismatch: the root
            # completes and the receiver hangs until the watchdog fires. Pad so
            # both sides agree. post_update reads each row with
            # sampled_tokens.stride(0) and stops at num_sampled, so the pad
            # columns are never observed.
            send_tokens = sampled_token_ids
            width = send_tokens.shape[-1]
            if width < self.max_sample_len:
                padded = send_tokens.new_zeros(
                    send_tokens.shape[0], self.max_sample_len
                )
                padded[:, :width] = send_tokens
                send_tokens = padded
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
