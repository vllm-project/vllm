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
    # Spec decode: proposed draft tokens relayed from the last rank, scattered
    # into the non-last rank's req_states.draft_tokens on consume. None when spec
    # is off (max_sample_len == 1).
    draft_tokens: torch.Tensor | None = None  # [num_reqs, max_sample_len - 1]


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
    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
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

        # Packed single-op broadcast layout (int64), per request row:
        #   [ sampled(max_sample_len) | num_sampled(1) | num_rejected(1)
        #     | draft(max_sample_len - 1) ]
        # One NCCL broadcast per step instead of three (sampled, combined,
        # draft) -- 3x fewer LL kernels spinning on the non-last ranks' side
        # stream. Width is static per deployment so send/recv element counts
        # always match.
        self.packed_width = 2 * self.max_sample_len + 1
        # Last rank, spec decode only: the packed buffer staged by broadcast()
        # (sampled + counts) awaiting the draft columns in broadcast_draft().
        self._staged_packed: torch.Tensor | None = None

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
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
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            draft_tokens=slot.draft_tokens,
        )

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
            # Single packed recv (see __init__ for the row layout). One NCCL op
            # per step instead of three.
            msl = self.max_sample_len
            packed = torch.empty(
                num_reqs, self.packed_width, dtype=torch.int64, device=self.device
            )
            torch.distributed.broadcast(
                packed, src=self.last_rank, group=self.broadcast_group
            )
            # Unpack. Keep downstream dtypes identical to the pre-packing code:
            # sampled int64 contiguous, counts int32, draft int64.
            sampled_tokens = packed[:, :msl].contiguous()
            num_sampled = packed[:, msl].to(torch.int32)
            num_rejected = packed[:, msl + 1].to(torch.int32)
            draft_tokens = packed[:, msl + 2 :] if msl > 1 else None
            event = self.broadcast_stream.record_event()
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream. `packed` covers the draft view.
            packed.record_stream(self.main_stream)
            sampled_tokens.record_stream(self.main_stream)
            num_sampled.record_stream(self.main_stream)
            num_rejected.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
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

        # Build the packed row (see __init__ for the layout) on the main stream.
        # The sampler may emit width < max_sample_len on steps with no draft
        # tokens (prefill, first decode); the unused sampled columns stay -1,
        # placeholders ignored by post_update, which advances each request by
        # its per-request num_sampled valid count. Draft columns (spec decode)
        # are filled by broadcast_draft() after propose(); the single NCCL op
        # is issued there so there is exactly ONE broadcast per step.
        msl = self.max_sample_len
        num_reqs = sampled_token_ids.shape[0]
        width = sampled_token_ids.shape[-1]
        assert width <= msl
        packed = torch.full(
            (num_reqs, self.packed_width), -1, dtype=torch.int64, device=self.device
        )
        packed[:, :width] = sampled_token_ids
        packed[:, msl] = num_sampled
        packed[:, msl + 1] = num_rejected
        if msl > 1:
            # Spec decode: stage; broadcast_draft() fills the draft columns and
            # sends. Assert the previous step's stage was flushed.
            assert self._staged_packed is None
            self._staged_packed = packed
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                packed, src=self.last_rank, group=self.broadcast_group
            )
            packed.record_stream(self.broadcast_stream)

    def broadcast_draft(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        """Fill the draft columns of the packed row staged by broadcast() and
        issue the step's SINGLE packed broadcast. Called AFTER propose() (the
        proposed draft tokens exist only then). Gated identically to
        `broadcast` so the staged/flush pairing stays matched. Without the
        draft relay, non-last ranks verify against a zero-init
        req_states.draft_tokens (near-zero acceptance, corrupt output)."""
        assert self.is_last_rank
        if self.max_sample_len <= 1:
            return
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs next step; `broadcast` staged
            # nothing, so there is nothing to send.
            assert self._staged_packed is None
            return
        packed = self._staged_packed
        assert packed is not None
        self._staged_packed = None
        # Fill draft columns on the main stream (propose()'s output lives there).
        packed[:, self.max_sample_len + 2 :] = draft_tokens
        with torch.cuda.stream(self.broadcast_stream):
            # wait_stream so the side-stream broadcast sees the draft fill.
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                packed, src=self.last_rank, group=self.broadcast_group
            )
            packed.record_stream(self.broadcast_stream)
