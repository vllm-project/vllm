# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import envs
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event | None

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    combined: torch.Tensor  # [2, num_reqs]: num_sampled, num_rejected
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
    ):
        pp_group = get_pp_group()
        self.is_last_rank = pp_group.is_last_rank
        self.last_rank = pp_group.last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        self.requested_recv_launch_delay = envs.VLLM_PP_DEFER_SAMPLED_TOKEN_RECV
        self.post_model_recv_launch = envs.VLLM_PP_POST_MODEL_SAMPLED_TOKEN_RECV
        self.collect_recv_wait_stats = envs.VLLM_PP_DEFER_SAMPLED_TOKEN_RECV_STATS
        if self.post_model_recv_launch and not self.requested_recv_launch_delay:
            raise ValueError(
                "VLLM_PP_POST_MODEL_SAMPLED_TOKEN_RECV requires a non-zero "
                "VLLM_PP_DEFER_SAMPLED_TOKEN_RECV"
            )
        if not 0 <= self.requested_recv_launch_delay < pp_group.world_size:
            raise ValueError(
                "VLLM_PP_DEFER_SAMPLED_TOKEN_RECV must satisfy "
                "0 <= delay < pp_size; got "
                f"delay={self.requested_recv_launch_delay}, "
                f"pp_size={pp_group.world_size}"
            )
        if self.requested_recv_launch_delay and not current_platform.is_cuda():
            raise ValueError(
                "VLLM_PP_DEFER_SAMPLED_TOKEN_RECV is currently supported only on CUDA"
            )
        # Warmup must retain upstream collective timing. The worker enables
        # the requested delay only after all compilation and warmup completes.
        # The last PP rank always broadcasts immediately.
        self.recv_launch_delay = 0
        self.deferred_collectives_active = False

        # Diagnostics are collected only when deferral is active. They make
        # idle-boundary flushes and late receives visible in benchmark logs.
        self.num_deferred_recv_launches = 0
        self.num_idle_boundaries = 0
        self.num_idle_flushes = 0
        self.num_idle_flushed_receives = 0
        self.num_consume_fallbacks = 0
        self.num_unready_at_consume = 0
        self.num_measured_consume_waits = 0
        self.total_consume_wait_ms = 0.0
        self.max_consume_wait_ms = 0.0
        self.pending_wait_events: deque[tuple[torch.cuda.Event, torch.cuda.Event]] = (
            deque()
        )
        self.pending_post_model_receive: PendingRecv | None = None

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * pp_group.world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = pp_group.make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        self.aux_hidden_state_relay_keys: tuple[str, ...] = ()

    def enable_deferred_collectives(self) -> bool:
        """Enable the requested receive delay after worker warmup."""
        if not self.requested_recv_launch_delay or self.deferred_collectives_active:
            return False

        self.recv_launch_delay = (
            0 if self.is_last_rank else self.requested_recv_launch_delay
        )
        self.deferred_collectives_active = True
        pp_group = get_pp_group()
        logger.info_once(
            "Enabled deferred PP sampled-token receive: "
            "pp_rank=%d/%d recv_delay_steps=%d post_model_launch=%s",
            pp_group.rank_in_group,
            pp_group.world_size,
            self.recv_launch_delay,
            self.post_model_recv_launch,
        )
        return True

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def configure_aux_hidden_state_relay(self, model: torch.nn.Module) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            aux_hidden_state_relay_keys,
        )

        self.aux_hidden_state_relay_keys = aux_hidden_state_relay_keys(model)

    def relay_aux_hidden_states(
        self,
        intermediate_tensors: IntermediateTensors | None,
        output_intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        if not self.aux_hidden_state_relay_keys:
            return output_intermediate_tensors
        assert intermediate_tensors is not None
        return IntermediateTensors(
            output_intermediate_tensors.tensors
            | {
                key: intermediate_tensors[key]
                for key in self.aux_hidden_state_relay_keys
            }
        )

    def _launch_receive(self, slot: PendingRecv) -> None:
        """Post one receiver's broadcasts exactly once."""
        if slot.event is not None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                slot.sampled_tokens,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            torch.distributed.broadcast(
                slot.combined,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            if slot.draft_tokens is not None:
                torch.distributed.broadcast(
                    slot.draft_tokens,
                    src=self.last_rank,
                    group=self.broadcast_group,
                )
            slot.event = self.broadcast_stream.record_event()
            # The tensors are populated on the broadcast stream and consumed
            # later on the main stream.
            slot.sampled_tokens.record_stream(self.main_stream)
            slot.combined.record_stream(self.main_stream)
            if slot.draft_tokens is not None:
                slot.draft_tokens.record_stream(self.main_stream)
        if self.recv_launch_delay:
            self.num_deferred_recv_launches += 1

    def _advance_receive_queue(self) -> PendingRecv | None:
        """Select the configured slot and return the slot due this step."""
        if self.recv_launch_delay:
            launch_slot = self.queue[-self.recv_launch_delay]
            if launch_slot is not None:
                if self.post_model_recv_launch:
                    if self.pending_post_model_receive is not None:
                        raise RuntimeError(
                            "Previous post-model sampled-token receive was not launched"
                        )
                    self.pending_post_model_receive = launch_slot
                else:
                    self._launch_receive(launch_slot)

        due_slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        return due_slot

    def _queue_receive(self, slot: PendingRecv) -> None:
        """Associate a receive with the current step and launch if immediate."""
        self.queue[-1] = slot
        if self.recv_launch_delay == 0:
            self._launch_receive(slot)

    def _ensure_receive_launched(self, slot: PendingRecv) -> None:
        if slot.event is not None:
            return
        # Defensive fallback for a changed cadence or an early finite drain:
        # never consume a slot without posting its collective.
        logger.warning_once(
            "Deferred PP sampled-token receive reached consumption before "
            "its configured launch step; posting it now"
        )
        self.num_consume_fallbacks += 1
        self._launch_receive(slot)

    def launch_post_model_receive(self) -> bool:
        """Post the selected receive behind this step's model kernels."""
        if not self.post_model_recv_launch or self.is_last_rank:
            return False
        slot = self.pending_post_model_receive
        if slot is None:
            return False
        self.pending_post_model_receive = None
        self._launch_receive(slot)
        return True

    def flush_pending_collectives(self, reason: str = "explicit") -> int:
        """Post all deferred receives before an idle boundary or shutdown."""
        if self.is_last_rank:
            return 0

        if reason == "idle" and self.recv_launch_delay:
            self.num_idle_boundaries += 1

        launched = 0
        if self.pending_post_model_receive is not None:
            slot = self.pending_post_model_receive
            self.pending_post_model_receive = None
            self._launch_receive(slot)
            launched += 1
        for slot in self.queue:
            if slot is not None and slot.event is None:
                self._launch_receive(slot)
                launched += 1
        if reason == "idle" and launched:
            self.num_idle_flushes += 1
            self.num_idle_flushed_receives += launched
        return launched

    def _collect_completed_wait_timings(self, force: bool = False) -> None:
        while self.pending_wait_events:
            start_event, end_event = self.pending_wait_events[0]
            if not force and not end_event.query():
                break
            self.pending_wait_events.popleft()
            elapsed_ms = start_event.elapsed_time(end_event)
            self.num_measured_consume_waits += 1
            self.total_consume_wait_ms += elapsed_ms
            self.max_consume_wait_ms = max(self.max_consume_wait_ms, elapsed_ms)

    def _wait_for_receive(self, event: torch.cuda.Event) -> None:
        """Wait on the main stream and measure a genuinely unready receive."""
        if self.collect_recv_wait_stats:
            # Keep diagnostic runs bounded instead of retaining two CUDA events
            # for every historical hard wait until shutdown.
            self._collect_completed_wait_timings()
        if (
            self.recv_launch_delay
            and self.collect_recv_wait_stats
            and not event.query()
        ):
            self.num_unready_at_consume += 1
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(self.main_stream)
            self.main_stream.wait_event(event)
            end_event.record(self.main_stream)
            self.pending_wait_events.append((start_event, end_event))
            return
        self.main_stream.wait_event(event)

    def log_deferred_receive_stats(self) -> None:
        """Log end-of-run diagnostics after the device has synchronized."""
        if self.is_last_rank or not self.deferred_collectives_active:
            return
        if self.collect_recv_wait_stats:
            self._collect_completed_wait_timings(force=True)
        logger.info(
            "Deferred PP sampled-token receive stats: launches=%d, "
            "idle_boundaries=%d, idle_flushes=%d, idle_flushed_receives=%d, "
            "consume_fallbacks=%d, unready_at_consume=%d, "
            "measured_consume_waits=%d, total_consume_wait_ms=%.3f, "
            "max_consume_wait_ms=%.3f, wait_timing_enabled=%s",
            self.num_deferred_recv_launches,
            self.num_idle_boundaries,
            self.num_idle_flushes,
            self.num_idle_flushed_receives,
            self.num_consume_fallbacks,
            self.num_unready_at_consume,
            self.num_measured_consume_waits,
            self.total_consume_wait_ms,
            self.max_consume_wait_ms,
            self.collect_recv_wait_stats,
        )

    def get_prev_sampled_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self._advance_receive_queue()
        if slot is None:
            return None

        self._ensure_receive_launched(slot)
        assert slot.event is not None

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

        self._wait_for_receive(slot.event)
        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            draft_tokens = slot.draft_tokens
            draft_idx_mapping = slot.idx_mapping
            if exclude_mask.any():
                keep = ~exclude_mask
                keep_t = torch.as_tensor(keep, device=self.device)
                draft_tokens = draft_tokens[keep_t]
                draft_idx_mapping = async_copy_to_gpu(
                    slot.idx_mapping_np[keep], device=self.device
                )
            draft_tokens_to_update[draft_idx_mapping] = draft_tokens
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        """Broadcast draft proposals so non-last ranks can embed real token ids."""
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens[input_batch.idx_mapping].contiguous()
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )
            send.record_stream(self.broadcast_stream)

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        # The source uses the same predicate in `broadcast`, which preserves
        # collective skip/order symmetry across PP ranks.
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            # Preserve upstream stream/allocation ordering even when launch is
            # deferred. The launch path waits again at T+delay so the receive
            # cannot race newer main-stream work.
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            num_sampled, num_rejected = combined.unbind(dim=0)
            draft_tokens = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
            event = None
            if self.recv_launch_delay == 0:
                # Keep the default path's ordering identical to upstream.
                torch.distributed.broadcast(
                    sampled_tokens, src=self.last_rank, group=self.broadcast_group
                )
                torch.distributed.broadcast(
                    combined, src=self.last_rank, group=self.broadcast_group
                )
                if draft_tokens is not None:
                    torch.distributed.broadcast(
                        draft_tokens,
                        src=self.last_rank,
                        group=self.broadcast_group,
                    )
                event = self.broadcast_stream.record_event()
            # These tensors remain live in the queue until the main stream
            # consumes them, including across a deferred launch.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        slot = PendingRecv(
            event,
            sampled_tokens,
            combined,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        self._queue_receive(slot)
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        # Keep this predicate identical to the receiver-side check above.
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids,
                (0, self.max_sample_len - sampled_token_ids.shape[-1]),
            )
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
