# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import GroupCoordinator, Handle, get_pp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


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


class PPActivationTransport:
    """Fixed-schema BF16 activation transport for V2 pipeline parallelism."""

    def __init__(
        self,
        schema: IntermediateTensors,
        chunk_tokens: int,
        ring_size: int,
        device: torch.device,
    ) -> None:
        if not schema.tensors:
            raise ValueError("PP activation schema cannot be empty")
        if ring_size < 1:
            raise ValueError(f"ring_size must be positive, got {ring_size}")

        self.keys = tuple(sorted(schema.tensors))
        self.shapes: dict[str, torch.Size] = {}
        for key in self.keys:
            tensor = schema.tensors[key]
            if tensor.shape[0] != chunk_tokens:
                raise ValueError(
                    f"Expected {chunk_tokens} rows for {key}, got {tensor.shape[0]}"
                )
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"Expected BF16 {key}, got {tensor.dtype}")
            if not tensor.is_contiguous():
                raise ValueError(f"Expected contiguous {key}")
            self.shapes[key] = tensor.shape

        self.chunk_tokens = chunk_tokens
        self.device = device
        self.group = get_pp_group()
        self.send_stream = torch.cuda.Stream(device=device)
        self.send_index = 0
        self.send_work: list[list[Handle] | None] = [None] * ring_size
        self.generic_send_work: list[list[Handle]] = []
        self.ready_events = [torch.cuda.Event() for _ in range(ring_size)]
        self.recv_tensors = (
            {}
            if self.group.is_first_rank
            else {key: schema.tensors[key] for key in self.keys}
        )
        self.send_ring = (
            []
            if self.group.is_last_rank
            else [
                {
                    key: torch.empty(
                        self.shapes[key],
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    for key in self.keys
                }
                for _ in range(ring_size)
            ]
        )

    def can_transfer(
        self,
        num_scheduled_tokens: int,
        all_gather_tensors: dict[str, bool],
    ) -> bool:
        return num_scheduled_tokens == self.chunk_tokens and all(
            all_gather_tensors.get(key, True) for key in self.keys
        )

    def receive(
        self,
    ) -> tuple[dict[str, torch.Tensor], list[Handle]]:
        if not self.recv_tensors:
            raise RuntimeError("The first PP rank cannot receive activations")
        handles = [self.group.irecv_tensor(self.recv_tensors[key]) for key in self.keys]
        return self.recv_tensors, handles

    def send(self, tensors: dict[str, torch.Tensor]) -> None:
        if not self.send_ring:
            raise RuntimeError("The last PP rank cannot send activations")
        if set(tensors) != set(self.keys):
            raise ValueError(
                f"Expected activation keys {self.keys}, got {tuple(sorted(tensors))}"
            )

        ring_index = self.send_index
        prior_work = self.send_work[ring_index]
        if prior_work is not None:
            for handle in prior_work:
                handle.wait()

        send_tensors = self.send_ring[ring_index]
        for key in self.keys:
            tensor = tensors[key]
            if tensor.shape != self.shapes[key] or tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"Unexpected {key} schema: shape={tensor.shape}, "
                    f"dtype={tensor.dtype}"
                )
            send_tensors[key].copy_(tensor)

        ready_event = self.ready_events[ring_index]
        ready_event.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.send_stream):
            self.send_stream.wait_event(ready_event)
            self.send_work[ring_index] = [
                self.group.isend_tensor(send_tensors[key]) for key in self.keys
            ]
        self.send_index = (ring_index + 1) % len(self.send_ring)

    def send_generic(
        self,
        tensor_dict: dict[str, torch.Tensor],
        all_gather_group: GroupCoordinator,
        all_gather_tensors: dict[str, bool],
    ) -> None:
        """Order a generic tail send after fixed sends on the send stream."""
        self._reap_generic_send_work()
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.send_stream):
            self.send_stream.wait_event(ready_event)
            self.generic_send_work.append(
                self.group.isend_tensor_dict(
                    tensor_dict,
                    all_gather_group=all_gather_group,
                    all_gather_tensors=all_gather_tensors,
                )
            )

    def _reap_generic_send_work(self) -> None:
        while self.generic_send_work:
            handles = self.generic_send_work[0]
            tensor_handles = handles[1:]
            if not tensor_handles or not all(
                handle.is_completed() for handle in tensor_handles
            ):
                break
            for handle in handles:
                handle.wait()
            self.generic_send_work.pop(0)

    def drain(self) -> None:
        for work in self.send_work:
            if work is not None:
                for handle in work:
                    handle.wait()
        self.send_work = [None] * len(self.send_work)
        for handles in self.generic_send_work:
            for handle in handles:
                handle.wait()
        self.generic_send_work.clear()
        self.group.drain_pending_isends()


def prepare_pp_intermediate_tensors(
    persistent: IntermediateTensors,
    received: IntermediateTensors,
    num_tokens: int,
    dummy_run: bool,
) -> IntermediateTensors:
    """Copy into V2 persistent buffers unless receive already targeted them."""
    tensors: dict[str, torch.Tensor] = {}
    for key, persistent_tensor in persistent.tensors.items():
        if dummy_run:
            dst = persistent_tensor[:num_tokens]
        else:
            src = received.tensors[key]
            dst = persistent_tensor[: src.shape[0]]
            if src.data_ptr() != dst.data_ptr():
                dst.copy_(src)
        tensors[key] = dst
    return IntermediateTensors(tensors)


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

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        self.activation_transport: PPActivationTransport | None = None

    def init_activation_transport(
        self,
        schema: IntermediateTensors,
        chunk_tokens: int,
    ) -> bool:
        descriptor = tuple(
            (
                key,
                tuple(tensor.shape),
                tensor.dtype,
                tensor.is_contiguous(),
            )
            for key, tensor in sorted(schema.tensors.items())
        )
        pp_group = get_pp_group()
        descriptors: list[object] = [None] * pp_group.world_size
        torch.distributed.all_gather_object(
            descriptors,
            descriptor,
            group=pp_group.cpu_group,
        )
        if any(peer_descriptor != descriptor for peer_descriptor in descriptors):
            logger.warning_once(
                "Disabling streamed PP transport because pipeline stages have "
                "different activation schemas."
            )
            return False
        if not descriptor or any(
            shape[0] != chunk_tokens or dtype != torch.bfloat16 or not is_contiguous
            for _, shape, dtype, is_contiguous in descriptor
        ):
            logger.warning_once(
                "Disabling streamed PP transport because its activation schema "
                "is not contiguous BF16 with %d rows.",
                chunk_tokens,
            )
            return False

        self.activation_transport = PPActivationTransport(
            schema=schema,
            chunk_tokens=chunk_tokens,
            ring_size=pp_group.world_size,
            device=self.device,
        )
        return True

    def drain_activation_transport(self) -> None:
        if self.activation_transport is not None:
            self.activation_transport.drain()

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
