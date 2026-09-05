# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.distributed.parallel_state import Handle, get_pp_group
from vllm.sequence import IntermediateTensors


class PPTransport:
    """Fixed-schema V1 PP transport backed by a bounded BF16 send-buffer ring."""

    def __init__(
        self,
        schema: IntermediateTensors,
        chunk_tokens: int,
        ring_size: int,
        device: torch.device,
    ) -> None:
        if set(schema.tensors) != {"hidden_states"}:
            raise ValueError(
                "ROCm PP transport currently requires a single hidden_states tensor"
            )
        hidden_states = schema.tensors["hidden_states"]
        if hidden_states.shape[0] != chunk_tokens:
            raise ValueError(
                f"Expected {chunk_tokens} schema tokens, got {hidden_states.shape[0]}"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                "ROCm PP transport requires BF16 hidden states, "
                f"got {hidden_states.dtype}"
            )
        if not hidden_states.is_contiguous():
            raise ValueError("ROCm PP transport requires contiguous hidden states")
        if ring_size < 1:
            raise ValueError(f"ring_size must be positive, got {ring_size}")

        self.chunk_tokens = chunk_tokens
        self.hidden_shape = hidden_states.shape
        self.device = device
        self._group = get_pp_group()
        self._send_stream = torch.cuda.Stream(device=device)
        self._send_index = 0
        self._send_work: list[list[Handle] | None] = [None] * ring_size
        self._generic_send_work: list[list[Handle]] = []
        self._ready_events = [torch.cuda.Event() for _ in range(ring_size)]

        self._recv_buffer = (
            None if self._group.is_first_rank else hidden_states
        )
        self._send_ring = (
            []
            if self._group.is_last_rank
            else [
                torch.empty(
                    self.hidden_shape,
                    dtype=torch.bfloat16,
                    device=device,
                )
                for _ in range(ring_size)
            ]
        )

    def can_transfer(self, num_scheduled_tokens: int) -> bool:
        return num_scheduled_tokens == self.chunk_tokens

    def receive(
        self,
    ) -> tuple[
        dict[str, torch.Tensor],
        list[Handle],
        list[Callable[[], None]],
    ]:
        if self._recv_buffer is None:
            raise RuntimeError("The first PP rank cannot receive activations")
        return (
            {"hidden_states": self._recv_buffer},
            [self._group.irecv_tensor(self._recv_buffer)],
            [],
        )

    def send(self, hidden_states: torch.Tensor) -> None:
        if not self._send_ring:
            raise RuntimeError("The last PP rank cannot send activations")
        if hidden_states.shape != self.hidden_shape:
            raise ValueError(
                f"Expected hidden state shape {self.hidden_shape}, "
                f"got {hidden_states.shape}"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 hidden states, got {hidden_states.dtype}")

        ring_index = self._send_index
        prior_work = self._send_work[ring_index]
        if prior_work is not None:
            for handle in prior_work:
                handle.wait()

        send_buffer = self._send_ring[ring_index]
        send_buffer.copy_(hidden_states)

        ready_event = self._ready_events[ring_index]
        ready_event.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._send_stream):
            self._send_stream.wait_event(ready_event)
            self._send_work[ring_index] = [
                self._group.isend_tensor(send_buffer),
            ]

        self._send_index = (ring_index + 1) % len(self._send_ring)

    def send_generic(
        self,
        tensor_dict: dict[str, torch.Tensor],
        all_gather_group,
        all_gather_tensors: dict[str, bool],
    ) -> None:
        """Order a generic tail send after fixed sends on the send stream."""
        self._reap_generic_send_work()
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._send_stream):
            self._send_stream.wait_event(ready_event)
            self._generic_send_work.append(
                self._group.isend_tensor_dict(
                    tensor_dict,
                    all_gather_group=all_gather_group,
                    all_gather_tensors=all_gather_tensors,
                )
            )

    def _reap_generic_send_work(self) -> None:
        while self._generic_send_work:
            handles = self._generic_send_work[0]
            tensor_handles = handles[1:]
            if not tensor_handles or not all(
                handle.is_completed() for handle in tensor_handles
            ):
                break
            for handle in handles:
                handle.wait()
            self._generic_send_work.pop(0)

    def drain(self) -> None:
        for work in self._send_work:
            if work is not None:
                for handle in work:
                    handle.wait()
        self._send_work = [None] * len(self._send_work)
        for handles in self._generic_send_work:
            for handle in handles:
                handle.wait()
        self._generic_send_work.clear()
