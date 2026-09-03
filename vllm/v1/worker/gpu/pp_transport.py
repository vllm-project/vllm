# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Literal

import torch

from vllm.distributed.parallel_state import Handle, get_pp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton

PPTransportMode = Literal["stream", "fp8"]


@triton.jit
def _dequant_fp8_per_token_kernel(
    src_ptr,
    scale_ptr,
    dst_ptr,
    numel,
    row_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    values = tl.load(src_ptr + offsets, mask=mask).to(tl.float32)
    scales = tl.load(scale_ptr + offsets // row_width, mask=mask)
    tl.store(dst_ptr + offsets, values * scales, mask=mask)


def dequant_fp8_per_token(
    src: torch.Tensor,
    scales: torch.Tensor,
    dst: torch.Tensor,
) -> None:
    """Dequantize contiguous per-token FP8 into caller-owned BF16 storage."""
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: source {src.shape}, destination {dst.shape}")
    if src.dtype != current_platform.fp8_dtype():
        raise ValueError(f"Expected FP8 source, got {src.dtype}")
    if dst.dtype != torch.bfloat16:
        raise ValueError(f"Expected BF16 destination, got {dst.dtype}")
    if not (src.is_contiguous() and scales.is_contiguous() and dst.is_contiguous()):
        raise ValueError("PP transport tensors must be contiguous")

    row_width = src.shape[-1]
    expected_scale_shape = (*src.shape[:-1], 1)
    if scales.shape != expected_scale_shape:
        raise ValueError(
            f"Expected scale shape {expected_scale_shape}, got {scales.shape}"
        )

    block = 256
    _dequant_fp8_per_token_kernel[(triton.cdiv(src.numel(), block),)](
        src,
        scales,
        dst,
        src.numel(),
        row_width=row_width,
        BLOCK=block,
        num_warps=4,
    )


class PPTransport:
    """Fixed-schema PP transport backed by a bounded send-buffer ring."""

    def __init__(
        self,
        mode: PPTransportMode,
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

        self.mode = mode
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
        self._recv_quant: torch.Tensor | None = None
        self._recv_scale: torch.Tensor | None = None
        self._send_scale_ring: list[torch.Tensor] = []

        send_dtype = torch.bfloat16
        if mode == "fp8":
            send_dtype = current_platform.fp8_dtype()
        self._send_ring = (
            []
            if self._group.is_last_rank
            else [
                torch.empty(
                    self.hidden_shape,
                    dtype=send_dtype,
                    device=device,
                )
                for _ in range(ring_size)
            ]
        )

        scale_shape = (*self.hidden_shape[:-1], 1)
        if mode == "fp8":
            if not self._group.is_last_rank:
                self._send_scale_ring = [
                    torch.empty(scale_shape, dtype=torch.float32, device=device)
                    for _ in range(ring_size)
                ]
            if not self._group.is_first_rank:
                self._recv_quant = torch.empty(
                    self.hidden_shape,
                    dtype=current_platform.fp8_dtype(),
                    device=device,
                )
                self._recv_scale = torch.empty(
                    scale_shape,
                    dtype=torch.float32,
                    device=device,
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

        if self.mode == "stream":
            return (
                {"hidden_states": self._recv_buffer},
                [self._group.irecv_tensor(self._recv_buffer)],
                [],
            )

        assert self._recv_quant is not None
        assert self._recv_scale is not None
        handles = [
            self._group.irecv_tensor(self._recv_quant),
            self._group.irecv_tensor(self._recv_scale),
        ]

        def dequantize() -> None:
            assert self._recv_quant is not None
            assert self._recv_scale is not None
            assert self._recv_buffer is not None
            dequant_fp8_per_token(
                self._recv_quant,
                self._recv_scale,
                self._recv_buffer,
            )

        return {"hidden_states": self._recv_buffer}, handles, [dequantize]

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
        if self.mode == "fp8":
            from aiter import dynamic_per_token_scaled_quant

            send_scale = self._send_scale_ring[ring_index]
            dynamic_per_token_scaled_quant(
                send_buffer,
                hidden_states,
                send_scale,
            )
        else:
            send_scale = None
            send_buffer.copy_(hidden_states)

        ready_event = self._ready_events[ring_index]
        ready_event.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._send_stream):
            self._send_stream.wait_event(ready_event)
            handles = [self._group.isend_tensor(send_buffer)]
            if send_scale is not None:
                handles.append(self._group.isend_tensor(send_scale))
            self._send_work[ring_index] = handles

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
