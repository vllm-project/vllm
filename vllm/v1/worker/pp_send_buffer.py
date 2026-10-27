# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated storage for a PP stage's outgoing intermediate tensors.

An NCCL send reads its source until the next stage posts the matching
receive, which is ordered after that stage's previous forward. Sending the
model outputs directly would make the next forward wait for the send, since it
may overwrite them: CUDA graph outputs are static, and relayed aux hidden
states alias the persistent receive buffer. Sending a snapshot in this buffer
lets the next forward run while the send is in flight; only the next snapshot
waits for it.
"""

from typing import Any

import torch

from vllm.distributed.parallel_state import GroupCoordinator, Handle

_ALIGNMENT = 256


def _layout(tensors: dict[str, Any]) -> tuple[dict[str, int], int]:
    """Aligned byte offset of each tensor in the buffer, and the bytes used."""
    offsets: dict[str, int] = {}
    end = 0
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            offsets[key] = -(-end // _ALIGNMENT) * _ALIGNMENT
            end = offsets[key] + value.nbytes
    return offsets, end


def packed_nbytes(tensors: dict[str, Any]) -> int:
    """Buffer bytes needed to send `tensors`."""
    return _layout(tensors)[1]


class PPSendBuffer:
    def __init__(self, nbytes: int, device: torch.device) -> None:
        self.buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
        self._send_work: list[Handle] = []

    def send(
        self,
        tensors: dict[str, Any],
        pp_group: GroupCoordinator,
        all_gather_group: GroupCoordinator | None,
    ) -> None:
        offsets, nbytes = _layout(tensors)
        devices = {tensors[k].device for k in offsets}
        if devices - {self.buffer.device}:
            raise RuntimeError(
                f"PP send tensors on {devices}, send buffer on {self.buffer.device}"
            )
        if nbytes > self.buffer.numel():
            shapes = {k: tuple(v.shape) for k, v in tensors.items() if k in offsets}
            raise RuntimeError(
                f"PP send of {nbytes} bytes {shapes} exceeds the "
                f"{self.buffer.numel()}-byte send buffer sized from the profile run"
            )
        # The previous snapshot's sends must have read the buffer. This is a
        # stream wait, so the CPU keeps going.
        for handle in self._send_work:
            handle.wait()
        staged: dict[str, Any] = {}
        for key, value in tensors.items():
            if key not in offsets:
                staged[key] = value
                continue
            start = offsets[key]
            view = self.buffer[start : start + value.nbytes]
            staged[key] = view.view(value.dtype).view(value.shape)
            staged[key].copy_(value)
        handles = pp_group.isend_tensor_dict(staged, all_gather_group=all_gather_group)
        # The metadata handle is reaped by the group coordinator.
        self._send_work = handles[1:]
