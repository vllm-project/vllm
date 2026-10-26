# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP transport over NCCL send/recv."""

from collections.abc import Callable
from typing import Any

import torch

from vllm.distributed.parallel_state import GroupCoordinator, Handle
from vllm.distributed.pp_transport import PPTransport
from vllm.sequence import IntermediateTensors


class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization"""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        comm_handles: list[Handle] | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        super().__init__(tensors)
        self._comm_handles = comm_handles
        self._comm_postprocess = comm_postprocess
        self._comm_waited = False

    def wait_for_comm(self) -> None:
        if self._comm_waited:
            return
        if self._comm_handles:
            for handle in self._comm_handles:
                handle.wait()
        if self._comm_postprocess:
            for fn in self._comm_postprocess:
                fn()
        self._comm_waited = True

    def __getattribute__(self, name: str):
        # ensure `.tensors` is ready before use
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)


class NcclTransport(PPTransport):
    """NCCL send/recv. A send reads its source until the next stage posts the
    matching receive, so with ``profile_send`` it sends from a ring of two
    buffers sized like the profile run's outputs: the next forward can then
    overwrite its outputs at once, and only waits when a slot comes around
    again. Without it, the next forward waits for the previous send."""

    def __init__(
        self,
        all_gather_group: GroupCoordinator,
        profile_send: dict[str, Any] | None,
    ) -> None:
        self.all_gather_group = all_gather_group
        self._slots: list[dict[str, torch.Tensor]] = []
        if profile_send is not None:
            self._slots = [
                {
                    k: torch.empty_like(v)
                    for k, v in profile_send.items()
                    if isinstance(v, torch.Tensor)
                }
                for _ in range(2)
            ]
        # Allocated after the profile run's peak, which absorbs them.
        self.unprofiled_bytes = sum(
            t.nbytes for slot in self._slots for t in slot.values()
        )
        self._slot_work: list[list[Handle]] = [[] for _ in self._slots]
        self._next_slot = 0
        self._send_work: list[Handle] = []

    def before_forward(self) -> None:
        for handle in self._send_work:
            handle.wait()
        self._send_work = []

    def send(self, tensors: dict[str, Any], pp_group: GroupCoordinator) -> None:
        if not self._slots:
            handles = pp_group.isend_tensor_dict(
                tensors, all_gather_group=self.all_gather_group
            )
            # The metadata handle is reaped by the group coordinator.
            self._send_work = handles[1:]
            return

        i = self._next_slot
        self._next_slot = (i + 1) % len(self._slots)
        for handle in self._slot_work[i]:
            handle.wait()
        staged: dict[str, Any] = {}
        for key, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                staged[key] = value
                continue
            buffer = self._slots[i].get(key)
            if (
                buffer is None
                or buffer.shape[0] < value.shape[0]
                or buffer.shape[1:] != value.shape[1:]
                or buffer.dtype != value.dtype
            ):
                raise RuntimeError(
                    f"PP send {key!r} {tuple(value.shape)} {value.dtype} does not "
                    "fit the buffer sized from the profile run"
                )
            staged[key] = buffer[: value.shape[0]].copy_(value)
        handles = pp_group.isend_tensor_dict(
            staged, all_gather_group=self.all_gather_group
        )
        self._slot_work[i] = handles[1:]

    def recv(self, pp_group: GroupCoordinator) -> AsyncIntermediateTensors:
        tensor_dict, comm_handles, comm_postprocess = pp_group.irecv_tensor_dict(
            all_gather_group=self.all_gather_group
        )
        assert tensor_dict is not None
        return AsyncIntermediateTensors(
            tensor_dict, comm_handles=comm_handles, comm_postprocess=comm_postprocess
        )
