# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Moves a PP stage's intermediate tensors to the next stage.

Sampled tokens flowing back from the last stage go through ``PPHandler``.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.sequence import IntermediateTensors


class PPTransport(ABC):
    """One rank's side of its PP stage boundaries: it sends to the next stage
    and receives from the previous one."""

    # Bytes this transport holds that memory profiling does not count.
    unprofiled_bytes: int = 0

    def before_forward(self) -> None:  # noqa: B027
        """Called before each forward pass, which may overwrite the previous
        pass's outputs."""

    @abstractmethod
    def send(self, tensors: dict[str, Any], pp_group: "GroupCoordinator") -> None:
        """Send this rank's outputs to the next stage."""

    @abstractmethod
    def recv(self, pp_group: "GroupCoordinator") -> "IntermediateTensors":
        """Receive the previous stage's outputs for this step."""


def create_pp_transport(profile_send: dict[str, Any] | None) -> PPTransport:
    """Called on every PP rank at startup.

    Args:
        profile_send: The profile run's outputs (non-last ranks), the largest
            send this rank makes.

    """
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.distributed.pp_transport.nccl import NcclTransport

    return NcclTransport(get_tp_group(), profile_send)
