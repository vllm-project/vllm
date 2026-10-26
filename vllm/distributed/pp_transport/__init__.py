# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Moves a PP stage's intermediate tensors to the next stage.

Sampled tokens flowing back from the last stage go through ``PPHandler``.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


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


def create_pp_transport(
    vllm_config: "VllmConfig",
    persistent: dict[str, torch.Tensor] | None,
    profile_send: dict[str, Any] | None,
) -> PPTransport:
    """Collective over the PP group, at startup.

    Args:
        vllm_config: The engine config.
        persistent: The model runner's persistent receive tensors (non-first
            ranks), into which received tensors are copied.
        profile_send: The profile run's outputs (non-last ranks), the largest
            send this rank makes.

    """
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group
    from vllm.distributed.pp_transport.nccl import NcclTransport

    mode = envs.VLLM_PP_TRANSPORT
    if mode != "nccl":
        from vllm.distributed.pp_transport.copy_engine import CopyEngineTransport

        transport, reason = CopyEngineTransport.create(
            vllm_config, get_pp_group(), persistent, profile_send
        )
        if transport is not None:
            logger.info_once("PP transport: copy engines")
            return transport
        if mode == "copy_engine":
            raise RuntimeError(
                f"VLLM_PP_TRANSPORT=copy_engine is not usable here: {reason}"
            )
        logger.info_once("PP transport: NCCL (copy engines unavailable: %s)", reason)
    return NcclTransport(get_tp_group(), profile_send)
