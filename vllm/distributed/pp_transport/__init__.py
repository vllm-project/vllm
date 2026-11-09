# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in alternatives to NCCL for sending a PP stage's intermediate tensors to
the next stage, selected by `VLLM_PP_TRANSPORT`."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class PPTransport(ABC):
    """One rank's side of its PP stage boundaries."""

    @abstractmethod
    def send(self, tensors: dict, pp_group: "GroupCoordinator") -> None:
        """Send this step's outputs to the next stage. The caller may
        overwrite `tensors` once this returns."""

    @abstractmethod
    def recv(self, pp_group: "GroupCoordinator") -> "IntermediateTensors":
        """Receive this step's inputs from the previous stage."""


def create_pp_transport(
    vllm_config: "VllmConfig", persistent: "IntermediateTensors | None"
) -> PPTransport | None:
    """Collective over the PP group. Returns None without PP.

    Args:
        vllm_config: The engine config.
        persistent: The model runner's persistent intermediate tensors (non-first
            ranks), which received tensors are copied into.

    """
    from vllm.distributed.parallel_state import get_pp_group

    pp_group = get_pp_group()
    if pp_group.world_size == 1:
        return None
    assert envs.VLLM_PP_TRANSPORT == "copy_engine"
    from vllm.distributed.pp_transport.copy_engine import CopyEngineTransport

    transport = CopyEngineTransport(
        vllm_config, pp_group, None if persistent is None else persistent.tensors
    )
    logger.info_once("PP transport: copy engines")
    return transport
