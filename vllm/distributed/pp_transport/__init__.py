# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alternatives to NCCL for sending a PP stage's intermediate tensors to the next
stage, selected by `VLLM_PP_TRANSPORT`."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

# A received tensor's dtype and per-token shape.
RowSpec = tuple[torch.dtype, tuple[int, ...]]


def row_specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, RowSpec]:
    return {k: (t.dtype, tuple(t.shape[1:])) for k, t in tensors.items()}


def check_intermediate_tensors(
    tensors: Mapping[str, torch.Tensor], expected: Mapping[str, RowSpec]
) -> None:
    """Raise unless `tensors` match the receiving stage's persistent intermediate
    tensors in keys, dtypes and per-token shapes, which every PP transport
    requires. Otherwise the copy into them would silently drop, cast or
    broadcast values."""
    actual = row_specs(tensors)
    if actual != expected:
        raise RuntimeError(
            f"PP intermediate tensors {actual} do not match the receiving "
            f"stage's {dict(expected)}. Declare them in the model's "
            "make_empty_intermediate_tensors."
        )


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
    """Collective over the PP group. Returns None (send over NCCL) without PP,
    or under "auto" when copy engines are not usable on some rank; every rank
    makes the same choice.

    Args:
        vllm_config: The engine config.
        persistent: The model runner's persistent intermediate tensors (non-first
            ranks), which received tensors are copied into.

    """
    from vllm.distributed.parallel_state import get_pp_group

    pp_group = get_pp_group()
    if pp_group.world_size == 1:
        return None
    mode = envs.VLLM_PP_TRANSPORT
    assert mode in ("auto", "copy_engine")
    from vllm.distributed.pp_transport.copy_engine import (
        CopyEngineTransport,
        CopyEngineUnavailable,
    )

    try:
        transport = CopyEngineTransport(
            vllm_config, pp_group, None if persistent is None else persistent.tensors
        )
    except CopyEngineUnavailable as e:
        if mode == "copy_engine":
            raise
        logger.info_once("PP transport: NCCL (%s)", e)
        return None
    logger.info_once("PP transport: copy engines")
    return transport
