# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared NCCL initialization helpers for weight transfer engines.

The dense (`NCCLWeightTransferEngine`) and sparse
(`SparseNCCLWeightTransferEngine`) backends are independent engines that share
*only* their process-group initialization. That common logic lives here so the
sparse engine does not have to subclass the dense one.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from vllm.distributed.weight_transfer.base import WeightTransferInitInfo


@dataclass
class NCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for NCCL-based weight transfer backends."""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device,
) -> "PyNcclCommunicator":
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)


def worker_init_process_group(
    init_info: NCCLWeightTransferInitInfo,
    parallel_config: "ParallelConfig",
) -> "PyNcclCommunicator":
    """Create the trainer<->worker NCCL group on an inference worker.

    Computes a unique rank for this worker across all data-parallel groups and
    joins the stateless process group with the trainer.
    """
    # Calculate the global rank in the trainer-worker process group.
    # Must account for data parallel to get unique ranks across all workers.
    dp_rank = parallel_config.data_parallel_index
    world_size_per_dp = parallel_config.world_size  # TP * PP
    rank_within_dp = parallel_config.rank

    # Unique rank across all DP groups
    worker_rank = dp_rank * world_size_per_dp + rank_within_dp
    rank = worker_rank + init_info.rank_offset

    device = torch.accelerator.current_device_index()
    return stateless_init_process_group(
        init_info.master_address,
        init_info.master_port,
        rank,
        init_info.world_size,
        device=device,
    )


def trainer_init(
    init_info: NCCLWeightTransferInitInfo | dict,
) -> "PyNcclCommunicator":
    """
    Initialize NCCL process group for trainer-side weight transfer.

    The trainer is always rank 0 in the process group. Uses the current
    CUDA device (torch.accelerator.current_device_index()).

    Args:
        init_info: Either an NCCLWeightTransferInitInfo object or a dict with keys:
            - master_address: str
            - master_port: int
            - world_size: int

    Returns:
        PyNcclCommunicator for weight transfer.
    """
    if isinstance(init_info, dict):
        master_address = init_info["master_address"]
        master_port = init_info["master_port"]
        world_size = init_info["world_size"]
    else:
        master_address = init_info.master_address
        master_port = init_info.master_port
        world_size = init_info.world_size

    # Trainer is always rank 0
    device = torch.accelerator.current_device_index()
    return stateless_init_process_group(
        master_address,
        master_port,
        0,
        world_size,
        device,
    )
